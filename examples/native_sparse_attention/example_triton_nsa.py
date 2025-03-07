# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ruff: noqa
import torch
from typing import Optional, Union

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.common.utils import (prepare_chunk_indices, prepare_lens, prepare_token_indices)
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

from reference import naive_nsa, naive_nsa_simple


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1]],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(q, k, v, o_slc, o_swa, lse_slc, lse_swa, scale, block_indices,
                            block_counts, offsets, token_indices, T, H: tl.constexpr,
                            HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                            S: tl.constexpr, BS: tl.constexpr, WS: tl.constexpr, BK: tl.constexpr,
                            BV: tl.constexpr, USE_OFFSETS: tl.constexpr,
                            USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    # if USE_BLOCK_COUNTS:
    #     NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    # else:
    NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK),
                            (1, 0))
    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o_slc = tl.make_block_ptr(o_slc + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV),
                                (G, BV), (1, 0))
    p_lse_slc = lse_slc + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    # [G, BV]
    b_o_slc = tl.zeros([G, BV], dtype=tl.float32)

    b_m_slc = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc_slc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [BK, BS]
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            # [BS, BV]
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))
            # [G, BS]
            b_s_slc = tl.dot(b_q, b_k_slc)
            if i_t == 6:
                print("b_s_slc", b_s_slc)
            b_s_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_slc, float('-inf'))

            # [G]
            b_m_slc, b_mp_slc = tl.maximum(b_m_slc, tl.max(b_s_slc, 1)), b_m_slc
            b_r_slc = tl.exp(b_mp_slc - b_m_slc)
            # [G, BS]
            b_p_slc = tl.exp(b_s_slc - b_m_slc[:, None])
            # [G]
            b_acc_slc = b_acc_slc * b_r_slc + tl.sum(b_p_slc, 1)
            # [G, BV]
            b_o_slc = b_o_slc * b_r_slc[:, None] + tl.dot(b_p_slc.to(b_q.dtype), b_v_slc)

            b_mp_slc = b_m_slc
    b_o_slc = b_o_slc / b_acc_slc[:, None]
    b_m_slc += tl.log(b_acc_slc)
    tl.store(p_o_slc, b_o_slc.to(p_o_slc.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse_slc, b_m_slc.to(p_lse_slc.dtype.element_ty))


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d sequence indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `token_indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o, lse = parallel_nsa_fwd(
            q=q, k=k, v=v, block_indices=block_indices, block_size=block_size, scale=scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.block_indices = block_indices
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    window_size: int,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    import math
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    scale = 1.0 / math.sqrt(K)
    G = HQ // H
    BS = block_size
    WS = window_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)
    o_slc = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    o_swa = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device) if window_size > 0 else None
    lse_slc = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    lse_swa = torch.empty(B, T, HQ, dtype=torch.float, device=q.device) if window_size > 0 else None

    parallel_nsa_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o_slc=o_slc,
        o_swa=o_swa,
        lse_slc=lse_slc,
        lse_swa=lse_swa,
        scale=scale,
        block_indices=block_indices,
        block_counts=block_counts,
        offsets=offsets,
        token_indices=token_indices,
        T=T,
        H=H,
        HQ=HQ,
        G=G,
        K=K,
        V=V,
        S=S,
        BS=BS,
        WS=WS,
        BK=BK,
        BV=BV,
    )
    return o_slc, lse_slc, o_swa, lse_swa


if __name__ == "__main__":
    B, T, H, HQ, D, S, block_size, dtype, scale = 2, 64, 1, 16, 32, 1, 32, torch.float16, 0.1
    torch.random.manual_seed(0)
    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, (t // block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]

    block_counts = torch.randint(1, S + 1, (B, T, H), device='cuda')

    ref = naive_nsa_simple(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
    )

    tri, _, _, _ = parallel_nsa_fwd(
        q=q,
        k=k,
        v=v,
        block_indices=block_indices,
        block_size=block_size,
        window_size=0,
        block_counts=block_counts)

    print("tri", tri)
    print("ref", ref)

    torch.testing.assert_close(ref, tri, atol=1e-2, rtol=1e-2)
