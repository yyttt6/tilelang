/*!
 * \file legalize_safe_memory_access.cc
 * \brief legalize safe memory access
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/builtin.h"
#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

// Helper class to find leaf For nodes in a given IR
class LeafForFinder : public StmtVisitor {
public:
  std::vector<For> leaf_for_nodes;

private:
  void VisitStmt_(const ForNode *op) final {
    has_child_for_ = false;
    bool parent_has_child_for = parent_has_child_for_;
    parent_has_child_for_ = false;

    StmtVisitor::VisitStmt(op->body);

    if (!has_child_for_) {
      leaf_for_nodes.push_back(GetRef<For>(op));
    }

    parent_has_child_for_ = parent_has_child_for;
    parent_has_child_for_ = true;
  }

private:
  bool has_child_for_ = false;
  bool parent_has_child_for_ = false;
};

// We will create a visitor to check BufferLoad and BufferStore nodes
// within this loop body. This visitor will:
// 1. Identify BufferLoad and BufferStore nodes.
// 2. Check if the buffer is in global scope.
// 3. For each index, compare against the buffer's shape.
//    If the index might exceed the shape (upper bound too large),
//    log a warning or handle accordingly.
struct GlobalMemChecker : public StmtExprVisitor {

  GlobalMemChecker(arith::Analyzer *analyzer) : analyzer_(analyzer) {}
  void VisitExpr_(const BufferLoadNode *op) final {
    // Check if the buffer is in global scope
    if (IsGlobalBuffer(op->buffer)) {
      CheckBufferIndices(op->buffer, op->indices, /*is_load=*/true);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    // Check if the buffer is in global scope
    if (IsGlobalBuffer(op->buffer)) {
      CheckBufferIndices(op->buffer, op->indices, /*is_load=*/false);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  // Helper function to determine if a buffer is global
  bool IsGlobalBuffer(const Buffer &buffer) {
    // The storage scope is often encoded in the buffer->data var name or
    // associated attributes. In typical TVM IR, global buffers have scope
    // "global". Here we assume a helper function GetPtrStorageScope is
    // available. If not, you might need to parse buffer->data->name_hint or
    // associated attributes.
    String scope = buffer.scope();
    return scope == "global";
  }

  // Check each index against the buffer shape dimensions
  void CheckBufferIndices(const Buffer &buffer, const Array<PrimExpr> &indices,
                          bool is_load) {
    // Ensure indices count matches buffer dimension
    if (indices.size() != buffer->shape.size()) {
      LOG(WARNING) << "Buffer access dimension mismatch: indices size ("
                   << indices.size() << ") vs. shape size ("
                   << buffer->shape.size() << ")";
      return;
    }

    for (size_t i = 0; i < indices.size(); i++) {
      PrimExpr index = indices[i];
      PrimExpr shape_dim = buffer->shape[i];

      bool has_variable = false;
      PostOrderVisit(index, [&](const ObjectRef &obj) {
        if (const VarNode *v = obj.as<VarNode>()) {
          has_variable = true;
        }
      });
      if (!has_variable) {
        continue;
      }

      // We want to check if index < shape_dim can be proven.
      // If analyzer->CanProve(index < shape_dim) returns false,
      // it means we cannot prove the access is within bounds.
      PrimExpr upper_bound_cond = index < shape_dim;
      if (!analyzer_->CanProve(upper_bound_cond,
                               arith::ProofStrength::kSymbolicBound)) {
        _conditions.push_back(upper_bound_cond);
      }
      // Check if index >= 0 can be proven.
      PrimExpr lower_bound_cond = index >= 0;
      if (!analyzer_->CanProve(lower_bound_cond,
                               arith::ProofStrength::kSymbolicBound)) {
        _conditions.push_back(lower_bound_cond);
      }
    }
  }

  Array<PrimExpr> GetConditions() { return _conditions; }

private:
  Array<PrimExpr> _conditions;
  arith::Analyzer *analyzer_;
};

class SafeMemorysRewriter : public StmtExprMutator {
  arith::Analyzer *analyzer_;

public:
  explicit SafeMemorysRewriter(Map<Buffer, PrimExpr> annotated_padding_map,
                               arith::Analyzer *analyzer)
      : annotated_padding_map_(annotated_padding_map), analyzer_(analyzer) {}

private:
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    // Check if the buffer is in global scope
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));

    GlobalMemChecker checker(analyzer_);
    checker(store);
    Array<PrimExpr> conditions = checker.GetConditions();

    // Skip boundary check if the store value is an IfThenElse
    if (const IfThenElseNode *if_node = store->value.as<IfThenElseNode>()) {
      if (conditions.size() > 0) {
        LOG(WARNING)
            << "Skipping boundary check for store with IfThenElse value: "
            << store->value
            << "\nAs manual boundary check detected, potential out-of-bounds "
               "access may occur."
            << "\nAuto detect boundaries are " << conditions;
        return store;
      }
      return store;
    }

    if (conditions.size() == 0) {
      return store;
    }

    auto value = store->value;
    if (IsGlobalBuffer(store->buffer)) {
      Stmt store_with_conditions = store;
      for (auto cond : conditions) {
        store_with_conditions = IfThenElse(cond, store_with_conditions);
      }
      return store_with_conditions;
    } else if (isSharedBuffer(store->buffer)) {
      PrimExpr value = store->value;
      for (auto cond : conditions) {
        ICHECK(cond.dtype() == DataType::Bool(1))
            << "condition is not a boolean: " << cond;
        value = if_then_else(cond, value, GetPadding(store->buffer));
      }
      store.CopyOnWrite()->value = value;
      return store;
    } else if (IsLocalBuffer(store->buffer)) {
      PrimExpr value = store->value;
      for (auto cond : conditions) {
        ICHECK(cond.dtype() == DataType::Bool(1))
            << "condition is not a boolean: " << cond;
        value = if_then_else(cond, value, GetPadding(store->buffer));
      }
      store.CopyOnWrite()->value = value;
      return store;
    } else {
      LOG(FATAL) << "Check store buffer: " << store->buffer
                 << " is not a global or shared or local buffer";
    }

    return store;
  }

  // Handle Call Nodes
  // For example
  // T.call_extern("handle", "atomicAddx2", T.address_of(C),
  // T.address_of(C_shared))
  Stmt VisitStmt_(const EvaluateNode *op) final {
    auto evaluate = Downcast<Evaluate>(StmtExprMutator::VisitStmt_(op));
    if (const CallNode *call_op = op->value.as<CallNode>()) {
      auto call = Downcast<Call>(evaluate->value);
      if (call->op == builtin::call_extern()) {
        GlobalMemChecker checker(analyzer_);
        checker(call);
        Array<PrimExpr> conditions = checker.GetConditions();

        if (conditions.size() == 0) {
          return evaluate;
        }

        Stmt evaluate_with_conditions = evaluate;
        for (auto cond : conditions) {
          evaluate_with_conditions = IfThenElse(cond, evaluate_with_conditions);
        }
        return evaluate_with_conditions;
      }
    }

    return evaluate;
  }

  bool IsLocalBuffer(const Buffer &buffer) {
    String scope = buffer.scope();
    return scope == "local" || scope == "local.fragment";
  }

  bool isSharedBuffer(const Buffer &buffer) {
    String scope = buffer.scope();
    return scope == "shared" || scope == "shared.dyn";
  }

  bool IsGlobalBuffer(const Buffer &buffer) {
    String scope = buffer.scope();
    return scope == "global";
  }
  // Get the padding of the buffer
  PrimExpr GetPadding(const Buffer &buffer) {
    if (annotated_padding_map_.count(buffer)) {
      return annotated_padding_map_[buffer];
    }
    return make_zero(buffer->dtype);
  }

  Map<Buffer, PrimExpr> annotated_padding_map_;
};

// Class to legalize safe memory access by transforming them appropriately
class SafeMemoryLegalizer : IRMutatorWithAnalyzer {
public:
  // Static method to substitute and transform the given PrimFunc
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    // Create an instance of the legalizer with the analyzer
    SafeMemoryLegalizer substituter(&analyzer);
    // Get a mutable copy of the function node
    PrimFuncNode *fptr = f.CopyOnWrite();
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  // Constructor initializing the base class with the analyzer
  SafeMemoryLegalizer(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  // Override the VisitStmt_ method to handle ForNode (loop statements)
  Stmt VisitStmt_(const ForNode *op) final {
    // Visit and potentially modify the loop node
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto has_inner_loop = HasInnerLoop(for_node->body);
    if (!has_inner_loop) {
      SafeMemorysRewriter rewriter(annotated_padding_map_, analyzer_);
      for_node.CopyOnWrite()->body = rewriter(for_node->body);
      // // Detect Buffer Load Node in the loop body, collect the indices and
      // buffer size

      // // Run the checker on the loop body
      // GlobalMemChecker checker(analyzer_);
      // checker(for_node->body);
      // Array<PrimExpr> conditions = checker.GetConditions();
      // auto body = for_node->body;
      // // Note that we might have duplicate conditions
      // // Which will be optimized by simplify pass
      // // Replace the loop body with the new body
      // for (auto cond : conditions) {
      //   body = IfThenElse(cond, body);
      // }
      // for_node.CopyOnWrite()->body = body;
      return std::move(for_node);
    }

    // Visit a For Node
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    if (op->annotations.count(attr::kPaddingMap)) {
      auto map = op->annotations.Get(attr::kPaddingMap)
                     ->as<Map<Var, PrimExpr>>()
                     .value();
      for (const auto &[var, padding] : map) {
        ICHECK(buffer_data_to_buffer_.count(var))
            << "buffer " << var << " is not found in the block "
            << buffer_data_to_buffer_;
        auto buffer = buffer_data_to_buffer_[var];
        annotated_padding_map_.Set(buffer, padding);
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  static bool HasInnerLoop(const Stmt &stmt) {
    LeafForFinder finder;
    finder(stmt);
    return finder.leaf_for_nodes.size() > 0;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, PrimExpr> annotated_padding_map_;
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass LegalizeSafeMemoryAccess() {
  using namespace tir::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    bool disable_safe_memory_legalize =
        ctx->GetConfig<Bool>(kDisableSafeMemoryLegalize, Bool(false)).value();
    if (disable_safe_memory_legalize) {
      return f;
    }
    return SafeMemoryLegalizer::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSafeMemoryAccess", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSafeMemoryAccess",
                        LegalizeSafeMemoryAccess);
});

} // namespace tl
} // namespace tvm
