/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"
#include "mqt/Dialect/QOracle/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qoracle {

#define GEN_PASS_DEF_SCHEDULEORACLE
#include "mqt/Dialect/QOracle/Transforms/Passes.h.inc"

namespace {

SmallVector<Value> getNestedOperands(Operation* op) {
  SmallVector<Value> operands;
  op->walk([&](Operation* nested) {
    for (auto operand : nested->getOperands()) {
      operands.push_back(operand);
    }
  });
  return operands;
}

/// Chooses the order of the operations of one block.
///
/// The scheduler follows which values the lowering will keep in qubits as it
/// places operations: consumed aliases, loop iteration arguments, the results
/// of updates in place, and values that get a helper register. An update may
/// overwrite such an operand once nothing unplaced still needs it. A temporary
/// with one reader has no qubits of its own; the lowering computes it in the
/// step of its reader, so its reads happen there.
///
/// The original order is kept, except for an update that is blocked: every
/// operand it could overwrite is still needed. Then a ready operation that
/// still needs one of those operands and is not blocked itself goes first, so
/// that the update can overwrite the operand afterwards. If there is none, the
/// update goes as it is. The result is deterministic.
class Scheduler {
public:
  Scheduler(Block& block, OracleOp oracle) : block_(block), oracle_(oracle) {
    for (auto argument : block.getArguments()) {
      if (oracle.isConsumedAlias(argument)) {
        stored_.insert(argument);
      }
      auto forOp = dyn_cast_or_null<scf::ForOp>(block.getParentOp());
      if (forOp && argument != forOp.getInductionVar()) {
        stored_.insert(argument);
      }
    }
    for (auto& op : block) {
      unplaced_.insert(&op);
    }
    unplaced_.erase(block.getTerminator());
  }

  /// Place the next operation of `unplaced`, in their original order.
  Operation* placeNext(ArrayRef<Operation*> unplaced) {
    for (auto* op : unplaced) {
      if (!isReady(op)) {
        continue;
      }
      auto candidates = storedCandidates(op);
      if (candidates.empty() || canOverwrite(op, candidates)) {
        return place(op);
      }
      for (auto* reader : unplaced) {
        if (reader == op || !isReady(reader) ||
            llvm::none_of(candidates, [&](Value value) {
              return needs(reader, value, op);
            })) {
          continue;
        }
        auto readerCandidates = storedCandidates(reader);
        if (readerCandidates.empty() ||
            canOverwrite(reader, readerCandidates)) {
          return place(reader);
        }
      }
      return place(op);
    }
    llvm_unreachable("a block without ready operations");
  }

private:
  Block& block_;
  OracleOp oracle_;
  DenseSet<Value> stored_;
  DenseSet<Operation*> unplaced_;

  bool isReady(Operation* op) const {
    for (auto operand : getNestedOperands(op)) {
      auto* definingOp = operand.getDefiningOp();
      if (definingOp != nullptr && unplaced_.contains(definingOp)) {
        return false;
      }
    }
    return true;
  }

  /// The operands in qubits that `op` would overwrite in place. A loop
  /// overwrites all of its initial values at once.
  SmallVector<Value> storedCandidates(Operation* op) const {
    SmallVector<Value> candidates;
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      llvm::append_range(candidates, forOp.getInitArgs());
      return candidates;
    }
    auto source = rotationSource(op);
    if (source && stored_.contains(source)) {
      candidates.push_back(source);
      return candidates;
    }
    if (!isa<arith::AddIOp, arith::SubIOp, arith::XOrIOp>(op)) {
      return candidates;
    }
    for (auto operand : op->getOperands()) {
      if (stored_.contains(operand)) {
        candidates.push_back(operand);
      }
    }
    return candidates;
  }

  bool canOverwrite(Operation* op, ArrayRef<Value> candidates) {
    if (isa<scf::ForOp>(op)) {
      return llvm::none_of(
          candidates, [&](Value value) { return isNeededLater(value, op); });
    }
    return llvm::any_of(candidates,
                        [&](Value value) { return !isNeededLater(value, op); });
  }

  /// Place `op` and record whether its result is kept in qubits.
  Operation* place(Operation* op) {
    auto candidates = storedCandidates(op);
    auto overwrites = !candidates.empty() && canOverwrite(op, candidates);
    unplaced_.erase(op);
    for (auto result : op->getResults()) {
      if (overwrites || isa<scf::ForOp>(op) || isa<arith::AddIOp>(op) ||
          (!result.hasOneUse() && !result.use_empty() && !rotationSource(op) &&
           !isa<arith::ShLIOp, arith::ShRUIOp>(op))) {
        stored_.insert(result);
      }
    }
    return op;
  }

  /// The value that `op` rotates, if `op` is `ori(shli x, a), (shrui x, b))`.
  static Value rotationSource(Operation* op) {
    auto orOp = dyn_cast<arith::OrIOp>(op);
    if (!orOp) {
      return nullptr;
    }
    auto* lhs = orOp.getLhs().getDefiningOp();
    auto* rhs = orOp.getRhs().getDefiningOp();
    if (!isa_and_nonnull<arith::ShLIOp, arith::ShRUIOp>(lhs) ||
        !isa_and_nonnull<arith::ShLIOp, arith::ShRUIOp>(rhs) ||
        lhs->getName() == rhs->getName() ||
        lhs->getOperand(0) != rhs->getOperand(0)) {
      return nullptr;
    }
    return lhs->getOperand(0);
  }

  /// Whether the unplaced `reader` needs `value`, directly or through a placed
  /// temporary, which the lowering computes in the step of its reader.
  bool needs(Operation* reader, Value value, Operation* op) const {
    for (auto* user : value.getUsers()) {
      auto* ancestor = block_.findAncestorOpInBlock(*user);
      if (ancestor == nullptr || ancestor == op) {
        continue;
      }
      if (ancestor == reader) {
        return true;
      }
      if (!unplaced_.contains(ancestor) && ancestor != block_.getTerminator() &&
          llvm::any_of(ancestor->getResults(), [&](Value result) {
            return !stored_.contains(result) && needs(reader, result, op);
          })) {
        return true;
      }
    }
    return false;
  }

  /// Whether something other than `op` still needs `value`: an unplaced
  /// operation, the terminator, an operation outside the block, or a placed
  /// temporary that something unplaced needs.
  bool isNeededLater(Value value, Operation* op) const {
    for (auto* user : value.getUsers()) {
      auto* ancestor = block_.findAncestorOpInBlock(*user);
      if (ancestor == nullptr || ancestor == block_.getTerminator()) {
        return true;
      }
      if (ancestor == op) {
        continue;
      }
      if (unplaced_.contains(ancestor)) {
        return true;
      }
      if (llvm::any_of(ancestor->getResults(), [&](Value result) {
            return !stored_.contains(result) && isNeededLater(result, op);
          })) {
        return true;
      }
    }
    return false;
  }
};

void scheduleBlock(Block& block, OracleOp oracle) {
  SmallVector<Operation*> unplaced;
  for (auto& op : block.without_terminator()) {
    unplaced.push_back(&op);
  }

  Scheduler scheduler(block, oracle);
  SmallVector<Operation*> order;
  while (!unplaced.empty()) {
    auto* next = scheduler.placeNext(unplaced);
    order.push_back(next);
    llvm::erase(unplaced, next);
  }

  auto* terminator = block.getTerminator();
  for (auto* op : order) {
    op->moveBefore(terminator);
  }
}

void scheduleOps(OracleOp oracle) {
  SmallVector<Block*> blocks;
  oracle->walk([&](Block* block) { blocks.push_back(block); });
  for (auto* block : blocks) {
    scheduleBlock(*block, oracle);
  }
}

struct ScheduleOracle final : impl::ScheduleOracleBase<ScheduleOracle> {
protected:
  void runOnOperation() override {
    getOperation()->walk([&](OracleOp oracle) { scheduleOps(oracle); });
  }
};

} // namespace

} // namespace mlir::qoracle
