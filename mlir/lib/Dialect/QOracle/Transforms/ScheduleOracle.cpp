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

bool isReady(Operation* op, ArrayRef<Operation*> unplaced) {
  for (auto operand : getNestedOperands(op)) {
    auto* definingOp = operand.getDefiningOp();
    if (definingOp == nullptr) {
      continue;
    }
    if (llvm::is_contained(unplaced, definingOp)) {
      return false;
    }
  }
  return true;
}

SmallVector<Value> operandsToOverwrite(Operation* op, OracleOp oracle) {
  if (!isa<arith::AddIOp, arith::SubIOp, arith::XOrIOp>(op)) {
    return {};
  }
  SmallVector<Value> canOverwrite;
  for (auto operand : op->getOperands()) {
    if (oracle.ownsValue(operand) && !matchPattern(operand, m_Constant())) {
      canOverwrite.push_back(operand);
    }
  }
  return canOverwrite;
}

// An operation that can overwrite an operand waits while an operation that
// cannot is ready, so that the last reader of a value is the one to overwrite
// it.
bool mustWait(Operation* op, OracleOp oracle) {
  return !operandsToOverwrite(op, oracle).empty();
}

Operation* chooseNext(ArrayRef<Operation*> unplaced, OracleOp oracle) {
  Operation* firstReady = nullptr;
  for (auto* op : unplaced) {
    if (!isReady(op, unplaced)) {
      continue;
    }
    if (firstReady == nullptr) {
      firstReady = op;
    }
    if (!mustWait(op, oracle)) {
      return op;
    }
  }
  return firstReady;
}

void scheduleBlock(Block& block, OracleOp oracle) {
  SmallVector<Operation*> unplaced;
  for (auto& op : block.without_terminator()) {
    unplaced.push_back(&op);
  }

  SmallVector<Operation*> order;
  while (!unplaced.empty()) {
    auto* next = chooseNext(unplaced, oracle);
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
