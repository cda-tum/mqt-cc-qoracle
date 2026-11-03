/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Conversion/QOracleToQC/QOracleToQC.h"

#include "mqt/Conversion/ConversionUtils.h"
#include "mqt/Dialect/CBit/IR/CBitDialect.h"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"
#include "mqt/Dialect/QTensor/IR/QTensorDialect.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir {

using namespace qoracle;
using namespace qc;

#define GEN_PASS_DEF_QORACLETOQC
#include "mqt/Conversion/QOracleToQC/QOracleToQC.h.inc"

namespace {

struct UpdateStep {
  Operation* updateOp;
  Value overwriteValue;
  Value otherValue;
  SmallVector<Operation*> temporaries;
  std::vector<UpdateStep> body;
};

struct RecordedGate {
  SmallVector<Value> controls;
  Value target;
};

SmallVector<Operation*>
reachableTemporaries(Value source, DenseMap<Value, Value>& containers) {
  SmallVector<Operation*> temporaries;
  SmallVector<std::pair<Value, bool>> toVisit{{source, false}};
  DenseSet<Operation*> collected;

  while (!toVisit.empty()) {
    auto [current, revisited] = toVisit.pop_back_val();
    if (containers.contains(current)) {
      continue;
    }
    auto* defOp = current.getDefiningOp();
    if (defOp == nullptr || collected.contains(defOp)) {
      continue;
    }

    if (revisited) {
      // second visit, collect temporary op.
      collected.insert(defOp);
      temporaries.push_back(defOp);
      continue;
    }

    // first visit, continue with operands and then revisit in post-order.
    toVisit.emplace_back(current, true);
    for (auto operand : llvm::reverse(defOp->getOperands())) {
      toVisit.emplace_back(operand, false);
    }
  }

  return temporaries;
}

bool isLastReader(Operation* op, Value value) {
  auto* block = op->getBlock();
  return llvm::all_of(value.getUsers(), [&](Operation* user) {
    auto* ancestor = block->findAncestorOpInBlock(*user);
    return ancestor == op || ancestor->isBeforeInBlock(op);
  });
}

std::pair<Value, Value> operandToOverwrite(Operation* op,
                                           DenseMap<Value, Value>& containers) {
  if (!isa<arith::XOrIOp, arith::AddIOp>(op)) {
    return {nullptr, nullptr};
  }

  size_t overwriteIndex = 2;
  for (auto [i, value] : llvm::enumerate(op->getOperands())) {
    if (!containers.contains(value)) {
      continue;
    }
    if (value.getParentBlock() != op->getBlock() || !isLastReader(op, value)) {
      continue;
    }
    overwriteIndex = i;
    break;
  }
  if (overwriteIndex > 1) {
    return {nullptr, nullptr};
  }

  auto toOverwrite = op->getOperand(overwriteIndex);
  auto other = op->getOperand(1 ^ overwriteIndex);
  if (toOverwrite == other) {
    return {nullptr, nullptr};
  }
  return {toOverwrite, other};
}

FailureOr<SmallVector<UpdateStep>>
planBlock(Block& block, OracleOp oracle, DenseMap<Value, Value>& containers);

FailureOr<UpdateStep> planLoop(scf::ForOp forOp, OracleOp oracle,
                               DenseMap<Value, Value>& containers) {
  if (!forOp.getInductionVar().use_empty()) {
    oracle.emitOpError(
        "cannot lower qoracle because a loop uses its induction variable");
    return failure();
  }
  for (auto [init, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (!containers.contains(init)) {
      oracle.emitOpError(
          "cannot lower qoracle because a loop initial value is not stored in "
          "consumed qubits");
      return failure();
    }
    if (!isLastReader(forOp, init)) {
      oracle.emitOpError(
          "cannot lower qoracle because a loop initial value is read after "
          "the loop");
      return failure();
    }
    containers[iterArg] = containers[init];
  }

  llvm::SetVector<Value> captured;
  getUsedValuesDefinedAbove(forOp.getRegion(), captured);
  for (auto value : captured) {
    if (llvm::is_contained(forOp.getInitArgs(), value)) {
      oracle.emitOpError(
          "cannot lower qoracle because a loop reads a value that it "
          "overwrites");
      return failure();
    }
    auto isAlias = llvm::is_contained(oracle.getRegion().getArguments(), value);
    auto isContainer = containers.contains(value);
    auto isConstant = matchPattern(value, m_Constant());
    if (!isAlias && !isContainer && !isConstant) {
      oracle.emitOpError(
          "cannot lower qoracle because a loop reads a value that is neither "
          "stored in qubits nor a constant");
      return failure();
    }
  }

  auto body = planBlock(forOp.getBodyRegion().front(), oracle, containers);
  if (failed(body)) {
    return failure();
  }

  for (auto [i, bodyYield, iterArg] :
       llvm::enumerate(forOp.getBody()->getTerminator()->getOperands(),
                       forOp.getRegionIterArgs())) {
    if (containers[bodyYield] != containers[iterArg]) {
      oracle.emitOpError("cannot lower qoracle because the loop yields value ")
          << i << " into other qubits than its iteration argument";
      return failure();
    }
  }

  for (auto [result, init] :
       llvm::zip_equal(forOp.getResults(), forOp.getInitArgs())) {
    containers[result] = containers[init];
  }

  UpdateStep step{
      .updateOp = forOp,
      .overwriteValue = nullptr,
      .otherValue = nullptr,
      .temporaries = {},
      .body = std::vector<UpdateStep>(std::make_move_iterator(body->begin()),
                                      std::make_move_iterator(body->end())),
  };
  return step;
}

FailureOr<SmallVector<UpdateStep>>
planBlock(Block& block, OracleOp oracle, DenseMap<Value, Value>& containers) {
  SmallVector<UpdateStep> updateSteps;
  for (auto& op : block.without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto step = planLoop(forOp, oracle, containers);
      if (failed(step)) {
        return failure();
      }
      updateSteps.push_back(std::move(*step));
      continue;
    }

    auto [overwriteValue, other] = operandToOverwrite(&op, containers);
    auto temporaries = reachableTemporaries(other, containers);
    if (!overwriteValue) {
      continue;
    }

    for (auto* tempOp : temporaries) {
      if (llvm::is_contained(tempOp->getOperands(), overwriteValue)) {
        oracle.emitOpError(
            "cannot lower qoracle because a temporary depends on "
            "the value to be overwritten");
        return failure();
      }
    }

    updateSteps.emplace_back(&op, overwriteValue, other, temporaries);
    containers[op.getResult(0)] = containers[overwriteValue];
  }

  DenseSet<Value> overwritten;
  for (auto& step : updateSteps) {
    for (auto* tempOp : step.temporaries) {
      for (auto operand : tempOp->getOperands()) {
        if (overwritten.contains(operand)) {
          oracle.emitOpError(
              "cannot lower qoracle because a temporary depends on a value "
              "that is overwritten in a previous step");
          return failure();
        }
      }
    }
    overwritten.insert(step.overwriteValue);
  }

  return updateSteps;
}

FailureOr<SmallVector<UpdateStep>> plan(OracleOp oracle) {
  DenseMap<Value, Value> containers;
  for (auto consumed : oracle.getConsumedAliases()) {
    containers[consumed] = consumed;
  }

  llvm::SetVector<Value> captured;
  getUsedValuesDefinedAbove(oracle.getRegion(), captured);
  for (auto value : captured) {
    if (!matchPattern(value, m_Constant())) {
      oracle.emitOpError("cannot lower qoracle because it captures a "
                         "classical value that is not a constant");
      return failure();
    }
  }

  auto steps = planBlock(oracle.getRegion().front(), oracle, containers);
  if (failed(steps)) {
    return failure();
  }
  auto updateSteps = std::move(*steps);

  qoracle::YieldOp yield =
      llvm::cast<qoracle::YieldOp>(oracle.getRegion().front().getTerminator());
  for (auto [i, operand] : llvm::enumerate(yield.getOperands())) {
    if (containers[operand] != oracle.getConsumedAliases()[i]) {
      oracle.emitOpError("cannot lower qoracle because the yielded value ")
          << i << " does not match the corresponding consumed alias";
      return failure();
    }
  }

  return success(updateSteps);
}

bool validRegion(Block* block) {
  if (block
          ->walk([&](Operation* op) {
            if (!isa<arith::ConstantOp, arith::XOrIOp, arith::AndIOp,
                     arith::AddIOp, arith::ExtUIOp, arith::TruncIOp,
                     qoracle::YieldOp, scf::ForOp, scf::YieldOp>(op)) {
              return WalkResult::interrupt();
            }
            if (op->getNumOperands() == 2 &&
                op->getOperand(0) == op->getOperand(1)) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return false;
  }

  return true;
}

void emitGate(Location loc, ArrayRef<Value> controls, Value target,
              PatternRewriter& rewriter) {
  if (!controls.empty()) {
    qc::CtrlOp::create(rewriter, loc, controls, {target}, [&](ValueRange args) {
      qc::XOp::create(rewriter, loc, args[0]);
    });
  } else {
    qc::XOp::create(rewriter, loc, target);
  }
}

void emitGateAndRecord(Location loc, ArrayRef<Value> controls, Value target,
                       SmallVector<RecordedGate>& records,
                       PatternRewriter& rewriter) {
  emitGate(loc, controls, target, rewriter);
  records.emplace_back(llvm::to_vector(controls), target);
}

void applyXorIInPlace(ArrayRef<Value> overwrite, ArrayRef<Value> other,
                      Location loc, PatternRewriter& rewriter, bool record,
                      SmallVector<RecordedGate>& records) {
  auto bitWidth = overwrite.size();
  for (size_t i = 0; i < bitWidth; i++) {
    if (record) {
      emitGateAndRecord(loc, {other[i]}, overwrite[i], records, rewriter);
    } else {
      emitGate(loc, {other[i]}, overwrite[i], rewriter);
    }
  }
}

void applyXorIInPlace(ArrayRef<Value> overwrite, ArrayRef<Value> other,
                      Location loc, PatternRewriter& rewriter) {
  SmallVector<RecordedGate> records;
  applyXorIInPlace(overwrite, other, loc, rewriter, false, records);
}

void applyAddIInPlace(ArrayRef<Value> overwrite, ArrayRef<Value> other,
                      Location loc, PatternRewriter& rewriter, bool record,
                      SmallVector<RecordedGate>& records) {
  auto bitWidth = overwrite.size();
  for (size_t i = 0; i < bitWidth; i++) {
    auto control = other[i];
    for (size_t j = bitWidth - 1; j > i; j--) {
      SmallVector<Value> args(overwrite.drop_front(i).take_front(j - i));
      args.push_back(control);
      if (record) {
        emitGateAndRecord(loc, args, overwrite[j], records, rewriter);
      } else {
        emitGate(loc, args, overwrite[j], rewriter);
      }
    }
    if (record) {
      emitGateAndRecord(loc, {control}, overwrite[i], records, rewriter);
    } else {
      emitGate(loc, {control}, overwrite[i], rewriter);
    }
  }
}

void applyAddIInPlace(ArrayRef<Value> overwrite, ArrayRef<Value> other,
                      Location loc, PatternRewriter& rewriter) {
  SmallVector<RecordedGate> records;
  applyAddIInPlace(overwrite, other, loc, rewriter, false, records);
}

Value allocateHelper(Operation* outerLoop, Location loc,
                     PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (outerLoop != nullptr) {
    rewriter.setInsertionPoint(outerLoop);
  }
  return qc::AllocOp::create(rewriter, loc);
}

void emitTemporary(Operation* op,
                   DenseMap<Value, SmallVector<Value>>& storageMap,
                   SmallVector<RecordedGate>& records, Operation* outerLoop,
                   SmallVector<Value>& allocated, PatternRewriter& rewriter) {
  auto res = op->getResult(0);
  auto bitWidth = res.getType().getIntOrFloatBitWidth();
  auto loc = op->getLoc();
  SmallVector<Value> helpers;

  if (auto extuiOp = dyn_cast<arith::ExtUIOp>(op)) {
    auto operand = extuiOp.getOperand();
    auto oldWidth = operand.getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < oldWidth; ++i) {
      helpers.push_back(storageMap.at(operand)[i]);
    }
    for (size_t i = oldWidth; i < bitWidth; ++i) {
      auto qubit = allocateHelper(outerLoop, loc, rewriter);
      helpers.push_back(qubit);
      allocated.push_back(qubit);
    }
    storageMap[res] = std::move(helpers);
    return;
  }

  if (auto truncOp = dyn_cast<arith::TruncIOp>(op)) {
    auto operand = truncOp.getOperand();
    for (size_t i = 0; i < bitWidth; ++i) {
      helpers.push_back(storageMap.at(operand)[i]);
    }
    storageMap[res] = std::move(helpers);
    return;
  }

  for (size_t i = 0; i < bitWidth; ++i) {
    auto qubit = allocateHelper(outerLoop, loc, rewriter);
    helpers.push_back(qubit);
    allocated.push_back(qubit);
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    auto value = cast<IntegerAttr>(constOp.getValue()).getValue();
    for (size_t i = 0; i < bitWidth; ++i) {
      if (value[i]) {
        emitGateAndRecord(loc, {}, helpers[i], records, rewriter);
      }
    }
  } else if (auto xoriOp = dyn_cast<arith::XOrIOp>(op)) {
    auto lhs = xoriOp.getOperand(0);
    auto rhs = xoriOp.getOperand(1);
    applyXorIInPlace(helpers, storageMap.at(lhs), loc, rewriter, true, records);
    applyXorIInPlace(helpers, storageMap.at(rhs), loc, rewriter, true, records);
  } else if (auto andiOp = dyn_cast<arith::AndIOp>(op)) {
    auto lhs = andiOp.getOperand(0);
    auto rhs = andiOp.getOperand(1);
    for (size_t i = 0; i < bitWidth; ++i) {
      emitGateAndRecord(loc, {storageMap.at(lhs)[i], storageMap.at(rhs)[i]},
                        helpers[i], records, rewriter);
    }
  } else if (auto addiOp = dyn_cast<arith::AddIOp>(op)) {
    auto lhs = addiOp.getOperand(0);
    auto rhs = addiOp.getOperand(1);
    for (size_t i = 0; i < bitWidth; ++i) {
      emitGateAndRecord(loc, {storageMap.at(lhs)[i]}, helpers[i], records,
                        rewriter);
    }
    applyAddIInPlace(helpers, storageMap.at(rhs), loc, rewriter, true, records);
  } else {
    llvm_unreachable(
        "cannot lower qoracle because the operation is not supported");
  }
  storageMap[res] = std::move(helpers);
}

void emitSteps(OracleOp op, ArrayRef<UpdateStep> updateSteps,
               Operation* outerLoop, Location loc,
               DenseMap<Value, SmallVector<Value>>& storageMap,
               SmallVector<Value>& loopHelpers, PatternRewriter& rewriter);

void emitForLoop(OracleOp op, scf::ForOp forOp, Operation* outerLoop,
                 const UpdateStep& step, Location loc,
                 DenseMap<Value, SmallVector<Value>>& storageMap,
                 SmallVector<Value>& loopHelpers, PatternRewriter& rewriter) {
  for (auto [init, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    auto qubits = storageMap.at(init);
    storageMap[iterArg] = qubits;
  }

  auto lowerBound = arith::ConstantIndexOp::create(
      rewriter, loc, *getConstantIntValue(forOp.getLowerBound()));
  auto upperBound = arith::ConstantIndexOp::create(
      rewriter, loc, *getConstantIntValue(forOp.getUpperBound()));
  auto stepValue = arith::ConstantIndexOp::create(
      rewriter, loc, *getConstantIntValue(forOp.getStep()));

  auto newForOp =
      scf::ForOp::create(rewriter, loc, lowerBound, upperBound, stepValue);
  rewriter.setInsertionPoint(newForOp.getBody()->getTerminator());
  emitSteps(op, step.body, !outerLoop ? newForOp : outerLoop, loc, storageMap,
            loopHelpers, rewriter);
  rewriter.setInsertionPointAfter(newForOp);

  for (auto [init, result] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getResults())) {
    auto qubits = storageMap.at(init);
    storageMap[result] = qubits;
  }

  if (!outerLoop) {
    for (auto qubit : loopHelpers) {
      qc::DeallocOp::create(rewriter, loc, qubit);
    }
    loopHelpers.clear();
  }
}

void emitSteps(OracleOp op, ArrayRef<UpdateStep> updateSteps,
               Operation* outerLoop, Location loc,
               DenseMap<Value, SmallVector<Value>>& storageMap,
               SmallVector<Value>& loopHelpers, PatternRewriter& rewriter) {
  for (const auto& step : updateSteps) {
    if (auto forOp = dyn_cast<scf::ForOp>(step.updateOp)) {
      emitForLoop(op, forOp, outerLoop, step, loc, storageMap, loopHelpers,
                  rewriter);
      continue;
    }
    SmallVector<RecordedGate> records;
    SmallVector<Value> allocated;

    for (auto* tempOp : step.temporaries) {
      emitTemporary(tempOp, storageMap, records, outerLoop, allocated,
                    rewriter);
    }

    if (isa<arith::XOrIOp>(step.updateOp)) {
      applyXorIInPlace(storageMap.at(step.overwriteValue),
                       storageMap.at(step.otherValue), loc, rewriter);
    } else if (isa<arith::AddIOp>(step.updateOp)) {
      applyAddIInPlace(storageMap.at(step.overwriteValue),
                       storageMap.at(step.otherValue), loc, rewriter);
    } else {
      llvm_unreachable("other updates are not lowered");
    }

    for (const auto& rec : llvm::reverse(records)) {
      emitGate(loc, rec.controls, rec.target, rewriter);
    }

    auto qubits = storageMap.at(step.overwriteValue);
    storageMap[step.updateOp->getResult(0)] = qubits;

    if (outerLoop != nullptr) {
      llvm::append_range(loopHelpers, allocated);
    } else {
      for (auto qubit : allocated) {
        qc::DeallocOp::create(rewriter, op->getLoc(), qubit);
      }
    }
  }
}

void rewriteOracle(OracleOp op, ArrayRef<UpdateStep> updateSteps,
                   PatternRewriter& rewriter) {
  rewriter.setInsertionPoint(op);
  auto loc = op->getLoc();

  // maps numeric values to qubit values
  DenseMap<Value, SmallVector<Value>> storageMap;

  for (auto [alias, operand] : llvm::zip_equal(
           op.getRegion().front().getArguments(), op->getOperands())) {
    if (isa<QubitType>(operand.getType())) {
      storageMap[alias] = {operand};
    } else {
      auto memrefType = cast<MemRefType>(operand.getType());
      auto size = memrefType.getDimSize(0);
      for (int64_t i = 0; i < size; ++i) {
        auto index = arith::ConstantIndexOp::create(rewriter, loc, i);
        auto load = memref::LoadOp::create(rewriter, loc, operand, {index});
        storageMap[alias].push_back(load.getResult());
      }
    }
  }

  SmallVector<Value> helpers;
  emitSteps(op, updateSteps, nullptr, loc, storageMap, helpers, rewriter);
}

struct ConvertOracleOp final : OpConversionPattern<OracleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OracleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    auto hasStatic =
        op->getParentOfType<ModuleOp>()
            .walk([](qc::StaticOp) { return WalkResult::interrupt(); })
            .wasInterrupted();
    if (hasStatic) {
      op.emitOpError("cannot lower qoracle in a program with static qubits");
      return failure();
    }

    if (!validRegion(&op.getRegion().front())) {
      op->emitOpError("contains invalid operations");
      return failure();
    }

    auto updateStepsOrFailure = plan(op);
    if (failed(updateStepsOrFailure)) {
      return failure();
    }
    rewriteOracle(op, *updateStepsOrFailure, rewriter);

    rewriter.eraseOp(op);
    return success();
  }
};

struct QOracleToQC final : impl::QOracleToQCBase<QOracleToQC> {
  using QOracleToQCBase::QOracleToQCBase;

protected:
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    auto moduleOp = getOperation();

    ConversionTarget target(*context);
    target.addIllegalDialect<QOracleDialect>();
    target.addLegalDialect<qc::QCDialect, arith::ArithDialect,
                           memref::MemRefDialect, scf::SCFDialect>();

    RewritePatternSet patterns(context);
    patterns.add<ConvertOracleOp>(context);

    // Avoid retaining replaced operations and SSA uses until conversion ends.
    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns),
                                      config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir
