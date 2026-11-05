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
#include "mlir/IR/Diagnostics.h"
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
  enum class Kind { Update, Create, Copy };

  Operation* updateOp;
  Value overwriteValue;
  Value otherValue;
  bool borrows = false;
  SmallVector<Operation*> temporaries;
  std::vector<UpdateStep> body;
  Kind kind = Kind::Update;
};

struct Statement {
  enum class Kind { Temp, Update };
  enum class Op { Xor, And, Maj };

  Kind kind;
  Op op;
  Value target;
  SmallVector<Value> operands;
  bool invert = false;
  bool inPlace = false;
  bool fused = false;
};

struct HelperAllocator {
  Operation* outerLoop;
  Location loc;
  PatternRewriter& rewriter;
  SmallVector<Value> allocated;

  Value allocate() {
    OpBuilder::InsertionGuard guard(rewriter);
    if (outerLoop != nullptr) {
      rewriter.setInsertionPoint(outerLoop);
    }
    auto qubit = qc::AllocOp::create(rewriter, loc);

    allocated.push_back(qubit);
    return qubit;
  }

  void release(SmallVector<Value>& container) {
    if (outerLoop != nullptr) {
      llvm::append_range(container, allocated);
    } else {
      for (auto qubit : allocated) {
        qc::DeallocOp::create(rewriter, loc, qubit);
      }
    }
  }

  void discard(Value qubit) {
    llvm::erase(allocated, qubit);
    rewriter.eraseOp(qubit.getDefiningOp());
  }
};

struct Rotation {
  Value source;
  int64_t amount;
};

SmallVector<Value> rotateList(ArrayRef<Value> list, int64_t amount) {
  SmallVector<Value> result(list.size());
  size_t n = list.size();
  for (size_t i = 0; i < list.size(); ++i) {
    result[i] = list[(i - amount + n) % n];
  }
  return result;
}

void sortBits(ArrayRef<Value> current, ArrayRef<Value> desired, Location loc,
              PatternRewriter& rewriter) {
  SmallVector<Value> currentList(current.begin(), current.end());
  size_t n = currentList.size();
  for (size_t i = 0; i < n; ++i) {
    if (currentList[i] == desired[i]) {
      continue;
    }
    auto* it = llvm::find(currentList, desired[i]);
    qc::SWAPOp::create(rewriter, loc, currentList[i], *it);
    *it = currentList[i];
    currentList[i] = desired[i];
  }
}

FailureOr<Rotation> matchRotation(Operation* op) {
  auto ori = dyn_cast<arith::OrIOp>(op);
  if (!ori) {
    return failure();
  }
  auto bitWidth = ori.getType().getIntOrFloatBitWidth();
  auto lhs = ori.getLhs();
  auto rhs = ori.getRhs();
  for (auto [hi, lo] : {std::pair{lhs, rhs}, std::pair{rhs, lhs}}) {
    auto shl = hi.getDefiningOp<arith::ShLIOp>();
    auto shr = lo.getDefiningOp<arith::ShRUIOp>();
    if (!shl || !shr) {
      continue;
    }
    if (shl.getLhs() != shr.getLhs()) {
      return failure();
    }
    auto k = getConstantIntValue(shl.getRhs());
    auto m = getConstantIntValue(shr.getRhs());
    if (!k || !m) {
      return failure();
    }
    if (*k <= 0 || *k >= bitWidth || *k + *m != bitWidth) {
      return failure();
    }
    if (!shl->hasOneUse() || !shr->hasOneUse()) {
      return failure();
    }
    return Rotation{.source = shl.getLhs(), .amount = *k};
  }
  return failure();
}

SmallVector<Value> getTemporaryInputs(Operation* op) {
  auto result = matchRotation(op);
  if (succeeded(result)) {
    return {result->source};
  }
  return llvm::to_vector(op->getOperands());
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

SmallVector<Statement>
createAdditionStatements(Value target, Value other, Statement::Kind kind,
                         DenseMap<Value, SmallVector<Value>>& storageMap,
                         HelperAllocator& allocator) {
  auto result = storageMap.at(target);
  auto otherValue = storageMap.at(other);
  SmallVector<Value> carries;
  SmallVector<Statement> statements;

  auto bitWidth = target.getType().getIntOrFloatBitWidth();
  for (size_t i = 0; i < bitWidth - 1; i++) {
    carries.push_back(allocator.allocate());
  }

  for (size_t i = 0; i < bitWidth; i++) {
    if (i + 1 < bitWidth) {
      if (i == 0) {
        statements.push_back({.kind = Statement::Kind::Temp,
                              .op = Statement::Op::And,
                              .target = carries[i],
                              .operands = {result[i], otherValue[i]}});
      } else {
        statements.push_back(
            {.kind = Statement::Kind::Temp,
             .op = Statement::Op::Maj,
             .target = carries[i],
             .operands = {result[i], otherValue[i], carries[i - 1]}});
      }
    }

    if (i == 0) {
      statements.push_back({
          .kind = kind,
          .op = Statement::Op::Xor,
          .target = result[i],
          .operands = {otherValue[i]},
      });
    } else {
      statements.push_back({
          .kind = kind,
          .op = Statement::Op::Xor,
          .target = result[i],
          .operands = {otherValue[i], carries[i - 1]},
      });
    }
  }

  return statements;
}

void computeInto(Operation* op, Statement::Kind kind,
                 DenseMap<Value, SmallVector<Value>>& storageMap,
                 HelperAllocator& allocator,
                 SmallVector<Statement>& statements) {
  auto result = storageMap.at(op->getResult(0));

  if (isa<arith::XOrIOp, arith::AndIOp>(op)) {
    auto lhs = storageMap.at(op->getOperand(0));
    auto rhs = storageMap.at(op->getOperand(1));

    auto operation =
        isa<arith::XOrIOp>(op) ? Statement::Op::Xor : Statement::Op::And;
    auto bitWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = operation,
                            .target = result[i],
                            .operands = {lhs[i], rhs[i]}});
    }
  } else if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    auto constantValue = cast<IntegerAttr>(constOp.getValue()).getValue();
    auto bitWidth = constOp.getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      if (!constantValue[i]) {
        continue;
      }
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {},
                            .invert = true});
    }
  } else if (auto addiOp = dyn_cast<arith::AddIOp>(op)) {
    auto lhs = storageMap.at(op->getOperand(0));
    auto bitWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {lhs[i]}});
    }
    auto additionStatements = createAdditionStatements(
        op->getResult(0), op->getOperand(1), kind, storageMap, allocator);
    statements.append(additionStatements.begin(), additionStatements.end());
  } else if (auto rotation = matchRotation(op); succeeded(rotation)) {
    auto source = rotateList(storageMap.at(rotation->source), rotation->amount);
    auto bitWidth = source.size();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {source[i]}});
    }
  } else {
    llvm_unreachable("Unexpected temporary operation");
  }
}

SmallVector<Statement>
buildStatements(const UpdateStep& update,
                DenseMap<Value, SmallVector<Value>>& storageMap,
                HelperAllocator& allocator) {
  SmallVector<Statement> statements;
  for (const auto& temporary : update.temporaries) {
    if (isa<arith::ExtUIOp, arith::TruncIOp>(temporary) ||
        succeeded(matchRotation(temporary))) {
      continue;
    }
    computeInto(temporary, Statement::Kind::Temp, storageMap, allocator,
                statements);
  }

  if (update.kind == UpdateStep::Kind::Create) {
    auto entry = storageMap.at(update.overwriteValue);
    storageMap[update.updateOp->getResult(0)] = entry;
    computeInto(update.updateOp, Statement::Kind::Update, storageMap, allocator,
                statements);
  } else if (isa<arith::XOrIOp>(update.updateOp)) {
    auto bitWidth =
        update.updateOp->getResult(0).getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = Statement::Kind::Update,
                            .op = Statement::Op::Xor,
                            .target = storageMap.at(update.overwriteValue)[i],
                            .operands = {storageMap.at(update.otherValue)[i]}});
    }
  } else if (isa<arith::AddIOp>(update.updateOp)) {
    auto additionStatements = createAdditionStatements(
        update.overwriteValue, update.otherValue, Statement::Kind::Update,
        storageMap, allocator);
    statements.append(additionStatements.begin(), additionStatements.end());
  } else {
    llvm_unreachable("Unexpected update operation");
  }
  return statements;
}

void emitStatement(const Statement& statement, Location loc,
                   PatternRewriter& rewriter) {
  auto target = statement.target;
  if (statement.op == Statement::Op::And) {
    emitGate(loc, statement.operands, target, rewriter);
  } else if (statement.op == Statement::Op::Xor) {
    for (auto control : statement.operands) {
      emitGate(loc, {control}, target, rewriter);
    }
    if (statement.invert) {
      emitGate(loc, {}, target, rewriter);
    }
  } else if (statement.inPlace) {
    auto x = statement.target;
    auto y = statement.operands[0];
    auto z = statement.operands[1];
    emitGate(loc, {x}, y, rewriter);
    emitGate(loc, {x}, z, rewriter);
    emitGate(loc, {y, z}, x, rewriter);
  } else if (statement.op == Statement::Op::Maj) {
    assert(statement.operands.size() == 3 &&
           "maj takes exactly three operands");
    auto x = statement.operands[0];
    auto y = statement.operands[1];
    auto z = statement.operands[2];
    emitGate(loc, {z}, x, rewriter);
    emitGate(loc, {z}, y, rewriter);
    emitGate(loc, {x, y}, statement.target, rewriter);
    emitGate(loc, {z}, statement.target, rewriter);
    emitGate(loc, {z}, y, rewriter);
    emitGate(loc, {z}, x, rewriter);
  }
}

SmallVector<size_t> rollbackStatement(ArrayRef<Statement> statements,
                                      size_t index) {
  DenseSet<Value> neededValues(llvm::from_range, statements[index].operands);
  BitVector picked(statements.size());
  bool found = true;

  while (found) {
    found = false;
    for (size_t i = index + 1; i < statements.size(); ++i) {
      const auto& u = statements[i];
      if (u.kind == Statement::Kind::Temp) {
        continue;
      }
      if (picked[i]) {
        continue;
      }

      if (neededValues.contains(u.target)) {
        picked[i] = true;
        found = true;
        for (auto operand : u.operands) {
          neededValues.insert(operand);
        }
      }
    }
  }

  SmallVector<size_t> pickedStatements;
  for (auto i : picked.set_bits()) {
    pickedStatements.push_back(i);
  }
  return pickedStatements;
}

void emitGadgetCleanup(const Statement& statement, Location loc,
                       PatternRewriter& rewriter) {
  auto x = statement.target;
  auto y = statement.operands[0];
  auto z = statement.operands[1];
  emitGate(loc, {y, z}, x, rewriter);
  emitGate(loc, {x}, z, rewriter);
  if (statement.fused) {
    emitGate(loc, {z}, y, rewriter);
  } else {
    emitGate(loc, {x}, y, rewriter);
  }
}

void temporaryCleanup(ArrayRef<Statement> statements, Location loc,
                      PatternRewriter& rewriter) {
  for (size_t i = statements.size(); i-- > 0;) {
    const auto& statement = statements[i];
    if (statement.kind != Statement::Kind::Temp) {
      continue;
    }
    if (statement.inPlace) {
      emitGadgetCleanup(statement, loc, rewriter);
      continue;
    }
    auto toRollBack = rollbackStatement(statements, i);
    for (auto j : llvm::reverse(toRollBack)) {
      emitStatement(statements[j], loc, rewriter);
    }
    emitStatement(statement, loc, rewriter);
    for (auto j : toRollBack) {
      assert(llvm::none_of(statements.drop_front(i + 1),
                           [&](const Statement& t) {
                             return t.kind == Statement::Kind::Temp &&
                                    llvm::is_contained(statements[j].operands,
                                                       t.target);
                           }) &&
             "rollback reads an already cleaned temporary");

      emitStatement(statements[j], loc, rewriter);
    }
  }
}

void makeGadgets(SmallVector<Statement>& statements,
                 HelperAllocator& allocator) {
  for (size_t i = 0; i < statements.size(); i++) {
    const auto& statement = statements[i];
    if (statement.kind != Statement::Kind::Temp) {
      continue;
    }
    if (statement.op != Statement::Op::Maj) {
      continue;
    }

    auto target = statement.target;
    bool done = false;
    auto operands = statement.operands;
    for (auto x : operands) {
      if (done) {
        break;
      }
      for (auto y : operands) {
        if (x == y) {
          continue;
        }
        auto z =
            *llvm::find_if(operands, [&](auto v) { return v != x && v != y; });

        size_t fusionIndex = statements.size();
        for (size_t j = i + 1; j < statements.size(); j++) {
          const auto& s = statements[j];
          if (s.kind == Statement::Kind::Update && s.op == Statement::Op::Xor &&
              s.target == y && s.operands.size() == 2 &&
              llvm::is_contained(s.operands, x) &&
              llvm::is_contained(s.operands, z)) {
            fusionIndex = j;
            break;
          }
        }
        if (fusionIndex < statements.size()) {
          for (size_t j = 0; j < i; j++) {
            const auto& s = statements[j];
            if (s.kind == Statement::Kind::Temp && !s.inPlace &&
                llvm::is_contained(rollbackStatement(statements, j),
                                   fusionIndex)) {
              fusionIndex = statements.size();
              break;
            }
          }
        }

        bool incompatible = false;
        for (size_t j = i + 1; j < statements.size(); j++) {
          if (j == fusionIndex) {
            continue;
          }
          if (statements[j].target == x || statements[j].target == y ||
              statements[j].target == z) {
            incompatible = true;
            break;
          }
          if (llvm::is_contained(statements[j].operands, x) ||
              llvm::is_contained(statements[j].operands, y) ||
              llvm::is_contained(statements[j].operands, z)) {
            incompatible = true;
            break;
          }
        }

        if (incompatible) {
          continue;
        }

        statements[i].target = x;
        statements[i].operands = {y, z};
        statements[i].inPlace = true;
        statements[i].fused = fusionIndex < statements.size();
        if (fusionIndex < statements.size()) {
          statements[fusionIndex].fused = true;
        }

        for (size_t j = i + 1; j < statements.size(); j++) {
          llvm::replace(statements[j].operands, target, x);
        }

        allocator.discard(target);
        done = true;
        break;
      }
    }
  }
}

void emitUpdateStep(const UpdateStep& step,
                    DenseMap<Value, SmallVector<Value>>& storageMap,
                    Location loc, Operation* outerLoop,
                    SmallVector<Value>& loopHelpers,
                    PatternRewriter& rewriter) {
  if (step.kind == UpdateStep::Kind::Copy) {
    auto source = storageMap.at(step.otherValue);
    auto target = storageMap.at(step.overwriteValue);
    for (auto [s, t] : llvm::zip(source, target)) {
      emitGate(loc, {s}, t, rewriter);
    }
    return;
  }

  auto rotation = matchRotation(step.updateOp);
  if (step.kind == UpdateStep::Kind::Update && succeeded(rotation)) {
    storageMap[step.updateOp->getResult(0)] =
        rotateList(storageMap.at(rotation->source), rotation->amount);
    return;
  }

  HelperAllocator allocator{
      .outerLoop = outerLoop, .loc = loc, .rewriter = rewriter};

  for (auto* temporary : step.temporaries) {
    auto temporaryRotation = matchRotation(temporary);
    if (succeeded(temporaryRotation)) {
      storageMap[temporary->getResult(0)] = rotateList(
          storageMap.at(temporaryRotation->source), temporaryRotation->amount);
      continue;
    }

    SmallVector<Value> helpers;
    auto bitWidth = temporary->getResult(0).getType().getIntOrFloatBitWidth();
    if (auto extuiOp = dyn_cast<arith::ExtUIOp>(temporary)) {
      auto operand = extuiOp.getOperand();
      auto oldWidth = operand.getType().getIntOrFloatBitWidth();
      for (size_t i = 0; i < oldWidth; ++i) {
        helpers.push_back(storageMap.at(operand)[i]);
      }
      for (size_t i = oldWidth; i < bitWidth; ++i) {
        auto qubit = allocator.allocate();
        helpers.push_back(qubit);
      }
      storageMap[extuiOp.getResult()] = std::move(helpers);
      continue;
    }
    if (auto truncOp = dyn_cast<arith::TruncIOp>(temporary)) {
      auto operand = truncOp.getOperand();
      for (size_t i = 0; i < bitWidth; ++i) {
        helpers.push_back(storageMap.at(operand)[i]);
      }
      storageMap[truncOp.getResult()] = std::move(helpers);
      continue;
    }

    for (size_t i = 0; i < bitWidth; ++i) {
      auto qubit = allocator.allocate();
      helpers.push_back(qubit);
    }
    storageMap[temporary->getResult(0)] = std::move(helpers);
  }

  auto statements = buildStatements(step, storageMap, allocator);
  makeGadgets(statements, allocator);

  for (const auto& statement : statements) {
    if (statement.kind == Statement::Kind::Update && statement.fused) {
      continue;
    }
    emitStatement(statement, loc, rewriter);
  }

  temporaryCleanup(statements, loc, rewriter);

  auto qubits = storageMap.at(step.overwriteValue);
  storageMap[step.updateOp->getResult(0)] = qubits;

  allocator.release(loopHelpers);
}

void emitInverseStep(const UpdateStep& step,
                     DenseMap<Value, SmallVector<Value>>& storageMap,
                     Location loc, PatternRewriter& rewriter) {
  SmallVector<Value> loopHelpers;
  if (isa<arith::XOrIOp>(step.updateOp)) {
    emitUpdateStep(step, storageMap, loc, nullptr, loopHelpers, rewriter);
    return;
  }
  if (isa<arith::AddIOp>(step.updateOp)) {
    auto bitWidth =
        step.updateOp->getResult(0).getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; i++) {
      emitGate(loc, {}, storageMap.at(step.overwriteValue)[i], rewriter);
    }
    emitUpdateStep(step, storageMap, loc, nullptr, loopHelpers, rewriter);
    for (size_t i = 0; i < bitWidth; i++) {
      emitGate(loc, {}, storageMap.at(step.overwriteValue)[i], rewriter);
    }
  }
}

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
    for (auto operand : llvm::reverse(getTemporaryInputs(defOp))) {
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

bool canRestore(SmallVector<UpdateStep> steps,
                DenseMap<Value, Value>& containers) {
  for (size_t i = 0; i < steps.size(); ++i) {
    if (!steps[i].borrows) {
      continue;
    }
    DenseSet<Value> reads;
    if (steps[i].otherValue && containers.contains(steps[i].otherValue)) {
      reads.insert(containers[steps[i].otherValue]);
    }
    for (auto* temporary : steps[i].temporaries) {
      for (auto operand : getTemporaryInputs(temporary)) {
        if (!containers.contains(operand)) {
          continue;
        }
        reads.insert(containers.at(operand));
      }
    }

    for (size_t j = i + 1; j < steps.size(); ++j) {
      if (steps[j].borrows) {
        continue;
      }
      DenseSet<Value> writes;
      if (auto forOp = dyn_cast<scf::ForOp>(steps[j].updateOp)) {
        for (auto arg : forOp.getInitArgs()) {
          writes.insert(containers.at(arg));
        }
      } else {
        writes.insert(containers.at(steps[j].overwriteValue));
      }

      if (llvm::any_of(reads,
                       [&](Value read) { return writes.contains(read); })) {
        return false;
      }
    }
  }
  return true;
}

bool canOverwrite(Operation* op, Value value,
                  DenseMap<Value, Value>& containers) {
  if (!containers.contains(value)) {
    return false;
  }
  if (value.getParentBlock() != op->getBlock() || !isLastReader(op, value)) {
    return false;
  }
  return true;
}

std::pair<Value, Value> operandToOverwrite(Operation* op, OracleOp oracleOp,
                                           DenseMap<Value, Value>& containers,
                                           bool allowBorrowing) {
  if (!isa<arith::XOrIOp, arith::AddIOp>(op)) {
    return {nullptr, nullptr};
  }

  size_t overwriteIndex = 2;
  for (bool kept : {false, true}) {
    if (overwriteIndex < 2 || (kept && !allowBorrowing)) {
      break;
    }
    for (auto [i, value] : llvm::enumerate(op->getOperands())) {
      if (oracleOp.isKeptAlias(containers.lookup(value)) != kept) {
        continue;
      }
      if (!canOverwrite(op, value, containers)) {
        continue;
      }
      overwriteIndex = i;
      break;
    }
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
planBlock(Block& block, OracleOp oracle, DenseMap<Value, Value>& containers,
          const DenseMap<Operation*, Value>& creations, bool allowBorrowing);

FailureOr<UpdateStep> planLoop(scf::ForOp forOp, OracleOp oracle,
                               DenseMap<Value, Value>& containers,
                               const DenseMap<Operation*, Value>& creations) {
  if (!forOp.getInductionVar().use_empty()) {
    oracle.emitOpError(
        "cannot lower qoracle because a loop uses its induction variable");
    return failure();
  }
  for (auto [init, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (!containers.contains(init) || oracle.isKeptAlias(containers[init])) {
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
    auto container = containers.at(init);
    containers[iterArg] = container;
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

  auto body = planBlock(forOp.getBodyRegion().front(), oracle, containers,
                        creations, false);
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
    auto container = containers.at(init);
    containers[result] = container;
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
planBlock(Block& block, OracleOp oracle, DenseMap<Value, Value>& containers,
          const DenseMap<Operation*, Value>& creations, bool allowBorrowing) {
  SmallVector<UpdateStep> updateSteps;
  for (auto& op : block.without_terminator()) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto step = planLoop(forOp, oracle, containers, creations);
      if (failed(step)) {
        return failure();
      }
      updateSteps.push_back(std::move(*step));
      continue;
    }

    if (auto zero = creations.lookup(&op)) {
      SetVector<Operation*> allReachableTemporaries;
      for (auto input : getTemporaryInputs(&op)) {
        allReachableTemporaries.insert_range(
            reachableTemporaries(input, containers));
      }
      updateSteps.push_back(UpdateStep{
          .updateOp = &op,
          .overwriteValue = zero,
          .otherValue = nullptr,
          .borrows = false,
          .temporaries = SmallVector<Operation*>(
              allReachableTemporaries.begin(), allReachableTemporaries.end()),
          .body = {},
          .kind = UpdateStep::Kind::Create,
      });
      containers[op.getResult(0)] = zero;
      continue;
    }

    auto rotation = matchRotation(&op);
    if (succeeded(rotation)) {
      auto source = rotation->source;
      if (canOverwrite(&op, source, containers) &&
          (allowBorrowing || !oracle.isKeptAlias(containers[source]))) {
        updateSteps.emplace_back(
            UpdateStep{.updateOp = &op,
                       .overwriteValue = source,
                       .otherValue = nullptr,
                       .borrows = oracle.isKeptAlias(containers[source])});
        auto container = containers.at(source);
        containers[op.getResult(0)] = container;
      }
      continue;
    }

    auto [overwriteValue, other] =
        operandToOverwrite(&op, oracle, containers, allowBorrowing);
    auto temporaries = reachableTemporaries(other, containers);
    if (!overwriteValue) {
      continue;
    }

    for (auto* tempOp : temporaries) {
      if (llvm::is_contained(getTemporaryInputs(tempOp), overwriteValue)) {
        oracle.emitOpError(
            "cannot lower qoracle because a temporary depends on "
            "the value to be overwritten");
        return failure();
      }
    }

    updateSteps.emplace_back(&op, overwriteValue, other,
                             oracle.isKeptAlias(containers[overwriteValue]),
                             temporaries);
    auto container = containers.at(overwriteValue);
    containers[op.getResult(0)] = container;
  }

  DenseSet<Value> overwritten;
  for (auto& step : updateSteps) {
    for (auto* tempOp : step.temporaries) {
      for (auto operand : getTemporaryInputs(tempOp)) {
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

Value chainInput(Operation* op, OracleOp oracle) {
  if (!isa<arith::XOrIOp, arith::AddIOp>(op) && failed(matchRotation(op))) {
    return nullptr;
  }
  for (auto input : getTemporaryInputs(op)) {
    if (input.getParentBlock() == op->getBlock() &&
        !oracle.isKeptAlias(input) && isLastReader(op, input)) {
      return input;
    }
  }
  return nullptr;
}

Value chainStart(Value value, OracleOp oracle) {
  while (auto* op = value.getDefiningOp()) {
    auto input = chainInput(op, oracle);
    if (!input) {
      break;
    }
    value = input;
  }
  return value;
}

FailureOr<DenseMap<Operation*, Value>> findCreations(OracleOp oracle) {
  DenseMap<Operation*, Value> creations;
  auto yield = oracle.getRegion().getBlocks().front().getTerminator();
  for (auto [index, yielded] : llvm::enumerate(yield->getOperands())) {
    auto consumed = oracle.getConsumedAliases()[index];
    if (!oracle.isZeroAlias(consumed)) {
      continue;
    }
    auto* op = chainStart(yielded, oracle).getDefiningOp();
    if (!op || isa<scf::ForOp>(op)) {
      continue;
    }
    if (creations.contains(op)) {
      continue;
    }
    if (!consumed.use_empty()) {
      oracle.emitOpError("cannot lower qoracle because a zero alias is read "
                         "while a value is created in its register");
      return failure();
    }
    if (!isa<arith::XOrIOp, arith::AddIOp, arith::AndIOp, arith::ConstantOp>(
            op) &&
        failed(matchRotation(op))) {
      oracle.emitOpError("cannot lower qoracle because an unsupported "
                         "operation result is created in a zero register");
      return failure();
    }
    creations[op] = consumed;
  }
  return creations;
}

FailureOr<SmallVector<UpdateStep>>
planWith(OracleOp oracle, bool allowBorrowing,
         DenseMap<Value, Value>& containers) {
  for (auto consumed : oracle.getConsumedAliases()) {
    containers[consumed] = consumed;
  }
  for (auto kept : oracle.getKeptAliases()) {
    containers[kept] = kept;
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

  auto creations = findCreations(oracle);
  if (failed(creations)) {
    return failure();
  }
  auto steps = planBlock(oracle.getRegion().front(), oracle, containers,
                         *creations, allowBorrowing);
  if (failed(steps)) {
    return failure();
  }
  auto updateSteps = std::move(*steps);

  qoracle::YieldOp yield =
      llvm::cast<qoracle::YieldOp>(oracle.getRegion().front().getTerminator());
  for (auto [i, operand] : llvm::enumerate(yield.getOperands())) {
    auto reg = oracle.getConsumedAliases()[i];
    if (containers.lookup(operand) == reg) {
      continue;
    }
    if (oracle.isZeroAlias(reg) && containers.contains(operand) &&
        reg.use_empty() &&
        !llvm::is_contained(llvm::make_second_range(*creations), reg)) {
      updateSteps.push_back(UpdateStep{
          .updateOp = yield,
          .overwriteValue = reg,
          .otherValue = operand,
          .borrows = false,
          .temporaries = {},
          .body = {},
          .kind = UpdateStep::Kind::Copy,
      });
      continue;
    }
    oracle.emitOpError("cannot lower qoracle because the yielded value ")
        << i << " does not match the corresponding consumed alias";
    return failure();
  }

  return success(updateSteps);
}

FailureOr<SmallVector<UpdateStep>> plan(OracleOp oracle) {
  {
    ScopedDiagnosticHandler silence(oracle.getContext(),
                                    [](Diagnostic&) { return success(); });
    DenseMap<Value, Value> containers;
    auto steps = planWith(oracle, true, containers);
    if (succeeded(steps) && canRestore(*steps, containers)) {
      return steps;
    }
  }
  DenseMap<Value, Value> containers;
  return planWith(oracle, false, containers);
}

bool validRegion(Block* block) {
  if (block
          ->walk([&](Operation* op) {
            if (isa<arith::OrIOp>(op) && succeeded(matchRotation(op))) {
              // OrIOps are allowed to represent a rotation.
              return WalkResult::advance();
            }
            if (isa<arith::ShLIOp, arith::ShRUIOp>(op) && op->hasOneUse() &&
                succeeded(matchRotation(*op->getUsers().begin()))) {
              // Shifts used inside a rotation are allowed.
              return WalkResult::advance();
            }
            if (!isa<arith::ConstantOp, arith::XOrIOp, arith::AndIOp,
                     arith::AddIOp, arith::ExtUIOp, arith::TruncIOp,
                     qoracle::YieldOp, scf::ForOp, scf::YieldOp>(op)) {
              return WalkResult::interrupt();
            }
            if (!isa<qoracle::YieldOp>(op) && op->getNumOperands() == 2 &&
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
  for (auto [iterArg, yieldArg] :
       llvm::zip_equal(forOp.getRegionIterArgs(),
                       forOp.getBody()->getTerminator()->getOperands())) {
    sortBits(storageMap.at(yieldArg), storageMap.at(iterArg), loc, rewriter);
  }

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
    emitUpdateStep(step, storageMap, loc, outerLoop, loopHelpers, rewriter);
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
  DenseSet<Value> copied;

  for (const auto& step : llvm::reverse(updateSteps)) {
    if (step.kind == UpdateStep::Kind::Copy) {
      copied.insert(step.overwriteValue);
    }
    if (!step.borrows) {
      continue;
    }
    emitInverseStep(step, storageMap, loc, rewriter);
  }

  for (auto [consumedArg, yieldArg] :
       llvm::zip_equal(op.getConsumedAliases(),
                       op.getRegion().front().getTerminator()->getOperands())) {
    if (copied.contains(consumedArg)) {
      continue;
    }
    sortBits(storageMap.at(yieldArg), storageMap.at(consumedArg), loc,
             rewriter);
  }
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
