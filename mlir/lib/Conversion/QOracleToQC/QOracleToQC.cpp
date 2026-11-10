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
  bool cleanup = false;
};

struct Statement {
  enum class Kind { Temp, Update };
  enum class Op { Xor, And, Maj, Or };

  Kind kind;
  Op op;
  Value target;
  SmallVector<Value> operands;
  bool invert = false;
  bool inPlace = false;
  bool fused = false;
  SmallVector<bool> negated;
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

struct Shift {
  Value source;
  int64_t amount;
};

struct Majority {
  Value x;
  Value y;
  Value z;
};

SmallVector<Value> rotateList(ArrayRef<Value> list, int64_t amount) {
  SmallVector<Value> result(list.size());
  size_t n = list.size();
  for (size_t i = 0; i < list.size(); ++i) {
    result[i] = list[(i - amount + n) % n];
  }
  return result;
}

SmallVector<Value> shiftList(ArrayRef<Value> list, int64_t amount) {
  SmallVector<Value> result(list.size());
  size_t n = list.size();
  for (size_t i = 0; i < list.size(); ++i) {
    result[i] = i + amount < n ? list[i + amount] : Value{};
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

FailureOr<Shift> matchShift(Operation* op) {
  auto shr = dyn_cast<arith::ShRUIOp>(op);
  if (!shr) {
    return failure();
  }
  auto bitWidth = shr.getType().getIntOrFloatBitWidth();
  auto k = getConstantIntValue(shr.getRhs());
  if (!k || *k <= 0 || *k >= bitWidth) {
    return failure();
  }
  return Shift{
      .source = shr.getLhs(),
      .amount = *k,
  };
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

FailureOr<SmallVector<Value>>
relabel(Operation* op, DenseMap<Value, SmallVector<Value>>& storageMap) {
  auto rotation = matchRotation(op);
  if (succeeded(rotation)) {
    return rotateList(storageMap.at(rotation->source), rotation->amount);
  }
  auto shift = matchShift(op);
  if (succeeded(shift)) {
    return shiftList(storageMap.at(shift->source), shift->amount);
  }
  return failure();
}

FailureOr<Majority> majorityOf(arith::AndIOp a, arith::AndIOp b,
                               arith::AndIOp c) {
  SetVector<Value> inputs;
  SetVector<Value> doubles;
  for (auto op : {a, b, c}) {
    for (auto operand : op.getOperands()) {
      if (!inputs.insert(operand)) {
        doubles.insert(operand);
      }
    }
  }
  if (inputs.size() != 3 || doubles.size() != 3) {
    return failure();
  }
  return Majority{
      .x = inputs[0],
      .y = inputs[1],
      .z = inputs[2],
  };
}

FailureOr<Majority> matchMajority(Operation* op) {
  if (!isa<arith::OrIOp, arith::XOrIOp>(op)) {
    return failure();
  }
  auto lhs = op->getOperand(0);
  auto rhs = op->getOperand(1);
  for (auto [single, pair] : {std::pair{lhs, rhs}, std::pair{rhs, lhs}}) {
    auto andOp = single.getDefiningOp<arith::AndIOp>();
    auto innerOp = pair.getDefiningOp();
    if (!andOp || !innerOp || !andOp->hasOneUse() || !innerOp->hasOneUse() ||
        !isa<arith::OrIOp, arith::XOrIOp>(innerOp)) {
      continue;
    }
    auto and1 = innerOp->getOperand(0).getDefiningOp<arith::AndIOp>();
    auto and2 = innerOp->getOperand(1).getDefiningOp<arith::AndIOp>();
    if (!and1 || !and2 || !and1->hasOneUse() || !and2->hasOneUse()) {
      continue;
    }
    return majorityOf(and1, and2, andOp);
  }
  return failure();
}

Operation* majorityRoot(Operation* op) {
  if (!isa<arith::AndIOp>(op) || !op->hasOneUse()) {
    return nullptr;
  }
  auto user = *op->getUsers().begin();
  if (succeeded(matchMajority(user))) {
    return user;
  }
  if (user->hasOneUse() &&
      succeeded(matchMajority(*user->getUsers().begin()))) {
    return *user->getUsers().begin();
  }
  return nullptr;
}

bool isConstantEquality(Operation* op) {
  auto cmp = dyn_cast<arith::CmpIOp>(op);
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq) {
    return false;
  }
  auto* lhs = cmp.getLhs().getDefiningOp();
  auto* rhs = cmp.getRhs().getDefiningOp();
  auto leftConst = lhs && isa<arith::ConstantOp>(lhs);
  auto rightConst = rhs && isa<arith::ConstantOp>(rhs);
  return leftConst ^ rightConst;
}

SmallVector<Value> getTemporaryInputs(Operation* op) {
  auto result = matchRotation(op);
  if (succeeded(result)) {
    return {result->source};
  }
  auto shift = matchShift(op);
  if (succeeded(shift)) {
    return {shift->source};
  }
  auto maj = matchMajority(op);
  if (succeeded(maj)) {
    return {maj->x, maj->y, maj->z};
  }
  if (isConstantEquality(op)) {
    auto cmp = cast<arith::CmpIOp>(op);
    return {cmp.getLhs().getDefiningOp<arith::ConstantOp>() ? cmp.getRhs()
                                                            : cmp.getLhs()};
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

SmallVector<Statement> createConstantAdditionStatements(
    Value target, APInt constant, Statement::Kind kind,
    DenseMap<Value, SmallVector<Value>>& storageMap,
    HelperAllocator& allocator) {
  auto result = storageMap.at(target);
  auto bitWidth = constant.getBitWidth();
  SmallVector<Statement> statements;
  Value carry = nullptr;
  for (size_t i = 0; i < bitWidth; ++i) {
    Value next;
    if (i + 1 < bitWidth) {
      if (carry == nullptr && constant[i]) {
        next = allocator.allocate();
        statements.push_back(Statement{.kind = Statement::Kind::Temp,
                                       .op = Statement::Op::Xor,
                                       .target = next,
                                       .operands = {result[i]}});
      } else if (carry != nullptr) {
        next = allocator.allocate();
        statements.push_back(Statement{.kind = Statement::Kind::Temp,
                                       .op = constant[i] ? Statement::Op::Or
                                                         : Statement::Op::And,
                                       .target = next,
                                       .operands = {result[i], carry}});
      }
    }
    if (carry != nullptr || constant[i]) {
      statements.push_back(Statement{
          .kind = kind,
          .op = Statement::Op::Xor,
          .target = result[i],
          .operands = carry ? SmallVector<Value>{carry} : SmallVector<Value>{},
          .invert = constant[i],
      });
    }
    carry = next;
  }
  return statements;
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

FailureOr<APInt> constantBits(Value value) {
  APInt bits;
  if (matchPattern(value, m_ConstantInt(&bits))) {
    return bits;
  }
  return failure();
}

bool isFolded(Operation* user, Value operand) {
  if (!operand || !isa<arith::XOrIOp, arith::AddIOp>(user)) {
    return false;
  }
  auto lhs = user->getOperand(0);
  auto rhs = user->getOperand(1);
  auto other = lhs == operand ? rhs : lhs;
  return matchPattern(operand, m_Constant()) &&
         !matchPattern(other, m_Constant());
}

void computeInto(Operation* op, Statement::Kind kind,
                 DenseMap<Value, SmallVector<Value>>& storageMap,
                 HelperAllocator& allocator,
                 SmallVector<Statement>& statements) {
  auto result = storageMap.at(op->getResult(0));

  if (auto updatedBits = relabel(op, storageMap); succeeded(updatedBits)) {
    auto bitWidth = updatedBits->size();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {(*updatedBits)[i]}});
    }
  } else if (isConstantEquality(op)) {
    auto lhs = op->getOperand(0);
    auto rhs = op->getOperand(1);
    auto constant = lhs.getDefiningOp<arith::ConstantOp>() ? lhs : rhs;
    auto other = lhs.getDefiningOp<arith::ConstantOp>() ? rhs : lhs;
    auto c = *constantBits(constant);
    auto bits = storageMap.at(other);
    SmallVector<std::pair<Value, bool>> inputs;
    for (size_t i = 0; i < c.getBitWidth(); ++i) {
      if (!bits[i]) {
        if (c[i]) {
          return;
        }
        continue;
      }
      inputs.push_back({bits[i], !c[i]});
    }

    if (inputs.empty()) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[0],
                            .operands = {},
                            .invert = true});
    } else {
      while (inputs.size() > 2) {
        SmallVector<std::pair<Value, bool>> nextInputs;
        for (size_t i = 0; i + 1 < inputs.size(); i += 2) {
          auto [p, notP] = inputs[i];
          auto [q, notQ] = inputs[i + 1];
          auto h = allocator.allocate();
          statements.push_back({
              .kind = Statement::Kind::Temp,
              .op = Statement::Op::And,
              .target = h,
              .operands = {p, q},
              .negated = {notP, notQ},
          });
          nextInputs.push_back({h, false});
        }
        if (inputs.size() % 2 == 1) {
          nextInputs.push_back(inputs.back());
        }
        inputs = nextInputs;
      }
      if (inputs.size() == 2) {
        statements.push_back({
            .kind = kind,
            .op = Statement::Op::And,
            .target = result[0],
            .operands = {inputs[0].first, inputs[1].first},
            .negated = {inputs[0].second, inputs[1].second},
        });
      } else {
        statements.push_back({
            .kind = kind,
            .op = Statement::Op::Xor,
            .target = result[0],
            .operands = {inputs[0].first},
            .invert = inputs[0].second,
        });
      }
    }
  } else if (auto majority = matchMajority(op); succeeded(majority)) {
    auto xs = storageMap.at(majority->x);
    auto ys = storageMap.at(majority->y);
    auto zs = storageMap.at(majority->z);
    for (size_t i = 0; i < xs.size(); ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Maj,
                            .target = result[i],
                            .operands = {xs[i], ys[i], zs[i]}});
    }
  } else if (isa<arith::XOrIOp>(op) && (isFolded(op, op->getOperand(0)) ||
                                        isFolded(op, op->getOperand(1)))) {
    Value folded =
        isFolded(op, op->getOperand(0)) ? op->getOperand(0) : op->getOperand(1);
    auto constant = *constantBits(folded);
    auto other =
        folded == op->getOperand(0) ? op->getOperand(1) : op->getOperand(0);
    auto otherBits = storageMap.at(other);
    auto bitWidth = constant.getBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {otherBits[i]},
                            .invert = constant[i]});
    }
  } else if (isa<arith::XOrIOp, arith::AndIOp, arith::OrIOp>(op)) {
    auto lhs = storageMap.at(op->getOperand(0));
    auto rhs = storageMap.at(op->getOperand(1));

    auto operation = isa<arith::XOrIOp>(op)   ? Statement::Op::Xor
                     : isa<arith::AndIOp>(op) ? Statement::Op::And
                                              : Statement::Op::Or;
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
    auto folded = isFolded(op, op->getOperand(0))   ? op->getOperand(0)
                  : isFolded(op, op->getOperand(1)) ? op->getOperand(1)
                                                    : nullptr;
    auto other =
        folded == op->getOperand(0) ? op->getOperand(1) : op->getOperand(0);
    auto bitWidth = op->getResult(0).getType().getIntOrFloatBitWidth();
    for (size_t i = 0; i < bitWidth; ++i) {
      statements.push_back({.kind = kind,
                            .op = Statement::Op::Xor,
                            .target = result[i],
                            .operands = {storageMap.at(other)[i]}});
    }
    if (folded) {
      auto constant = *constantBits(folded);
      statements.append(createConstantAdditionStatements(
          op->getResult(0), constant, kind, storageMap, allocator));
    } else {
      auto additionStatements = createAdditionStatements(
          op->getResult(0), op->getOperand(1), kind, storageMap, allocator);
      statements.append(additionStatements.begin(), additionStatements.end());
    }
  } else {
    llvm_unreachable("Unexpected temporary operation");
  }
}

SetVector<Operation*> readers(Value value) {
  SetVector<Operation*> result;
  for (auto* user : value.getUsers()) {
    if (isa<arith::ShLIOp, arith::ShRUIOp>(user) && user->hasOneUse() &&
        succeeded(matchRotation(*user->user_begin()))) {
      user = *user->user_begin();
    }
    if (auto* majRoot = majorityRoot(user)) {
      user = majRoot;
    }
    result.insert(user);
  }
  return result;
}

bool needsRegister(Operation* op) {
  if (isa<arith::ConstantOp, arith::ExtUIOp, arith::TruncIOp, arith::ShLIOp>(
          op)) {
    return false;
  }
  if (succeeded(matchRotation(op)) || succeeded(matchShift(op))) {
    return false;
  }

  auto value = op->getResult(0);
  if (value.use_empty()) {
    return false;
  }
  if (isConstantEquality(op)) {
    return !(op->hasOneUse() && isa<arith::XOrIOp>(*op->user_begin()));
  }
  if (isa<arith::AddIOp>(op)) {
    return true;
  }
  auto reads = readers(value);
  if (reads.size() == 1 && reads[0]->getBlock() == op->getBlock()) {
    // A single reader computes it as a temporary, unless that reader relabels
    // it.
    return succeeded(matchRotation(reads[0])) ||
           succeeded(matchShift(reads[0]));
  }
  return true;
}

void dropZeroBits(SmallVector<Statement>& statements) {
  SmallVector<Statement> result;
  for (auto& statement : statements) {
    SmallVector<Value> newOperands;
    size_t nulls = 0;
    for (auto operand : statement.operands) {
      if (!operand) {
        nulls += 1;
        continue;
      }
      newOperands.push_back(operand);
    }
    statement.operands = newOperands;
    if (statement.op == Statement::Op::Xor && newOperands.empty() &&
        !statement.invert) {
      continue;
    }
    if (statement.op == Statement::Op::And && nulls > 0) {
      continue;
    }
    if (statement.op == Statement::Op::Maj && nulls >= 2) {
      continue;
    }
    if (statement.op == Statement::Op::Maj && nulls == 1) {
      statement.op = Statement::Op::And;
    }
    if (statement.op == Statement::Op::Or && nulls >= 2) {
      continue;
    }
    if (statement.op == Statement::Op::Or && nulls == 1) {
      statement.op = Statement::Op::Xor;
    }
    result.push_back(statement);
  }
  statements = result;
}

void fuseXorTemporaries(SmallVector<Statement>& statements,
                        HelperAllocator& allocator) {
  for (size_t i = 0; i < statements.size(); ++i) {
    auto& statement = statements[i];
    if (statement.kind != Statement::Kind::Temp ||
        statement.op != Statement::Op::Xor) {
      continue;
    }
    if (llvm::any_of(statements, [&](auto& other) {
          return &other != &statement && statement.target == other.target;
        })) {
      continue;
    }
    BitVector readers(statements.size() - i - 1);
    bool found = false;
    for (size_t j = i + 1; j < statements.size(); ++j) {
      auto& reader = statements[j];
      if (!llvm::is_contained(reader.operands, statement.target)) {
        if (!found && llvm::is_contained(statement.operands, reader.target)) {
          break;
        }
        continue;
      }
      found = true;
      readers[j - i - 1] = true;
    }
    if (readers.count() != 1) {
      continue;
    }
    auto& user = statements[i + 1 + readers.find_first()];
    if (user.op != Statement::Op::Xor ||
        llvm::count(user.operands, statement.target) != 1) {
      continue;
    }

    llvm::erase(user.operands, statement.target);
    user.operands.append(statement.operands.begin(), statement.operands.end());
    user.invert ^= statement.invert;
    allocator.discard(statement.target);
    statements.erase(statements.begin() + i);
    --i;
  }
}

SmallVector<Statement>
buildStatements(const UpdateStep& update,
                DenseMap<Value, SmallVector<Value>>& storageMap,
                HelperAllocator& allocator) {
  SmallVector<Statement> statements;
  for (const auto& temporary : update.temporaries) {
    if (isa<arith::ExtUIOp, arith::TruncIOp>(temporary) ||
        succeeded(matchRotation(temporary)) ||
        succeeded(matchShift(temporary))) {
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
    auto constant = constantBits(update.otherValue);
    if (succeeded(constant)) {
      for (size_t i = 0; i < bitWidth; ++i) {
        if (!(*constant)[i]) {
          continue;
        }
        statements.push_back({.kind = Statement::Kind::Update,
                              .op = Statement::Op::Xor,
                              .target = storageMap.at(update.overwriteValue)[i],
                              .operands = {},
                              .invert = true});
      }
    } else {
      for (size_t i = 0; i < bitWidth; ++i) {
        statements.push_back(
            {.kind = Statement::Kind::Update,
             .op = Statement::Op::Xor,
             .target = storageMap.at(update.overwriteValue)[i],
             .operands = {storageMap.at(update.otherValue)[i]}});
      }
    }
  } else if (isa<arith::AddIOp>(update.updateOp)) {
    auto constant = constantBits(update.otherValue);
    if (succeeded(constant)) {
      statements.append(createConstantAdditionStatements(
          update.overwriteValue, *constant, Statement::Kind::Update, storageMap,
          allocator));
    } else {
      auto additionStatements = createAdditionStatements(
          update.overwriteValue, update.otherValue, Statement::Kind::Update,
          storageMap, allocator);
      statements.append(additionStatements.begin(), additionStatements.end());
    }
  } else {
    llvm_unreachable("Unexpected update operation");
  }
  dropZeroBits(statements);
  fuseXorTemporaries(statements, allocator);
  return statements;
}

void emitStatement(const Statement& statement, Location loc,
                   PatternRewriter& rewriter) {
  auto target = statement.target;
  if (statement.op == Statement::Op::And) {
    for (size_t i = 0; i < statement.negated.size(); ++i) {
      if (statement.negated[i]) {
        emitGate(loc, {}, statement.operands[i], rewriter);
      }
    }
    emitGate(loc, statement.operands, target, rewriter);
    for (size_t i = 0; i < statement.negated.size(); ++i) {
      if (statement.negated[i]) {
        emitGate(loc, {}, statement.operands[i], rewriter);
      }
    }
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
  } else if (statement.op == Statement::Op::Or) {
    auto x = statement.operands[0];
    auto y = statement.operands[1];
    emitGate(loc, {}, x, rewriter);
    emitGate(loc, {}, y, rewriter);
    emitGate(loc, {x, y}, statement.target, rewriter);
    emitGate(loc, {}, x, rewriter);
    emitGate(loc, {}, y, rewriter);
    emitGate(loc, {}, statement.target, rewriter);
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

  if (step.kind == UpdateStep::Kind::Create &&
      !storageMap.contains(step.overwriteValue)) {
    auto width = step.overwriteValue.getType().getIntOrFloatBitWidth();
    SmallVector<Value> reg;
    for (size_t i = 0; i < width; i++) {
      reg.push_back(qc::AllocOp::create(rewriter, loc));
    }
    storageMap[step.overwriteValue] = reg;
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
    auto temporaryBits = relabel(temporary, storageMap);
    if (succeeded(temporaryBits)) {
      storageMap[temporary->getResult(0)] = *temporaryBits;
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
  Block* block = rewriter.getInsertionBlock();
  auto end = rewriter.getInsertionPoint();
  Operation* marker = end == block->begin() ? nullptr : &*std::prev(end);

  SmallVector<Value> loopHelpers;
  emitUpdateStep(step, storageMap, loc, nullptr, loopHelpers, rewriter);

  auto first =
      (marker != nullptr) ? std::next(marker->getIterator()) : block->begin();
  SmallVector<Operation*> gates;
  for (auto& op : llvm::make_range(first, rewriter.getInsertionPoint())) {
    if (isa<qc::XOp, qc::CtrlOp>(op)) {
      gates.push_back(&op);
    }
  }
  if (gates.empty()) {
    return;
  }

  Operation* front = gates[0];
  for (auto* gate : ArrayRef(gates).drop_front()) {
    gate->moveBefore(front);
    front = gate;
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
      if (isFolded(defOp, operand)) {
        continue;
      }
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

SmallVector<Value> writtenContainers(const UpdateStep& step,
                                     DenseMap<Value, Value>& containers) {
  SmallVector<Value> result;
  if (auto forOp = dyn_cast<scf::ForOp>(step.updateOp)) {
    for (auto value : forOp.getInitArgs()) {
      result.push_back(containers.at(value));
    }
    return result;
  }
  return {containers.at(step.overwriteValue)};
}

DenseSet<Value> readContainers(const UpdateStep& step,
                               DenseMap<Value, Value>& containers) {
  DenseSet<Value> reads;
  if (step.otherValue && containers.contains(step.otherValue)) {
    reads.insert(containers[step.otherValue]);
  }
  for (auto* temporary : step.temporaries) {
    for (auto operand : getTemporaryInputs(temporary)) {
      if (!containers.contains(operand)) {
        continue;
      }
      reads.insert(containers.at(operand));
    }
  }
  if (step.kind == UpdateStep::Kind::Create) {
    for (auto input : getTemporaryInputs(step.updateOp)) {
      if (isFolded(step.updateOp, input) || !containers.contains(input)) {
        continue;
      }
      reads.insert(containers.at(input));
    }
  }
  return reads;
}

bool planCleanup(SmallVector<UpdateStep>& steps, OracleOp oracle,
                 DenseMap<Value, Value>& containers) {
  for (auto& step : steps) {
    if (llvm::any_of(writtenContainers(step, containers), [&](Value write) {
          return !oracle.isConsumedAlias(write);
        })) {
      step.cleanup = true;
      if (isa<scf::ForOp>(step.updateOp)) {
        return false;
      }
    }
  }
  for (size_t i = 0; i < steps.size(); ++i) {
    if (!steps[i].cleanup) {
      continue;
    }
    for (size_t j = i + 1; j < steps.size(); ++j) {
      if (steps[j].cleanup) {
        continue;
      }
      auto writesJ = writtenContainers(steps[j], containers);
      auto readsI = readContainers(steps[i], containers);
      if (llvm::any_of(writesJ,
                       [&](Value write) { return readsI.contains(write); })) {
        return false;
      }
    }
  }
  return true;
}

bool isReadByLaterStep(Operation* op, Value value,
                       DenseMap<Value, Value>& containers) {
  for (auto* user : value.getUsers()) {
    auto* ancestor = op->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor || ancestor == op) {
      continue;
    }
    if (!ancestor->isBeforeInBlock(op)) {
      return true;
    }
    if (ancestor->getNumResults() == 0 ||
        containers.contains(ancestor->getResult(0))) {
      continue;
    }
    if (isReadByLaterStep(op, ancestor->getResult(0), containers)) {
      return true;
    }
  }
  return false;
}

bool canOverwrite(Operation* op, Value value,
                  DenseMap<Value, Value>& containers) {
  if (!containers.contains(value)) {
    return false;
  }
  if (value.getParentBlock() != op->getBlock() || !isLastReader(op, value) ||
      isReadByLaterStep(op, value, containers)) {
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
          const DenseMap<Operation*, Value>& creations, bool allowBorrowing,
          bool allowRegisters);

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
                        creations, false, false);
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

UpdateStep buildCreationStep(Operation* op, Value value,
                             DenseMap<Value, Value>& containers) {
  SetVector<Operation*> allReachableTemporaries;
  for (auto input : getTemporaryInputs(op)) {
    if (isFolded(op, input)) {
      continue;
    }
    allReachableTemporaries.insert_range(
        reachableTemporaries(input, containers));
  }
  return UpdateStep{
      .updateOp = op,
      .overwriteValue = value,
      .otherValue = nullptr,
      .borrows = false,
      .temporaries = SmallVector<Operation*>(allReachableTemporaries.begin(),
                                             allReachableTemporaries.end()),
      .body = {},
      .kind = UpdateStep::Kind::Create,
  };
}

FailureOr<SmallVector<UpdateStep>>
planBlock(Block& block, OracleOp oracle, DenseMap<Value, Value>& containers,
          const DenseMap<Operation*, Value>& creations, bool allowBorrowing,
          bool allowRegisters) {
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
      updateSteps.push_back(buildCreationStep(&op, zero, containers));
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
    auto temporaries = isFolded(&op, other)
                           ? SmallVector<Operation*>{}
                           : reachableTemporaries(other, containers);
    if (!overwriteValue) {
      if (allowRegisters && needsRegister(&op)) {
        auto value = op.getResult(0);
        updateSteps.push_back(buildCreationStep(&op, value, containers));
        containers[value] = value;
      }
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

    if (step.kind != UpdateStep::Kind::Create) {
      overwritten.insert(step.overwriteValue);
    }
  }

  return updateSteps;
}

Value chainInput(Operation* op, OracleOp oracle) {
  if (!isa<arith::XOrIOp, arith::AddIOp>(op) && failed(matchRotation(op))) {
    return nullptr;
  }
  if (succeeded(matchMajority(op))) {
    return nullptr;
  }
  for (auto input : getTemporaryInputs(op)) {
    if (input.getDefiningOp<arith::ExtUIOp>() ||
        input.getDefiningOp<arith::TruncIOp>()) {
      continue;
    }
    if (input.getParentBlock() == op->getBlock() &&
        !oracle.isKeptAlias(input) && !isFolded(op, input) &&
        isLastReader(op, input)) {
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
    if (!isa<arith::XOrIOp, arith::AddIOp, arith::AndIOp, arith::ConstantOp,
             arith::OrIOp>(op) &&
        failed(matchShift(op)) && !isConstantEquality(op)) {
      oracle.emitOpError("cannot lower qoracle because an unsupported "
                         "operation result is created in a zero register");
      return failure();
    }
    creations[op] = consumed;
  }
  return creations;
}

FailureOr<SmallVector<UpdateStep>>
planWith(OracleOp oracle, bool allowBorrowing, bool allowRegisters,
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
                         *creations, allowBorrowing, allowRegisters);
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
    for (auto [registers, borrowing] : SmallVector<std::pair<bool, bool>>{
             {true, true}, {true, false}, {false, true}}) {
      ScopedDiagnosticHandler silence(oracle.getContext(),
                                      [](Diagnostic&) { return success(); });
      DenseMap<Value, Value> containers;
      auto steps = planWith(oracle, borrowing, registers, containers);
      if (succeeded(steps) && planCleanup(*steps, oracle, containers)) {
        return steps;
      }
    }
  }
  DenseMap<Value, Value> containers;
  auto steps = planWith(oracle, false, false, containers);
  if (failed(steps)) {
    return failure();
  }
  planCleanup(*steps, oracle, containers);
  return steps;
}

bool validRegion(Block* block) {
  if (block
          ->walk([&](Operation* op) {
            if (succeeded(matchShift(op))) {
              // A right shift by a constant is a relabeling.
              return WalkResult::advance();
            }
            if (isa<arith::ShLIOp, arith::ShRUIOp>(op) && op->hasOneUse() &&
                succeeded(matchRotation(*op->getUsers().begin()))) {
              // Shifts used inside a rotation are allowed.
              return WalkResult::advance();
            }
            if (isConstantEquality(op)) {
              // Constant equalities represent predicates
              return WalkResult::advance();
            }
            if (!isa<arith::ConstantOp, arith::XOrIOp, arith::AndIOp,
                     arith::AddIOp, arith::OrIOp, arith::ExtUIOp,
                     arith::TruncIOp, qoracle::YieldOp, scf::ForOp,
                     scf::YieldOp>(op)) {
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
    if (!step.cleanup) {
      continue;
    }
    emitInverseStep(step, storageMap, loc, rewriter);
    if (step.kind == UpdateStep::Kind::Create &&
        step.overwriteValue == step.updateOp->getResult(0)) {
      for (auto qubit : storageMap.at(step.overwriteValue)) {
        qc::DeallocOp::create(rewriter, loc, qubit);
      }
    }
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
