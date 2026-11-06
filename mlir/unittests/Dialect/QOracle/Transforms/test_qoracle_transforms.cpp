/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qoracle_transforms.cpp
/// Unit tests for QOracle dialect transformations.

#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"
#include "mqt/Dialect/QOracle/Transforms/Passes.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

namespace {
using PassFactory = std::unique_ptr<Pass> (*)();

/// Largest number of input bits for which the truth table is enumerated.
constexpr unsigned MAX_INPUT_BITS = 20;

/// The `arith.addi` operations of the sum that defines `root`, descending
/// through operands that are single-use additions, and the sum's terms.
struct Sum {
  SmallVector<arith::AddIOp> additions;
  SmallVector<Value> terms;
};
} // namespace

[[nodiscard]] static std::string print(Operation* op) {
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  op->print(stream);
  return printed;
}

[[nodiscard]] static qoracle::OracleOp findOracle(ModuleOp moduleOp) {
  qoracle::OracleOp oracle;
  moduleOp.walk([&](qoracle::OracleOp op) { oracle = op; });
  return oracle;
}

using Env = DenseMap<Value, APInt>;

[[nodiscard]] static FailureOr<SmallVector<APInt>> evaluateBlock(Block& block,
                                                                 Env& env);

/// Evaluate the single-result classical operation `op` on the values in
/// `env`, computing operands defined outside the oracle on demand.
[[nodiscard]] static FailureOr<APInt> evaluateOp(Operation& op, Env& env);

/// Look up `value`, computing it on demand if it is defined outside the
/// oracle, for example a captured constant or sum.
[[nodiscard]] static FailureOr<APInt> lookup(Value value, Env& env) {
  if (auto it = env.find(value); it != env.end()) {
    return it->second;
  }
  auto* op = value.getDefiningOp();
  if (op == nullptr) {
    ADD_FAILURE() << "block argument has no assignment";
    return failure();
  }
  auto result = evaluateOp(*op, env);
  if (succeeded(result)) {
    env[value] = *result;
  }
  return result;
}

static FailureOr<APInt> evaluateOp(Operation& op, Env& env) {
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    return cast<IntegerAttr>(constant.getValue()).getValue();
  }
  SmallVector<APInt, 2> operands;
  for (auto operand : op.getOperands()) {
    auto value = lookup(operand, env);
    if (failed(value)) {
      return failure();
    }
    operands.push_back(*value);
  }
  const auto width = op.getResult(0).getType().getIntOrFloatBitWidth();
  if (isa<arith::ExtUIOp>(op)) {
    return operands[0].zext(width);
  }
  if (isa<arith::TruncIOp>(op)) {
    return operands[0].trunc(width);
  }
  if (isa<arith::AddIOp>(op)) {
    return operands[0] + operands[1];
  }
  if (isa<arith::SubIOp>(op)) {
    return operands[0] - operands[1];
  }
  if (isa<arith::XOrIOp>(op)) {
    return operands[0] ^ operands[1];
  }
  if (isa<arith::AndIOp>(op)) {
    return operands[0] & operands[1];
  }
  if (isa<arith::OrIOp>(op)) {
    return operands[0] | operands[1];
  }
  ADD_FAILURE() << "unsupported operation "
                << op.getName().getStringRef().str();
  return failure();
}

/// Evaluate `block` on the values in `env` and return its yielded values.
/// Supports the classical operations these tests use.
static FailureOr<SmallVector<APInt>> evaluateBlock(Block& block, Env& env) {
  for (auto& op : block) {
    if (isa<qoracle::YieldOp, scf::YieldOp>(op)) {
      SmallVector<APInt> results;
      for (auto operand : op.getOperands()) {
        auto value = lookup(operand, env);
        if (failed(value)) {
          return failure();
        }
        results.push_back(*value);
      }
      return results;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      auto condition = lookup(ifOp.getCondition(), env);
      if (failed(condition)) {
        return failure();
      }
      auto& branch = condition->getBoolValue() ? ifOp.getThenRegion().front()
                                               : ifOp.getElseRegion().front();
      auto results = evaluateBlock(branch, env);
      if (failed(results)) {
        return failure();
      }
      for (auto [result, value] :
           llvm::zip_equal(ifOp.getResults(), *results)) {
        env[result] = value;
      }
      continue;
    }
    auto result = evaluateOp(op, env);
    if (failed(result)) {
      return failure();
    }
    env[op.getResult(0)] = *result;
  }
  ADD_FAILURE() << "block has no yield";
  return failure();
}

/// Tabulate the yielded values of the module's oracle for every assignment to
/// its aliases and to the arguments of the enclosing function, in that order.
[[nodiscard]] static std::vector<SmallVector<APInt>>
truthTable(ModuleOp moduleOp) {
  auto oracle = findOracle(moduleOp);
  auto function = oracle->getParentOfType<func::FuncOp>();
  SmallVector<Value> inputs(oracle.getRegion().getArguments());
  llvm::append_range(inputs, function.getArguments());

  unsigned totalBits = 0;
  for (auto input : inputs) {
    totalBits += input.getType().getIntOrFloatBitWidth();
  }
  if (totalBits > MAX_INPUT_BITS) {
    ADD_FAILURE() << "too many input bits to enumerate: " << totalBits;
    return {};
  }

  std::vector<SmallVector<APInt>> table;
  for (uint64_t code = 0; code < (uint64_t{1} << totalBits); ++code) {
    Env env;
    unsigned offset = 0;
    for (auto input : inputs) {
      const auto width = input.getType().getIntOrFloatBitWidth();
      env[input] =
          APInt(width, (code >> offset) & ((uint64_t{1} << width) - 1));
      offset += width;
    }
    auto results = evaluateBlock(oracle.getRegion().front(), env);
    if (failed(results)) {
      return {};
    }
    table.push_back(std::move(*results));
  }
  return table;
}

static void collectSum(Value value, Sum& sum) {
  auto addition = value.getDefiningOp<arith::AddIOp>();
  if (!addition) {
    sum.terms.push_back(value);
    return;
  }
  sum.additions.push_back(addition);
  for (auto operand : addition->getOperands()) {
    if (operand.getDefiningOp<arith::AddIOp>() && operand.hasOneUse()) {
      collectSum(operand, sum);
    } else {
      sum.terms.push_back(operand);
    }
  }
}

[[nodiscard]] static Sum collectSum(Value root) {
  Sum sum;
  collectSum(root, sum);
  return sum;
}

/// Check whether an innermost addition of `sum`, one that reads no other
/// addition of `sum`, has `host` as an operand.
[[nodiscard]] static bool startsAt(const Sum& sum, Value host) {
  SmallPtrSet<Operation*, 8> additions;
  for (auto addition : sum.additions) {
    additions.insert(addition);
  }
  return llvm::any_of(sum.additions, [&](arith::AddIOp addition) {
    const bool innermost = llvm::none_of(addition->getOperands(), [&](Value v) {
      return additions.contains(v.getDefiningOp());
    });
    return innermost && llvm::is_contained(addition->getOperands(), host);
  });
}

/// Check whether `sum` is a single chain: no addition of `sum` adds two
/// other additions of `sum`, so each one adds a single new term.
[[nodiscard]] static bool isSingleChain(const Sum& sum) {
  SmallPtrSet<Operation*, 8> additions;
  for (auto addition : sum.additions) {
    additions.insert(addition);
  }
  return llvm::all_of(sum.additions, [&](arith::AddIOp addition) {
    return llvm::count_if(addition->getOperands(), [&](Value v) {
             return additions.contains(v.getDefiningOp());
           }) <= 1;
  });
}

[[nodiscard]] static Value yieldedValue(qoracle::OracleOp oracle) {
  return oracle.getRegion().front().getTerminator()->getOperand(0);
}

namespace {
class QOracleTransformsTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                qc::QCDialect, qoracle::QOracleDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
  }

  [[nodiscard]] OwningOpRef<ModuleOp> parse(StringRef source) const {
    return parseSourceString<ModuleOp>(source, context.get());
  }

  [[nodiscard]] bool runPass(ModuleOp moduleOp, PassFactory createPass) const {
    PassManager manager(context.get());
    manager.addPass(createPass());
    return succeeded(manager.run(moduleOp));
  }

  /// Run the pass on `source` and check the contract every test shares: the
  /// pass succeeds, its output verifies, the oracle computes the same function,
  /// and a second run changes nothing.
  [[nodiscard]] OwningOpRef<ModuleOp> transform(StringRef source,
                                                PassFactory createPass) const {
    auto original = parse(source);
    auto moduleOp = parse(source);
    if (!original || !moduleOp) {
      ADD_FAILURE() << "test input does not parse";
      return {};
    }
    if (!runPass(*moduleOp, createPass)) {
      ADD_FAILURE() << "pass failed on valid IR";
      return {};
    }
    if (failed(verify(*moduleOp))) {
      ADD_FAILURE() << "pass output does not verify";
      return {};
    }
    EXPECT_EQ(truthTable(*original), truthTable(*moduleOp))
        << "pass changed the computed function:\n"
        << print(*moduleOp);

    const auto once = print(*moduleOp);
    auto again = parse(once);
    EXPECT_TRUE(again && runPass(*again, createPass));
    if (again) {
      EXPECT_EQ(print(*again), once) << "pass is not idempotent";
    }
    return moduleOp;
  }

  /// Expect the pass to leave `source` unchanged.
  void expectUnchanged(StringRef source, PassFactory createPass) const {
    auto moduleOp = transform(source, createPass);
    ASSERT_TRUE(moduleOp);
    EXPECT_EQ(print(*moduleOp), print(*parse(source)));
  }
};

//===----------------------------------------------------------------------===//
// qoracle-reassociate
//===----------------------------------------------------------------------===//

constexpr PassFactory REASSOCIATE = &qoracle::createReassociateOracle;

TEST_F(QOracleTransformsTest, ReassociateStartsSumAtConsumedAlias) {
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %A, %B : i3
        %s = arith.addi %t, %C : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto sum = collectSum(yieldedValue(oracle));
  EXPECT_EQ(sum.additions.size(), 2U);
  EXPECT_TRUE(startsAt(sum, oracle.getConsumedAliases()[0])) << print(oracle);
}

TEST_F(QOracleTransformsTest, ReassociateStartsSumAtSingleUseTemporary) {
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.andi %A, %B : i3
        %u = arith.addi %A, %B : i3
        %s = arith.addi %u, %t : i3
        %r = arith.xori %C, %s : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  arith::AndIOp temporary;
  oracle.walk([&](arith::AndIOp op) { temporary = op; });
  ASSERT_TRUE(temporary);
  auto xorOp = yieldedValue(oracle).getDefiningOp<arith::XOrIOp>();
  ASSERT_TRUE(xorOp);
  auto sum = collectSum(xorOp.getRhs());
  EXPECT_EQ(sum.additions.size(), 2U);
  EXPECT_TRUE(startsAt(sum, temporary)) << print(oracle);
}

// A single-use temporary has no qubits of its own; the lowering computes it
// into fresh helpers. So a sum starts at it only when no other term can host,
// as in SHA-256's W_t = ((s1 + W_{t-7}) + s0) + W_{t-16}.
TEST_F(QOracleTransformsTest, ReassociatePrefersConsumedAliasOverTemporary) {
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %d = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B,
                          %d : memref<3x!qc.qubit> as %D)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %x = arith.xori %A, %B : i3
        %y = arith.xori %A, %D : i3
        %s1 = arith.addi %x, %B : i3
        %s2 = arith.addi %s1, %y : i3
        %s3 = arith.addi %s2, %C : i3
        qoracle.yield %s3 : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto sum = collectSum(yieldedValue(oracle));
  EXPECT_EQ(sum.additions.size(), 3U);
  EXPECT_TRUE(isSingleChain(sum)) << print(oracle);
  EXPECT_TRUE(startsAt(sum, oracle.getConsumedAliases()[0])) << print(oracle);
}

// The preferred host need not be an alias: `%w` is a sum with a second reader,
// stored in the qubits of `%C`. In SHA-256, W_{t-16} for t >= 32 is such a sum.
TEST_F(QOracleTransformsTest, ReassociatePrefersSharedSumOverTemporary) {
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      %d = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C,
                              %d : memref<3x!qc.qubit> as %D) {
        %w = arith.addi %C, %A : i3
        %v = arith.xori %D, %w : i3
        %x = arith.xori %A, %B : i3
        %s1 = arith.addi %x, %B : i3
        %s2 = arith.addi %s1, %w : i3
        qoracle.yield %s2, %v : i3, i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto shared = oracle.getConsumedAliases()[0].getUsers().begin()->getResult(0);
  ASSERT_TRUE(shared.getDefiningOp<arith::AddIOp>());
  auto sum = collectSum(yieldedValue(oracle));
  EXPECT_EQ(sum.additions.size(), 2U);
  EXPECT_TRUE(startsAt(sum, shared)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ReassociateNeverStartsSumAtKeptAlias) {
  // Every term is a kept alias or a captured value, so none may be overwritten.
  expectUnchanged(R"mlir(
    func.func @main(%x: i3) {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %u = arith.addi %A, %B : i3
        %s = arith.addi %u, %x : i3
        %r = arith.xori %C, %s : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                  REASSOCIATE);
}

TEST_F(QOracleTransformsTest,
       ReassociateStartsAtHostWhoseOtherReadersAreIndependent) {
  // %q also reads %C but does not depend on the sum, so the sum can be the
  // last reader of %C.
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %A, %B : i3
        %s = arith.addi %t, %C : i3
        %q = arith.andi %C, %B : i3
        %r = arith.xori %q, %s : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto xorOp = yieldedValue(oracle).getDefiningOp<arith::XOrIOp>();
  ASSERT_TRUE(xorOp);
  auto root = xorOp.getLhs().getDefiningOp<arith::AddIOp>() ? xorOp.getLhs()
                                                            : xorOp.getRhs();
  auto sum = collectSum(root);
  EXPECT_EQ(sum.additions.size(), 2U);
  EXPECT_TRUE(startsAt(sum, oracle.getConsumedAliases()[0])) << print(oracle);
}

TEST_F(QOracleTransformsTest, ReassociateSkipsHostReadAfterTheSum) {
  // %q reads %C and depends on the sum, so %C is still live after the sum and
  // cannot be overwritten by it.
  expectUnchanged(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %A, %B : i3
        %s = arith.addi %t, %C : i3
        %q = arith.andi %C, %s : i3
        qoracle.yield %q : i3
      }
      return
    })mlir",
                  REASSOCIATE);
}

TEST_F(QOracleTransformsTest, ReassociateKeepsSharedIntermediateSums) {
  // %t has two uses, so it is a term of %s rather than part of its chain, and
  // %s already starts at %C.
  expectUnchanged(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %A, %B : i3
        %s = arith.addi %t, %C : i3
        %w = arith.xori %t, %s : i3
        qoracle.yield %w : i3
      }
      return
    })mlir",
                  REASSOCIATE);
}

TEST_F(QOracleTransformsTest, ReassociateFoldsConstantTerms) {
  auto moduleOp = transform(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %one = arith.constant 1 : i3
        %two = arith.constant 2 : i3
        %t = arith.addi %C, %one : i3
        %u = arith.addi %t, %A : i3
        %s = arith.addi %u, %two : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto sum = collectSum(yieldedValue(oracle));
  EXPECT_EQ(sum.additions.size(), 2U) << print(oracle);
  EXPECT_TRUE(startsAt(sum, oracle.getConsumedAliases()[0])) << print(oracle);

  SmallVector<APInt> constants;
  for (auto term : sum.terms) {
    if (auto constant = term.getDefiningOp<arith::ConstantOp>()) {
      constants.push_back(cast<IntegerAttr>(constant.getValue()).getValue());
    }
  }
  ASSERT_EQ(constants.size(), 1U) << print(oracle);
  EXPECT_EQ(constants.front(), APInt(3, 3));
}

TEST_F(QOracleTransformsTest, ReassociateLeavesNormalSumsUnchanged) {
  expectUnchanged(R"mlir(
    func.func @main() {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %C, %A : i3
        %s = arith.addi %t, %B : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                  REASSOCIATE);
}

TEST_F(QOracleTransformsTest, ReassociateFlattensPartialSumsIntoOneChain) {
  // `%B + %x` adds two values that must not be overwritten, so it would need a
  // fresh register; the single chain adds every term into `%C`.
  auto moduleOp = transform(R"mlir(
    func.func @main(%x: i3) {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %l = arith.addi %C, %A : i3
        %r = arith.addi %B, %x : i3
        %s = arith.addi %l, %r : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto sum = collectSum(yieldedValue(oracle));
  EXPECT_EQ(sum.additions.size(), 3U) << print(oracle);
  EXPECT_TRUE(isSingleChain(sum)) << print(oracle);
  EXPECT_TRUE(startsAt(sum, oracle.getConsumedAliases()[0])) << print(oracle);
}

TEST_F(QOracleTransformsTest, ReassociateFlattensPartialSumsWithoutHost) {
  // No term may be overwritten, but one chain still needs one fresh register
  // where the two partial sums need two.
  auto moduleOp = transform(R"mlir(
    func.func @main(%x: i3, %y: i3) {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %l = arith.addi %A, %B : i3
        %r = arith.addi %x, %y : i3
        %s = arith.addi %l, %r : i3
        %t = arith.xori %C, %s : i3
        qoracle.yield %t : i3
      }
      return
    })mlir",
                            REASSOCIATE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  auto xorOp = yieldedValue(oracle).getDefiningOp<arith::XOrIOp>();
  ASSERT_TRUE(xorOp);
  auto sum = collectSum(xorOp.getRhs());
  EXPECT_EQ(sum.additions.size(), 3U) << print(oracle);
  EXPECT_TRUE(isSingleChain(sum)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ReassociateLeavesCodeOutsideOraclesUnchanged) {
  // The sum and the foldable constant outside the oracle must stay as written.
  expectUnchanged(R"mlir(
    func.func @main(%x: i3, %y: i3, %z: i3) {
      %one = arith.constant 1 : i3
      %two = arith.addi %one, %one : i3
      %o = arith.addi %x, %y : i3
      %p = arith.addi %o, %z : i3
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.addi %C, %p : i3
        %s = arith.addi %t, %two : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                  REASSOCIATE);
}

//===----------------------------------------------------------------------===//
// qoracle-schedule
//===----------------------------------------------------------------------===//

constexpr PassFactory SCHEDULE = &qoracle::createScheduleOracle;

TEST_F(QOracleTransformsTest, ScheduleMovesNonOverwritingReaderFirst) {
  // Only the last reader of %C can overwrite it, so the AND, which never
  // overwrites an operand, moves before the addition.
  auto moduleOp = transform(R"mlir(
    func.func @main(%k: i3, %m: i3) {
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle consumed(%c : memref<3x!qc.qubit> as %C) {
        %p = arith.addi %C, %k : i3
        %q = arith.andi %C, %m : i3
        %r = arith.xori %p, %q : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                            SCHEDULE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  arith::AddIOp addition;
  arith::AndIOp conjunction;
  oracle.walk([&](arith::AddIOp op) { addition = op; });
  oracle.walk([&](arith::AndIOp op) { conjunction = op; });
  ASSERT_TRUE(addition && conjunction);
  EXPECT_TRUE(conjunction->isBeforeInBlock(addition)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ScheduleLetsTemporaryBeOverwritten) {
  auto moduleOp = transform(R"mlir(
    func.func @main(%k: i3, %m: i3) {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %b = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A,
                          %b : memref<3x!qc.qubit> as %B)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %t = arith.andi %A, %B : i3
        %u = arith.addi %t, %k : i3
        %v = arith.andi %t, %m : i3
        %r = arith.xori %u, %v : i3
        %s = arith.xori %C, %r : i3
        qoracle.yield %s : i3
      }
      return
    })mlir",
                            SCHEDULE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  arith::AddIOp addition;
  arith::AndIOp reader;
  oracle.walk([&](arith::AddIOp op) { addition = op; });
  oracle.walk([&](arith::AndIOp op) {
    if (op.getLhs().getDefiningOp<arith::AndIOp>()) {
      reader = op;
    }
  });
  ASSERT_TRUE(addition && reader);
  EXPECT_TRUE(reader->isBeforeInBlock(addition)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ScheduleKeepsOrderWhenNoReaderCanOverwrite) {
  // %A is kept and %x is captured, so no reader may overwrite them, and ties
  // keep the original order.
  expectUnchanged(R"mlir(
    func.func @main(%x: i3, %m: i3) {
      %a = memref.alloc() : memref<3x!qc.qubit>
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%a : memref<3x!qc.qubit> as %A)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %u = arith.addi %A, %x : i3
        %v = arith.andi %A, %m : i3
        %w = arith.addi %x, %m : i3
        %y = arith.andi %x, %A : i3
        %r = arith.xori %u, %v : i3
        %q = arith.xori %w, %y : i3
        %s = arith.xori %r, %q : i3
        %t = arith.xori %C, %s : i3
        qoracle.yield %t : i3
      }
      return
    })mlir",
                  SCHEDULE);
}

TEST_F(QOracleTransformsTest, ScheduleTreatsNestedReadsAsReadsOfTheParent) {
  // The scf.if reads %C inside its then-branch, so it must run before the
  // addition can be the last reader of %C.
  auto moduleOp = transform(R"mlir(
    func.func @main(%k: i3, %m: i3) {
      %q = qc.alloc : !qc.qubit
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%q : !qc.qubit as %K)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %p = arith.addi %C, %k : i3
        %n = scf.if %K -> (i3) {
          %v = arith.andi %C, %m : i3
          scf.yield %v : i3
        } else {
          scf.yield %m : i3
        }
        %r = arith.xori %p, %n : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                            SCHEDULE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  scf::IfOp ifOp;
  arith::AddIOp addition;
  oracle.walk([&](scf::IfOp op) { ifOp = op; });
  oracle.walk([&](arith::AddIOp op) { addition = op; });
  ASSERT_TRUE(ifOp && addition);
  EXPECT_TRUE(ifOp->isBeforeInBlock(addition)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ScheduleKeepsNestedUsesAfterTheirDefinitions) {
  // %p waits for %q, and the scf.if cannot overwrite anything, but its branch
  // uses %p, so it must stay after %p.
  auto moduleOp = transform(R"mlir(
    func.func @main(%k: i3, %m: i3) {
      %q = qc.alloc : !qc.qubit
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%q : !qc.qubit as %K)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %p = arith.addi %C, %k : i3
        %n = scf.if %K -> (i3) {
          %v = arith.andi %p, %m : i3
          scf.yield %v : i3
        } else {
          scf.yield %m : i3
        }
        %w = arith.andi %C, %m : i3
        %r = arith.xori %n, %w : i3
        qoracle.yield %r : i3
      }
      return
    })mlir",
                            SCHEDULE);
  ASSERT_TRUE(moduleOp);
  auto oracle = findOracle(*moduleOp);
  scf::IfOp ifOp;
  arith::AddIOp addition;
  oracle.walk([&](scf::IfOp op) { ifOp = op; });
  oracle.walk([&](arith::AddIOp op) { addition = op; });
  ASSERT_TRUE(ifOp && addition);
  EXPECT_TRUE(addition->isBeforeInBlock(ifOp)) << print(oracle);
}

TEST_F(QOracleTransformsTest, ScheduleKeepsOperationsInTheirBlocks) {
  const auto* source = R"mlir(
    func.func @main(%k: i3, %m: i3) {
      %q = qc.alloc : !qc.qubit
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle kept(%q : !qc.qubit as %K)
                     consumed(%c : memref<3x!qc.qubit> as %C) {
        %n = scf.if %K -> (i3) {
          %p = arith.addi %C, %k : i3
          %v = arith.andi %C, %m : i3
          %r = arith.xori %p, %v : i3
          scf.yield %r : i3
        } else {
          %w = arith.andi %C, %k : i3
          scf.yield %w : i3
        }
        %s = arith.xori %C, %n : i3
        qoracle.yield %s : i3
      }
      return
    })mlir";

  // The sorted operation names of every block in the oracle, in block order.
  auto blockContents = [](ModuleOp moduleOp) {
    std::vector<std::vector<std::string>> contents;
    findOracle(moduleOp).walk([&](Block* block) {
      auto& names = contents.emplace_back();
      for (auto& op : *block) {
        names.push_back(op.getName().getStringRef().str());
      }
      llvm::sort(names);
    });
    return contents;
  };

  auto moduleOp = transform(source, SCHEDULE);
  ASSERT_TRUE(moduleOp);
  EXPECT_EQ(blockContents(*moduleOp), blockContents(*parse(source)));
}

TEST_F(QOracleTransformsTest, ScheduleLeavesCodeOutsideOraclesUnchanged) {
  // Outside oracles, reordering would be a plain change of the program.
  expectUnchanged(R"mlir(
    func.func @main(%x: i3, %k: i3, %m: i3) -> (i3, i3) {
      %p = arith.addi %x, %k : i3
      %q = arith.andi %x, %m : i3
      %c = memref.alloc() : memref<3x!qc.qubit>
      qoracle.oracle consumed(%c : memref<3x!qc.qubit> as %C) {
        %s = arith.xori %C, %k : i3
        qoracle.yield %s : i3
      }
      return %p, %q : i3, i3
    })mlir",
                  SCHEDULE);
}
} // namespace
