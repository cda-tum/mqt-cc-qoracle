/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qoracle_ir.cpp
/// Unit tests for the QOracle MLIR dialect.

#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>

using namespace mlir;

namespace {
class QOracleIRTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;
  std::unique_ptr<ScopedDiagnosticHandler> handler;
  std::string diagnostics;

  void SetUp() override {
    DialectRegistry registry;
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, qc::QCDialect, qco::QCODialect,
                    qoracle::QOracleDialect, scf::SCFDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
    handler = std::make_unique<ScopedDiagnosticHandler>(
        context.get(), [this](Diagnostic& diagnostic) {
          diagnostics += diagnostic.str();
          diagnostics += '\n';
          return success();
        });
  }

  /// Parse and verify a function body wrapped in `@main`.
  [[nodiscard]] OwningOpRef<ModuleOp> parseBody(StringRef body) {
    diagnostics.clear();
    const std::string source =
        "module {\nfunc.func @main() {\n" + body.str() + "\nreturn\n}\n}";
    return parseSourceString<ModuleOp>(source, context.get());
  }

  /// Expect that `body` fails to parse or verify with `message`.
  void expectRejected(StringRef body, StringRef message) {
    EXPECT_FALSE(parseBody(body)) << body.str();
    EXPECT_NE(diagnostics.find(message.str()), std::string::npos)
        << "missing '" << message.str() << "' in:\n"
        << diagnostics;
  }

  [[nodiscard]] static qoracle::OracleOp findOracle(ModuleOp moduleOp) {
    qoracle::OracleOp oracle;
    moduleOp.walk([&](qoracle::OracleOp op) { oracle = op; });
    return oracle;
  }

  [[nodiscard]] static std::string print(ModuleOp moduleOp) {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    moduleOp.print(stream);
    return printed;
  }
};

struct RoundTripCase {
  const char* name;
  const char* body;
  size_t numKept;
  size_t numConsumed;
};

TEST_F(QOracleIRTest, RoundTripsEveryOperandGroupCombination) {
  const std::array<RoundTripCase, 4> cases = {
      {
          {
              .name = "kept and consumed",
              .body = R"mlir(
         %ctrl = qc.alloc : !qc.qubit
         %reg = memref.alloc() : memref<8x!qc.qubit>
         qoracle.oracle kept(%ctrl : !qc.qubit as %k)
                        consumed(%reg : memref<8x!qc.qubit> as %r) {
           %step = arith.extui %k : i1 to i8
           %next = arith.addi %r, %step : i8
           qoracle.yield %next : i8
         })mlir",
              .numKept = 1,
              .numConsumed = 1,
          },
          {
              .name = "kept only",
              .body = R"mlir(
         %q = qc.alloc : !qc.qubit
         qoracle.oracle kept(%q : !qc.qubit as %k) {
           qoracle.yield
         })mlir",
              .numKept = 1,
              .numConsumed = 0,
          },
          {
              .name = "consumed only",
              .body = R"mlir(
         %reg = memref.alloc() : memref<4x!qc.qubit>
         qoracle.oracle consumed(%reg : memref<4x!qc.qubit> as %r) {
           %one = arith.constant 1 : i4
           %next = arith.addi %r, %one : i4
           qoracle.yield %next : i4
         })mlir",
              .numKept = 0,
              .numConsumed = 1,
          },
          {
              .name = "no operands",
              .body = R"mlir(
         qoracle.oracle {
           qoracle.yield
         })mlir",
              .numKept = 0,
              .numConsumed = 0,
          },
      },
  };

  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto moduleOp = parseBody(testCase.body);
    ASSERT_TRUE(moduleOp) << diagnostics;

    auto oracle = findOracle(*moduleOp);
    ASSERT_TRUE(oracle);
    EXPECT_EQ(oracle.getKeptAliases().size(), testCase.numKept);
    EXPECT_EQ(oracle.getConsumedAliases().size(), testCase.numConsumed);

    // Each group is printed only when it has operands.
    const auto printed = print(*moduleOp);
    EXPECT_EQ(printed.find("kept(") != std::string::npos, testCase.numKept != 0)
        << printed;
    EXPECT_EQ(printed.find("consumed(") != std::string::npos,
              testCase.numConsumed != 0)
        << printed;

    auto reparsed = parseSourceString<ModuleOp>(printed, context.get());
    ASSERT_TRUE(reparsed) << printed << diagnostics;
    EXPECT_EQ(print(*reparsed), printed);
  }
}

TEST_F(QOracleIRTest, RoundTripsZeroMarker) {
  auto moduleOp = parseBody(R"mlir(
    %a = qc.alloc : !qc.qubit
    %x = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : !qc.qubit as %va)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            zero %r : memref<2x!qc.qubit> as %vr,
                            zero %q : !qc.qubit as %vq) {
      qoracle.yield %vx, %vr, %vq : i2, i2, i1
    })mlir");
  ASSERT_TRUE(moduleOp) << diagnostics;
  auto oracle = findOracle(*moduleOp);
  ASSERT_TRUE(oracle);

  auto consumed = oracle.getConsumedAliases();
  EXPECT_FALSE(oracle.isZeroAlias(oracle.getKeptAliases()[0]));
  EXPECT_FALSE(oracle.isZeroAlias(consumed[0]));
  EXPECT_TRUE(oracle.isZeroAlias(consumed[1]));
  EXPECT_TRUE(oracle.isZeroAlias(consumed[2]));

  // The marker is printed inline, not as an attribute after the region.
  const auto printed = print(*moduleOp);
  EXPECT_EQ(printed.find("consumed_zeros"), std::string::npos) << printed;
  size_t markers = 0;
  for (auto pos = printed.find("zero %"); pos != std::string::npos;
       pos = printed.find("zero %", pos + 1)) {
    ++markers;
  }
  EXPECT_EQ(markers, 2U) << printed;

  auto reparsed = parseSourceString<ModuleOp>(printed, context.get());
  ASSERT_TRUE(reparsed) << printed << diagnostics;
  EXPECT_EQ(print(*reparsed), printed);
}

TEST_F(QOracleIRTest, OmitsZeroAttributeWithoutMarkers) {
  auto moduleOp = parseBody(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle consumed(%q : !qc.qubit as %vq) {
      qoracle.yield %vq : i1
    })mlir");
  ASSERT_TRUE(moduleOp) << diagnostics;
  auto oracle = findOracle(*moduleOp);
  ASSERT_TRUE(oracle);
  EXPECT_FALSE(oracle.getConsumedZerosAttr());
  EXPECT_FALSE(oracle.isZeroAlias(oracle.getConsumedAliases()[0]));
}

TEST_F(QOracleIRTest, AliasesFollowOperandOrder) {
  auto moduleOp = parseBody(R"mlir(
    %a = qc.alloc : !qc.qubit
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %d = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : !qc.qubit as %ka, %b : memref<3x!qc.qubit> as %kb)
                   consumed(%c : memref<2x!qc.qubit> as %cc, %d : !qc.qubit as %cd) {
      qoracle.yield %cc, %cd : i2, i1
    })mlir");
  ASSERT_TRUE(moduleOp) << diagnostics;
  auto oracle = findOracle(*moduleOp);
  ASSERT_TRUE(oracle);

  auto kept = oracle.getKeptAliases();
  auto consumed = oracle.getConsumedAliases();
  ASSERT_EQ(kept.size(), 2U);
  ASSERT_EQ(consumed.size(), 2U);

  Builder builder(context.get());
  EXPECT_EQ(kept[0].getType(), builder.getI1Type());
  EXPECT_EQ(kept[1].getType(), builder.getIntegerType(3));
  EXPECT_EQ(consumed[0].getType(), builder.getIntegerType(2));
  EXPECT_EQ(consumed[1].getType(), builder.getI1Type());
  EXPECT_EQ(kept[1].getArgNumber(), 1U);
  EXPECT_EQ(consumed[0].getArgNumber(), 2U);
}

TEST_F(QOracleIRTest, AliasTypeCoversQubitsAndStaticQubitMemRefs) {
  Builder builder(context.get());
  auto qubit = qc::QubitType::get(context.get());

  EXPECT_EQ(qoracle::OracleOp::getAliasType(qubit), builder.getI1Type());
  EXPECT_EQ(qoracle::OracleOp::getAliasType(MemRefType::get({8}, qubit)),
            builder.getIntegerType(8));

  EXPECT_FALSE(qoracle::OracleOp::getAliasType(
      MemRefType::get({ShapedType::kDynamic}, qubit)));
  EXPECT_FALSE(qoracle::OracleOp::getAliasType(MemRefType::get({2, 2}, qubit)));
  EXPECT_FALSE(qoracle::OracleOp::getAliasType(
      MemRefType::get({8}, builder.getI1Type())));
  EXPECT_FALSE(qoracle::OracleOp::getAliasType(builder.getI8Type()));
  EXPECT_FALSE(qoracle::OracleOp::getAliasType(
      MemRefType::get({int64_t{IntegerType::kMaxWidth} + 1}, qubit)));
}

TEST_F(QOracleIRTest, AcceptsCapturedClassicalValuesAndNestedControlFlow) {
  auto moduleOp = parseBody(R"mlir(
    %step = arith.constant 3 : i8
    %ctrl = qc.alloc : !qc.qubit
    %reg = memref.alloc() : memref<8x!qc.qubit>
    qoracle.oracle kept(%ctrl : !qc.qubit as %k)
                   consumed(%reg : memref<8x!qc.qubit> as %r) {
      %next = scf.if %k -> (i8) {
        %sum = arith.addi %r, %step : i8
        scf.yield %sum : i8
      } else {
        scf.yield %r : i8
      }
      qoracle.yield %next : i8
    })mlir");
  EXPECT_TRUE(moduleOp) << diagnostics;
}

TEST_F(QOracleIRTest, RejectsUnsupportedOperandTypes) {
  constexpr auto message = "expected a qubit or a static 1-D memref of qubits";
  expectRejected(R"mlir(
    %c4 = arith.constant 4 : index
    %reg = memref.alloc(%c4) : memref<?x!qc.qubit>
    qoracle.oracle kept(%reg : memref<?x!qc.qubit> as %k) {
      qoracle.yield
    })mlir",
                 message);
  expectRejected(R"mlir(
    %bits = memref.alloc() : memref<4xi1>
    qoracle.oracle kept(%bits : memref<4xi1> as %k) {
      qoracle.yield
    })mlir",
                 message);
}

TEST_F(QOracleIRTest, RejectsYieldCountMismatch) {
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle consumed(%q : !qc.qubit as %c) {
      qoracle.yield
    })mlir",
                 "must yield the same number of values (0) as consumed by the "
                 "parent qoracle.oracle (1)");
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(%q : !qc.qubit as %k) {
      qoracle.yield %k : i1
    })mlir",
                 "must yield the same number of values (1) as consumed by the "
                 "parent qoracle.oracle (0)");
}

TEST_F(QOracleIRTest, RejectsYieldWidthMismatch) {
  expectRejected(R"mlir(
    %reg = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%reg : memref<4x!qc.qubit> as %r) {
      %narrow = arith.trunci %r : i4 to i3
      qoracle.yield %narrow : i3
    })mlir",
                 "type of yielded value 0 ('i3') does not match the expected "
                 "alias type of consumed qubit 0 ('i4')");
}

// The custom parser derives the block arguments from the operands, so these
// mismatches are only reachable through the generic form or C++ builders.
TEST_F(QOracleIRTest, RejectsBlockArgumentMismatch) {
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    "qoracle.oracle"(%q) <{operandSegmentSizes = array<i32: 0, 1>}> ({
      %t = arith.constant true
      qoracle.yield %t : i1
    }) : (!qc.qubit) -> ())mlir",
                 "must have the same number of block arguments (0) as the sum "
                 "of kept qubits (0) and consumed qubits (1)");
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    "qoracle.oracle"(%q) <{operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%c: i2):
      qoracle.yield %c : i2
    }) : (!qc.qubit) -> ())mlir",
                 "type of block argument 0 ('i2') does not match the expected "
                 "alias type ('i1')");
}

TEST_F(QOracleIRTest, RejectsZeroMarkerOnKeptOperand) {
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(zero %q : !qc.qubit as %vq) {
      qoracle.yield
    })mlir",
                 "expected SSA operand");
}

// The custom parser creates one flag per consumed operand, so a wrong count is
// only reachable through the generic form or C++ builders.
TEST_F(QOracleIRTest, RejectsZeroFlagCountMismatch) {
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    "qoracle.oracle"(%q) <{consumed_zeros = array<i1: true, false>,
                           operandSegmentSizes = array<i32: 0, 1>}> ({
    ^bb0(%c: i1):
      qoracle.yield %c : i1
    }) : (!qc.qubit) -> ())mlir",
                 "must have one zero flag per consumed operand");
}

TEST_F(QOracleIRTest, RejectsDuplicateOperands) {
  constexpr auto message = "has duplicate operands";
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(%q : !qc.qubit as %k) consumed(%q : !qc.qubit as %c) {
      qoracle.yield %c : i1
    })mlir",
                 message);
  expectRejected(R"mlir(
    %reg = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle consumed(%reg : memref<2x!qc.qubit> as %a,
                            %reg : memref<2x!qc.qubit> as %b) {
      qoracle.yield %a, %b : i2, i2
    })mlir",
                 message);
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(%q : !qc.qubit as %a, %q : !qc.qubit as %b) {
      qoracle.yield
    })mlir",
                 message);
}

TEST_F(QOracleIRTest, RejectsOtherTerminators) {
  context->allowUnregisteredDialects();
  expectRejected(R"mlir(
    qoracle.oracle {
      "test.terminator"() : () -> ()
    })mlir",
                 "must have a qoracle.yield operation as the terminator");
}

TEST_F(QOracleIRTest, RejectsQuantumValuesInBody) {
  constexpr auto message =
      "must not use qubit operands or have side effects in its region";
  // Each body is free of side effects; only the type check rejects it.
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    %ctrl = qc.alloc : !qc.qubit
    qoracle.oracle kept(%ctrl : !qc.qubit as %k) {
      %s = arith.select %k, %q, %q : !qc.qubit
      qoracle.yield
    })mlir",
                 message);
  expectRejected(R"mlir(
    %reg = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle {
      %c0 = arith.constant 0 : index
      %size = memref.dim %reg, %c0 : memref<4x!qc.qubit>
      qoracle.yield
    })mlir",
                 message);
  expectRejected(R"mlir(
    qoracle.oracle {
      %q = qco.static 0 : !qco.qubit
      %q1 = qco.x %q : !qco.qubit
      qoracle.yield
    })mlir",
                 message);
  // A QC gate on a captured qubit.
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    qoracle.oracle {
      qc.x %q : !qc.qubit
      qoracle.yield
    })mlir",
                 message);
}

TEST_F(QOracleIRTest, RejectsSideEffectsInBody) {
  expectRejected(R"mlir(
    %buffer = memref.alloc() : memref<1xi8>
    %q = qc.alloc : !qc.qubit
    qoracle.oracle kept(%q : !qc.qubit as %k) {
      %c0 = arith.constant 0 : index
      %v = arith.extui %k : i1 to i8
      memref.store %v, %buffer[%c0] : memref<1xi8>
      qoracle.yield
    })mlir",
                 "must not use qubit operands or have side effects in its "
                 "region");
}
} // namespace
