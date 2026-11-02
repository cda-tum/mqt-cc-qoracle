/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file test_qoracle_to_qc.cpp
/// Unit tests for the QOracle-to-QC conversion.

#include "mqt/Conversion/QOracleToQC/QOracleToQC.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
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
/// Largest number of register bits for which every basis state is simulated.
constexpr unsigned MAX_INPUT_BITS = 16;

/// The value of each register, bit 0 being the qubit at index 0.
using RegisterValues = SmallVector<uint64_t>;

/// Simulation of a circuit of X gates with controls on a computational basis
/// state, where each qubit holds a single bit.
class BasisSimulator {
public:
  /// Run the entry block of `function`.
  ///
  /// Each allocation in `inputs` starts with the matching value of `initial`;
  /// every other qubit starts at 0. Fails if a qubit is used after its
  /// deallocation or is not 0 when it is deallocated. Returns the final value
  /// of each input allocation.
  [[nodiscard]] FailureOr<RegisterValues> run(func::FuncOp function,
                                              ArrayRef<Operation*> inputs,
                                              ArrayRef<uint64_t> initial);

private:
  std::vector<bool> bits_;
  std::vector<bool> live_;
  DenseMap<Value, size_t> qubits_;
  DenseMap<Value, SmallVector<size_t>> registers_;
  DenseMap<Value, int64_t> indices_;

  [[nodiscard]] SmallVector<size_t> allocate(size_t width, uint64_t value);
  [[nodiscard]] FailureOr<size_t> bitOf(Value qubit);
  [[nodiscard]] LogicalResult apply(Operation& op);
};
} // namespace

SmallVector<size_t> BasisSimulator::allocate(const size_t width,
                                             const uint64_t value) {
  SmallVector<size_t> allocated;
  for (size_t i = 0; i < width; ++i) {
    allocated.push_back(bits_.size());
    bits_.push_back(((value >> i) & 1U) != 0);
    live_.push_back(true);
  }
  return allocated;
}

FailureOr<size_t> BasisSimulator::bitOf(Value qubit) {
  const auto bit = qubits_.at(qubit);
  if (!live_[bit]) {
    ADD_FAILURE() << "qubit is used after its deallocation";
    return failure();
  }
  return bit;
}

LogicalResult BasisSimulator::apply(Operation& op) {
  if (auto x = dyn_cast<qc::XOp>(op)) {
    auto bit = bitOf(x.getQubitIn());
    if (failed(bit)) {
      return failure();
    }
    bits_[*bit] = !bits_[*bit];
    return success();
  }
  if (auto ctrl = dyn_cast<qc::CtrlOp>(op)) {
    for (auto control : ctrl.getControls()) {
      auto bit = bitOf(control);
      if (failed(bit)) {
        return failure();
      }
      if (!bits_[*bit]) {
        return success();
      }
    }
    auto& body = ctrl.getRegion().front();
    for (auto [argument, target] :
         llvm::zip_equal(body.getArguments(), ctrl.getTargets())) {
      const auto bit = qubits_.at(target);
      qubits_[argument] = bit;
    }
    for (auto& nested : body.without_terminator()) {
      if (failed(apply(nested))) {
        return failure();
      }
    }
    return success();
  }
  ADD_FAILURE() << "unsupported operation "
                << op.getName().getStringRef().str();
  return failure();
}

FailureOr<RegisterValues> BasisSimulator::run(func::FuncOp function,
                                              ArrayRef<Operation*> inputs,
                                              ArrayRef<uint64_t> initial) {
  DenseMap<Operation*, uint64_t> initialValues;
  for (auto [input, value] : llvm::zip_equal(inputs, initial)) {
    initialValues[input] = value;
  }

  DenseMap<Operation*, SmallVector<size_t>> inputBits;
  for (auto& op : function.getBody().front()) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto allocated =
          allocate(alloc.getType().getDimSize(0), initialValues.lookup(&op));
      registers_[alloc.getResult()] = allocated;
      inputBits[&op] = allocated;
    } else if (auto alloc = dyn_cast<qc::AllocOp>(op)) {
      auto allocated = allocate(1, initialValues.lookup(&op));
      qubits_[alloc.getResult()] = allocated.front();
      inputBits[&op] = allocated;
    } else if (auto constant = dyn_cast<arith::ConstantIndexOp>(op)) {
      indices_[constant.getResult()] = constant.value();
    } else if (isa<arith::ConstantOp>(op)) {
      // Constants captured by an oracle stay behind and hold no qubits.
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      const auto index = indices_.at(load.getIndices().front());
      const auto bit = registers_.at(load.getMemRef())[index];
      qubits_[load.getResult()] = bit;
    } else if (auto dealloc = dyn_cast<qc::DeallocOp>(op)) {
      auto bit = bitOf(dealloc.getQubit());
      if (failed(bit)) {
        return failure();
      }
      if (bits_[*bit]) {
        ADD_FAILURE() << "qubit is not 0 when it is deallocated";
        return failure();
      }
      live_[*bit] = false;
    } else if (isa<func::ReturnOp>(op)) {
      break;
    } else if (failed(apply(op))) {
      return failure();
    }
  }

  RegisterValues final;
  for (auto* input : inputs) {
    uint64_t value = 0;
    for (auto [i, bit] : llvm::enumerate(inputBits.at(input))) {
      if (bits_[bit]) {
        value |= uint64_t{1} << i;
      }
    }
    final.push_back(value);
  }
  return final;
}

namespace {
class QOracleToQCTest : public ::testing::Test {
protected:
  std::unique_ptr<MLIRContext> context;
  std::unique_ptr<ScopedDiagnosticHandler> handler;
  std::string diagnostics;

  void SetUp() override {
    DialectRegistry registry;
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                qc::QCDialect, qoracle::QOracleDialect>();
    context = std::make_unique<MLIRContext>(registry);
    context->loadAllAvailableDialects();
    handler = std::make_unique<ScopedDiagnosticHandler>(
        context.get(), [this](Diagnostic& diagnostic) {
          diagnostics += diagnostic.str();
          diagnostics += '\n';
          return success();
        });
  }

  /// Parse a function body wrapped in `@main`, which takes `arguments`.
  [[nodiscard]] OwningOpRef<ModuleOp> parseBody(StringRef body,
                                                StringRef arguments = "") {
    diagnostics.clear();
    const std::string source = "module {\nfunc.func @main(" + arguments.str() +
                               ") {\n" + body.str() + "\nreturn\n}\n}";
    return parseSourceString<ModuleOp>(source, context.get());
  }

  [[nodiscard]] bool runPass(ModuleOp moduleOp) const {
    PassManager manager(context.get());
    manager.addPass(createQOracleToQC());
    return succeeded(manager.run(moduleOp));
  }

  [[nodiscard]] static std::string print(ModuleOp moduleOp) {
    std::string printed;
    llvm::raw_string_ostream stream(printed);
    moduleOp.print(stream);
    return printed;
  }

  /// Lower `body` and check that, on every basis state, the circuit maps the
  /// initial register values to `expected(initial)`.
  ///
  /// The registers are the allocations in `body`, in program order. The
  /// simulation also checks that every helper qubit is 0 when it is
  /// deallocated.
  void expectLowersTo(
      StringRef body,
      llvm::function_ref<RegisterValues(ArrayRef<uint64_t>)> expected) {
    auto moduleOp = parseBody(body);
    ASSERT_TRUE(moduleOp) << diagnostics;
    auto function = *moduleOp->getOps<func::FuncOp>().begin();

    SmallVector<Operation*> inputs;
    SmallVector<unsigned> widths;
    for (auto& op : function.getBody().front()) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
        inputs.push_back(&op);
        widths.push_back(alloc.getType().getDimSize(0));
      } else if (isa<qc::AllocOp>(op)) {
        inputs.push_back(&op);
        widths.push_back(1);
      }
    }
    unsigned totalBits = 0;
    for (auto width : widths) {
      totalBits += width;
    }
    ASSERT_LE(totalBits, MAX_INPUT_BITS);

    ASSERT_TRUE(runPass(*moduleOp)) << diagnostics;
    ASSERT_TRUE(succeeded(verify(*moduleOp))) << diagnostics;
    const auto lowered = print(*moduleOp);

    for (uint64_t code = 0; code < (uint64_t{1} << totalBits); ++code) {
      RegisterValues initial;
      RegisterValues want;
      unsigned offset = 0;
      for (auto width : widths) {
        initial.push_back((code >> offset) & ((uint64_t{1} << width) - 1));
        offset += width;
      }
      for (auto [value, width] : llvm::zip_equal(expected(initial), widths)) {
        want.push_back(value & ((uint64_t{1} << width) - 1));
      }

      auto got = BasisSimulator().run(function, inputs, initial);
      ASSERT_TRUE(succeeded(got)) << lowered;
      ASSERT_EQ(*got, want)
          << "for initial values " << ::testing::PrintToString(initial) << ":\n"
          << lowered;
    }
  }

  /// Expect that the pass rejects `body` with `message` and leaves the IR
  /// unchanged.
  void expectRejected(StringRef body, StringRef message,
                      StringRef arguments = "") {
    auto moduleOp = parseBody(body, arguments);
    ASSERT_TRUE(moduleOp) << diagnostics;
    const auto before = print(*moduleOp);
    EXPECT_FALSE(runPass(*moduleOp));
    EXPECT_NE(diagnostics.find(message.str()), std::string::npos)
        << "missing '" << message.str() << "' in:\n"
        << diagnostics;
    EXPECT_EQ(print(*moduleOp), before);
  }
};
} // namespace

TEST_F(QOracleToQCTest, XorsKeptQubitIntoConsumedQubit) {
  expectLowersTo(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : !qc.qubit as %vr) {
      %next = arith.xori %vk, %vr : i1
      qoracle.yield %next : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0]};
                 });
}

TEST_F(QOracleToQCTest, ComputesAndIntoHelpers) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i2
      %next = arith.xori %vr, %ab : i2
      qoracle.yield %next : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ (in[0] & in[1])};
                 });
}

TEST_F(QOracleToQCTest, ComputesNestedTemporaries) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %c : memref<2x!qc.qubit> as %vc)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.xori %va, %vb : i2
      %abc = arith.andi %ab, %vc : i2
      %next = arith.xori %vr, %abc : i2
      qoracle.yield %next : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {
                       in[0],
                       in[1],
                       in[2],
                       in[3] ^ ((in[0] ^ in[1]) & in[2]),
                   };
                 });
}

TEST_F(QOracleToQCTest, XorsConstantIntoConsumedRegister) {
  expectLowersTo(
      R"mlir(
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle consumed(%r : memref<3x!qc.qubit> as %vr) {
      %five = arith.constant 5 : i3
      %next = arith.xori %vr, %five : i3
      qoracle.yield %next : i3
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0] ^ 5U}; });
}

TEST_F(QOracleToQCTest, ZeroExtendsIntoFreshHighBits) {
  expectLowersTo(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %wide = arith.extui %vk : i1 to i3
      %next = arith.xori %vr, %wide : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0]};
                 });
}

TEST_F(QOracleToQCTest, TruncatesToLowBits) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %low = arith.trunci %va : i3 to i2
      %next = arith.xori %vr, %low : i2
      qoracle.yield %next : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ (in[0] & 3U)};
                 });
}

TEST_F(QOracleToQCTest, ChainsUpdatesOfOneRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %k = qc.alloc : !qc.qubit
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %k : !qc.qubit as %vk)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i2
      %r1 = arith.xori %vr, %ab : i2
      %wide = arith.extui %vk : i1 to i2
      %r2 = arith.xori %r1, %wide : i2
      qoracle.yield %r2 : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {
                       in[0],
                       in[1],
                       in[2],
                       in[3] ^ (in[0] & in[1]) ^ in[2],
                   };
                 });
}

TEST_F(QOracleToQCTest, ReadsConsumedRegisterThatIsNotOverwritten) {
  expectLowersTo(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %t = arith.andi %vx, %vk : i2
      %next = arith.xori %vy, %t : i2
      qoracle.yield %vx, %next : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ (in[1] & in[0])};
                 });
}

TEST_F(QOracleToQCTest, ReadsResultOfEarlierUpdate) {
  expectLowersTo(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %x1 = arith.xori %vx, %vk : i2
      %t = arith.andi %x1, %vk : i2
      %y1 = arith.xori %vy, %t : i2
      qoracle.yield %x1, %y1 : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const auto x1 = in[1] ^ in[0];
                   return {in[0], x1, in[2] ^ (x1 & in[0])};
                 });
}

TEST_F(QOracleToQCTest, LowersEveryOracle) {
  expectLowersTo(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %wide = arith.extui %vk : i1 to i2
      %next = arith.xori %vr, %wide : i2
      qoracle.yield %next : i2
    }
    qoracle.oracle consumed(%r : memref<2x!qc.qubit> as %vr) {
      %two = arith.constant 2 : i2
      %next = arith.xori %vr, %two : i2
      qoracle.yield %next : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0] ^ 2U};
                 });
}

TEST_F(QOracleToQCTest, AddsKeptRegisterIntoConsumedRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %next = arith.addi %vr, %va : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, AddsIntoRightOperand) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %next = arith.addi %va, %vr : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, AddsSingleBits) {
  expectLowersTo(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : !qc.qubit as %vr) {
      %next = arith.addi %vr, %vk : i1
      qoracle.yield %next : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, AddsConstant) {
  expectLowersTo(
      R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %eleven = arith.constant 11 : i4
      %next = arith.addi %vr, %eleven : i4
      qoracle.yield %next : i4
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0] + 11U}; });
}

TEST_F(QOracleToQCTest, IncrementsByControlQubit) {
  expectLowersTo(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %step = arith.extui %vk : i1 to i4
      %next = arith.addi %vr, %step : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, AddsTemporaryIntoConsumedRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %next = arith.addi %vr, %ab : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] + (in[0] & in[1])};
                 });
}

TEST_F(QOracleToQCTest, AddsConsumedRegisterThatIsNotOverwritten) {
  expectLowersTo(R"mlir(
    %x = memref.alloc() : memref<3x!qc.qubit>
    %y = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle consumed(%x : memref<3x!qc.qubit> as %vx,
                            %y : memref<3x!qc.qubit> as %vy) {
      %next = arith.addi %vy, %vx : i3
      qoracle.yield %vx, %next : i3, i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, MixesAdditionsAndXors) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %r1 = arith.addi %vr, %va : i3
      %r2 = arith.xori %r1, %vb : i3
      %r3 = arith.addi %r2, %vb : i3
      qoracle.yield %r3 : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], ((in[2] + in[0]) ^ in[1]) + in[1]};
                 });
}

TEST_F(QOracleToQCTest, ComputesSumIntoHelpers) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.addi %va, %vb : i3
      %next = arith.xori %vr, %ab : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ (in[0] + in[1])};
                 });
}

TEST_F(QOracleToQCTest, ComputesNestedSumsIntoHelpers) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %c : memref<2x!qc.qubit> as %vc)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.addi %va, %vb : i2
      %abc = arith.addi %ab, %vc : i2
      %next = arith.xori %vr, %abc : i2
      qoracle.yield %next : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {
                       in[0],
                       in[1],
                       in[2],
                       in[3] ^ (in[0] + in[1] + in[2]),
                   };
                 });
}

TEST_F(QOracleToQCTest, RejectsAdditionReadingTheOverwrittenValue) {
  expectRejected(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %t = arith.andi %vr, %vk : i2
      %next = arith.addi %vr, %t : i2
      qoracle.yield %next : i2
    })mlir",
                 "a temporary depends on the value to be overwritten");
}

TEST_F(QOracleToQCTest, RejectsUnsupportedOperation) {
  expectRejected(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : !qc.qubit as %vr) {
      %next = arith.ori %vr, %vk : i1
      qoracle.yield %next : i1
    })mlir",
                 "contains invalid operations");
}

TEST_F(QOracleToQCTest, RejectsYieldOfTemporary) {
  expectRejected(R"mlir(
    %a = qc.alloc : !qc.qubit
    %b = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : !qc.qubit as %va, %b : !qc.qubit as %vb)
                   consumed(%r : !qc.qubit as %vr) {
      %ab = arith.andi %va, %vb : i1
      qoracle.yield %ab : i1
    })mlir",
                 "yielded value 0 does not match");
}

TEST_F(QOracleToQCTest, RejectsSwappedYield) {
  expectRejected(R"mlir(
    %x = qc.alloc : !qc.qubit
    %y = qc.alloc : !qc.qubit
    qoracle.oracle consumed(%x : !qc.qubit as %vx, %y : !qc.qubit as %vy) {
      qoracle.yield %vy, %vx : i1, i1
    })mlir",
                 "yielded value 0 does not match");
}

TEST_F(QOracleToQCTest, RejectsTemporaryReadingTheOverwrittenValue) {
  expectRejected(R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : !qc.qubit as %vr) {
      %t = arith.andi %vr, %vk : i1
      %next = arith.xori %vr, %t : i1
      qoracle.yield %next : i1
    })mlir",
                 "a temporary depends on the value to be overwritten");
}

TEST_F(QOracleToQCTest, RejectsTemporaryReadingValueOverwrittenEarlier) {
  expectRejected(R"mlir(
    %k = qc.alloc : !qc.qubit
    %x = qc.alloc : !qc.qubit
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%x : !qc.qubit as %vx, %y : !qc.qubit as %vy) {
      %t = arith.andi %vx, %vk : i1
      %x1 = arith.xori %vx, %vk : i1
      %y1 = arith.xori %vy, %t : i1
      qoracle.yield %x1, %y1 : i1, i1
    })mlir",
                 "overwritten in a previous step");
}

TEST_F(QOracleToQCTest, LowersCapturedConstant) {
  expectLowersTo(
      R"mlir(
    %r = memref.alloc() : memref<2x!qc.qubit>
    %three = arith.constant 3 : i2
    qoracle.oracle consumed(%r : memref<2x!qc.qubit> as %vr) {
      %next = arith.addi %vr, %three : i2
      qoracle.yield %next : i2
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0] + 3U}; });
}

TEST_F(QOracleToQCTest, RejectsCapturedFunctionArgument) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle consumed(%r : memref<2x!qc.qubit> as %vr) {
      %next = arith.xori %vr, %x : i2
      qoracle.yield %next : i2
    })mlir",
                 "captures a classical value that is not a constant", "%x: i2");
}

TEST_F(QOracleToQCTest, RejectsCapturedMeasurementResult) {
  expectRejected(R"mlir(
    %q = qc.alloc : !qc.qubit
    %r = qc.alloc : !qc.qubit
    %m = qc.measure %q : !qc.qubit -> i1
    qoracle.oracle consumed(%r : !qc.qubit as %vr) {
      %next = arith.xori %vr, %m : i1
      qoracle.yield %next : i1
    })mlir",
                 "captures a classical value that is not a constant");
}
