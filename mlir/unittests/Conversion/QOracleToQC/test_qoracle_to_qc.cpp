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
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QC/IR/QCOps.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  /// Run the body of `function`, repeating the body of each `scf.for`.
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
  DenseMap<Operation*, uint64_t> initialValues_;
  DenseMap<Operation*, SmallVector<size_t>> inputBits_;

  [[nodiscard]] SmallVector<size_t> allocate(size_t width, uint64_t value);
  [[nodiscard]] FailureOr<size_t> bitOf(Value qubit);
  [[nodiscard]] LogicalResult apply(Operation& op);
  [[nodiscard]] LogicalResult runBlock(Block& block);
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

LogicalResult BasisSimulator::runBlock(Block& block) {
  for (auto& op : block) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op)) {
      auto allocated =
          allocate(alloc.getType().getDimSize(0), initialValues_.lookup(&op));
      registers_[alloc.getResult()] = allocated;
      inputBits_[&op] = allocated;
    } else if (auto alloc = dyn_cast<qc::AllocOp>(op)) {
      auto allocated = allocate(1, initialValues_.lookup(&op));
      qubits_[alloc.getResult()] = allocated.front();
      inputBits_[&op] = allocated;
    } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (loop.getNumRegionIterArgs() != 0) {
        ADD_FAILURE() << "loop carries values";
        return failure();
      }
      const auto upper = indices_.at(loop.getUpperBound());
      const auto step = indices_.at(loop.getStep());
      for (auto i = indices_.at(loop.getLowerBound()); i < upper; i += step) {
        if (failed(runBlock(*loop.getBody()))) {
          return failure();
        }
      }
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
    } else if (isa<func::ReturnOp, scf::YieldOp>(op)) {
      break;
    } else if (failed(apply(op))) {
      return failure();
    }
  }
  return success();
}

FailureOr<RegisterValues> BasisSimulator::run(func::FuncOp function,
                                              ArrayRef<Operation*> inputs,
                                              ArrayRef<uint64_t> initial) {
  for (auto [input, value] : llvm::zip_equal(inputs, initial)) {
    initialValues_[input] = value;
  }
  if (failed(runBlock(function.getBody().front()))) {
    return failure();
  }

  RegisterValues final;
  for (auto* input : inputs) {
    uint64_t value = 0;
    for (auto [i, bit] : llvm::enumerate(inputBits_.at(input))) {
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
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, mqt::MQTDialect, qc::QCDialect,
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

  /// Parse a function body wrapped in the entry point `@main`, which takes
  /// `arguments`.
  [[nodiscard]] OwningOpRef<ModuleOp> parseBody(StringRef body,
                                                StringRef arguments = "") {
    diagnostics.clear();
    const std::string source = "module {\nfunc.func @main(" + arguments.str() +
                               ") attributes {mqt.entry_point} {\n" +
                               body.str() + "\nreturn\n}\n}";
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
  /// check also requires the output to follow the MQT allocation rules and
  /// every helper qubit to be 0 when it is deallocated.
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
    ASSERT_TRUE(succeeded(mqt::verifyQuantumAllocations(*moduleOp)))
        << diagnostics << lowered;

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

  /// Lower `body` and return the number of helper qubits the circuit
  /// allocates.
  [[nodiscard]] size_t countHelpers(StringRef body) {
    auto moduleOp = parseBody(body);
    if (!moduleOp) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    const auto before = countAllocations(*moduleOp);
    if (!runPass(*moduleOp)) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    return countAllocations(*moduleOp) - before;
  }

  [[nodiscard]] static size_t countAllocations(ModuleOp moduleOp) {
    size_t allocations = 0;
    moduleOp.walk([&](qc::AllocOp) { ++allocations; });
    return allocations;
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

// R2 computes every carry above the first into a qubit of the addition, so an
// update with `+` needs one helper, as in Cuccaro's adder.
TEST_F(QOracleToQCTest, AddsWithOneHelper) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %next = arith.addi %vr, %va : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] + in[0]};
  });
  EXPECT_EQ(countHelpers(body), 1U);
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

// A kept register that is not read after a step may hold the step's result.
// R6 restores it at the end of the oracle, so `a ^ b` needs no helpers.
TEST_F(QOracleToQCTest, BorrowsKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.xori %va, %vb : i3
      %next = arith.addi %vr, %ab : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] + (in[0] ^ in[1])};
  });
  // The addition needs its carry helper.
  EXPECT_EQ(countHelpers(body), 1U);
}

// Restoring a register that holds a sum needs a subtraction.
TEST_F(QOracleToQCTest, BorrowsKeptRegisterForAddition) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.addi %va, %vb : i3
      %next = arith.xori %vr, %ab : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] + in[1])};
  });
  // The addition and its inverse each need one carry helper.
  EXPECT_EQ(countHelpers(body), 2U);
}

// The second step reads the register that the first step borrows, so R6 must
// restore `b` before `a`.
TEST_F(QOracleToQCTest, RestoresBorrowedRegistersInReverseOrder) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.xori %va, %vb : i2
      %bab = arith.xori %vb, %ab : i2
      %next = arith.xori %vr, %bab : i2
      qoracle.yield %next : i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ in[0]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// Restoring `a` at the end would read `r ^ b` instead of `r`, so the pass
// must not borrow `a`.
TEST_F(QOracleToQCTest, KeepsRegisterWhenRestoreWouldReadChangedValue) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    %s = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr,
                            %s : memref<2x!qc.qubit> as %vs) {
      %ar = arith.xori %va, %vr : i2
      %nextS = arith.xori %vs, %ar : i2
      %nextR = arith.xori %vr, %vb : i2
      qoracle.yield %nextR, %nextS : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ in[1], in[3] ^ in[0] ^ in[2]};
                 });
}

// Borrowing `a` would overwrite it before the temporary `a & b` is
// recomputed, so the pass lowers without borrowing and reports nothing.
TEST_F(QOracleToQCTest, KeepsRegisterWhenTemporaryReadsItLater) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %c : memref<2x!qc.qubit> as %vc)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i2
      %ac = arith.xori %va, %vc : i2
      %r1 = arith.xori %vr, %ab : i2
      %r2 = arith.xori %r1, %ac : i2
      qoracle.yield %r2 : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2],
                           in[3] ^ (in[0] & in[1]) ^ (in[0] ^ in[2])};
                 });
  EXPECT_EQ(diagnostics, "");
}

// Inverting a loop is not supported, so a loop cannot borrow its initial
// value. The rejection reports only the attempt without borrowing.
TEST_F(QOracleToQCTest, RejectsLoopOverKeptRegister) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %va) -> (i2) {
        %next = arith.xori %acc, %vr : i2
        scf.yield %next : i2
      }
      %result = arith.xori %vr, %out : i2
      qoracle.yield %result : i2
    })mlir",
                 "a loop initial value is not stored in consumed qubits");
  EXPECT_EQ(StringRef(diagnostics).count("cannot lower qoracle"), 1U)
      << diagnostics;
}

TEST_F(QOracleToQCTest, RepeatsLoopBody) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %out = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %vr) -> (i3) {
        %next = arith.addi %acc, %va : i3
        scf.yield %next : i3
      }
      qoracle.yield %out : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + (3 * in[0])};
                 });
}

TEST_F(QOracleToQCTest, RecomputesTemporariesInEachIteration) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %out:2 = scf.for %i = %c0 to %c3 step %c1
          iter_args(%ix = %vx, %iy = %vy) -> (i2, i2) {
        %t = arith.andi %ix, %va : i2
        %y1 = arith.addi %iy, %t : i2
        %x1 = arith.xori %ix, %vb : i2
        scf.yield %x1, %y1 : i2, i2
      }
      qoracle.yield %out#0, %out#1 : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   auto x = in[2];
                   auto y = in[3];
                   for (int i = 0; i < 3; ++i) {
                     y += x & in[0];
                     x ^= in[1];
                   }
                   return {in[0], in[1], x, y};
                 });
}

TEST_F(QOracleToQCTest, LowersLoopWithOffsetAndStep) {
  expectLowersTo(
      R"mlir(
    %r = memref.alloc() : memref<3x!qc.qubit>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    qoracle.oracle consumed(%r : memref<3x!qc.qubit> as %vr) {
      %one = arith.constant 1 : i3
      %out = scf.for %i = %c1 to %c6 step %c2
          iter_args(%acc = %vr) -> (i3) {
        %next = arith.addi %acc, %one : i3
        scf.yield %next : i3
      }
      qoracle.yield %out : i3
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0] + 3U}; });
}

TEST_F(QOracleToQCTest, LowersLoopWithoutIterations) {
  expectLowersTo(
      R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %out = scf.for %i = %c2 to %c2 step %c1
          iter_args(%acc = %vr) -> (i2) {
        %next = arith.addi %acc, %va : i2
        scf.yield %next : i2
      }
      qoracle.yield %out : i2
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0], in[1]}; });
}

TEST_F(QOracleToQCTest, LowersNestedLoops) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%outer = %vr) -> (i3) {
        %inner = scf.for %j = %c0 to %c3 step %c1
            iter_args(%acc = %outer) -> (i3) {
          %ab = arith.andi %va, %vb : i3
          %next = arith.addi %acc, %ab : i3
          scf.yield %next : i3
        }
        scf.yield %inner : i3
      }
      qoracle.yield %out : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] + (6 * (in[0] & in[1]))};
                 });
}

TEST_F(QOracleToQCTest, UpdatesBeforeAndAfterLoop) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %r1 = arith.xori %vr, %va : i3
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %r1) -> (i3) {
        %next = arith.addi %acc, %vb : i3
        scf.yield %next : i3
      }
      %r2 = arith.xori %out, %va : i3
      qoracle.yield %r2 : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {
                       in[0],
                       in[1],
                       ((in[2] ^ in[0]) + (2 * in[1])) ^ in[0],
                   };
                 });
}

TEST_F(QOracleToQCTest, UpdatesValueReadByEarlierLoop) {
  expectLowersTo(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %out = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %vy) -> (i2) {
        %next = arith.addi %acc, %vx : i2
        scf.yield %next : i2
      }
      %x1 = arith.xori %vx, %vk : i2
      qoracle.yield %x1, %out : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0], in[2] + (3 * in[1])};
                 });
}

TEST_F(QOracleToQCTest, RejectsLoopUsingItsInductionVariable) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %one = arith.constant 1 : i2
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%outer = %vr) -> (i2) {
        %inner = scf.for %j = %c0 to %i step %c1
            iter_args(%acc = %outer) -> (i2) {
          %next = arith.addi %acc, %one : i2
          scf.yield %next : i2
        }
        scf.yield %inner : i2
      }
      qoracle.yield %out : i2
    })mlir",
                 "uses its induction variable");
}

TEST_F(QOracleToQCTest, RejectsLoopReadingTemporaryFromOutside) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %ab = arith.andi %va, %vb : i2
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %vr) -> (i2) {
        %next = arith.xori %acc, %ab : i2
        scf.yield %next : i2
      }
      qoracle.yield %out : i2
    })mlir",
                 "neither stored in qubits nor a constant");
}

TEST_F(QOracleToQCTest, RejectsLoopReadingValueItOverwrites) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %vr) -> (i2) {
        %next = arith.addi %acc, %vr : i2
        scf.yield %next : i2
      }
      qoracle.yield %out : i2
    })mlir",
                 "reads a value that it overwrites");
}

TEST_F(QOracleToQCTest, RejectsInitialValueReadAfterLoop) {
  expectRejected(R"mlir(
    %k = qc.alloc : !qc.qubit
    %x = qc.alloc : !qc.qubit
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%x : !qc.qubit as %vx, %y : !qc.qubit as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %vx) -> (i1) {
        %next = arith.xori %acc, %vk : i1
        scf.yield %next : i1
      }
      %y1 = arith.xori %vy, %vx : i1
      qoracle.yield %out, %y1 : i1, i1
    })mlir",
                 "initial value is read after the loop");
}

TEST_F(QOracleToQCTest, RejectsLoopSwappingIterationArguments) {
  expectRejected(R"mlir(
    %x = qc.alloc : !qc.qubit
    %y = qc.alloc : !qc.qubit
    qoracle.oracle consumed(%x : !qc.qubit as %vx, %y : !qc.qubit as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %out:2 = scf.for %i = %c0 to %c2 step %c1
          iter_args(%p = %vx, %q = %vy) -> (i1, i1) {
        scf.yield %q, %p : i1, i1
      }
      qoracle.yield %out#0, %out#1 : i1, i1
    })mlir",
                 "loop yields value 0");
}

TEST_F(QOracleToQCTest, LeavesLoopsOutsideOraclesUnchanged) {
  expectLowersTo(
      R"mlir(
    %q = qc.alloc : !qc.qubit
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %i = %c0 to %c3 step %c1 {
      qc.x %q : !qc.qubit
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0] ^ 1U}; });
}

TEST_F(QOracleToQCTest, NeverOverwritesOuterValueInsideLoop) {
  expectLowersTo(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %out = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %vy) -> (i2) {
        %next = arith.addi %vx, %acc : i2
        scf.yield %next : i2
      }
      %x1 = arith.xori %vx, %vk : i2
      qoracle.yield %x1, %out : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0], in[2] + (3 * in[1])};
                 });
}

TEST_F(QOracleToQCTest, RejectsProgramWithStaticQubits) {
  expectRejected(R"mlir(
    %k = qc.static 0 : !qc.qubit
    %r = qc.static 1 : !qc.qubit
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(%r : !qc.qubit as %vr) {
      %next = arith.xori %vr, %vk : i1
      qoracle.yield %next : i1
    })mlir",
                 "in a program with static qubits");
}
