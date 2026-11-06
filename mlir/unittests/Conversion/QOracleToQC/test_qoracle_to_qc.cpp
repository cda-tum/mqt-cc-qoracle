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
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"

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

#include <array>
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

/// Rotate the `width`-bit `value` left by `amount`.
uint64_t rotl(uint64_t value, unsigned amount, unsigned width) {
  const auto mask = (uint64_t{1} << width) - 1;
  value &= mask;
  return ((value << amount) | (value >> (width - amount))) & mask;
}

/// Simulation of a circuit of X and SWAP gates with controls on a
/// computational basis state, where each qubit holds a single bit.
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
  if (auto swap = dyn_cast<qc::SWAPOp>(op)) {
    auto first = bitOf(swap.getQubit0In());
    auto second = bitOf(swap.getQubit1In());
    if (failed(first) || failed(second)) {
      return failure();
    }
    if (*first == *second) {
      ADD_FAILURE() << "SWAP of a qubit with itself";
      return failure();
    }
    const bool value = bits_[*first];
    bits_[*first] = bits_[*second];
    bits_[*second] = value;
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
  /// The registers are the allocations in `body`, in program order. Registers
  /// passed to an oracle as `zero` operands start at 0 only. The check also
  /// requires the output to follow the MQT allocation rules and every helper
  /// qubit to be 0 when it is deallocated.
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

    SmallVector<bool> startsAtZero(inputs.size(), false);
    for (auto oracle : function.getOps<qoracle::OracleOp>()) {
      for (auto [qubits, alias] : llvm::zip_equal(
               oracle.getConsumedQubits(), oracle.getConsumedAliases())) {
        const auto* input = llvm::find(inputs, qubits.getDefiningOp());
        ASSERT_NE(input, inputs.end());
        if (oracle.isZeroAlias(alias)) {
          startsAtZero[input - inputs.begin()] = true;
        }
      }
    }

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
      if (llvm::any_of(llvm::zip_equal(initial, startsAtZero), [](auto entry) {
            auto [value, zero] = entry;
            return zero && value != 0;
          })) {
        continue;
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

  /// Lower `body` and return the number of Toffoli gates in the circuit, with
  /// each loop body counted once.
  [[nodiscard]] size_t countToffolis(StringRef body) {
    auto moduleOp = parseBody(body);
    if (!moduleOp || !runPass(*moduleOp)) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    size_t toffolis = 0;
    moduleOp->walk([&](qc::CtrlOp ctrl) {
      if (ctrl.getNumControls() == 2) {
        ++toffolis;
      }
    });
    return toffolis;
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

// The temporary comes first, so the register to overwrite is the second
// operand.
TEST_F(QOracleToQCTest, XorsTemporaryIntoRightOperand) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i2
      %next = arith.xori %ab, %vr : i2
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

// The constant's bits are known, so no qubits hold them and each carry is a
// function of two qubits. Below the lowest 1-bit at index `low`, every carry is
// 0; the carry into bit `low + 1` is a copy. Each later carry into bits up to
// w - 1 costs one Toffoli to compute and one to clean up, so a w-bit addition
// takes 2 * max(0, w - 2 - low) Toffolis and at most w - 1 - low helpers.
TEST_F(QOracleToQCTest, AddsConstantWithoutConstantQubits) {
  struct ConstantCase {
    uint64_t value;
    size_t toffolis;
    size_t maxHelpers;
  };
  const std::array<ConstantCase, 6> cases = {{
      {.value = 1, .toffolis = 8, .maxHelpers = 5},
      {.value = 4, .toffolis = 4, .maxHelpers = 3},
      {.value = 45, .toffolis = 8, .maxHelpers = 5}, // 0b101101
      {.value = 63, .toffolis = 8, .maxHelpers = 5}, // all ones: r - 1
      {.value = 48, .toffolis = 0, .maxHelpers = 1}, // 0b110000
      {.value = 32, .toffolis = 0, .maxHelpers = 0}, // only the top bit
  }};
  for (const auto& testCase : cases) {
    SCOPED_TRACE(testCase.value);
    const auto body = R"mlir(
    %r = memref.alloc() : memref<6x!qc.qubit>
    qoracle.oracle consumed(%r : memref<6x!qc.qubit> as %vr) {
      %c = arith.constant )mlir" +
                      std::to_string(testCase.value) + R"mlir( : i6
      %next = arith.addi %vr, %c : i6
      qoracle.yield %next : i6
    })mlir";
    expectLowersTo(body, [&](ArrayRef<uint64_t> in) -> RegisterValues {
      return {in[0] + testCase.value};
    });
    EXPECT_EQ(countToffolis(body), testCase.toffolis);
    EXPECT_LE(countHelpers(body), testCase.maxHelpers);
  }
}

TEST_F(QOracleToQCTest, AddsConstantFromLeft) {
  constexpr StringRef body = R"mlir(
    %r = memref.alloc() : memref<6x!qc.qubit>
    qoracle.oracle consumed(%r : memref<6x!qc.qubit> as %vr) {
      %c = arith.constant 4 : i6
      %next = arith.addi %c, %vr : i6
      qoracle.yield %next : i6
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0] + 4U};
  });
  EXPECT_EQ(countToffolis(body), 4U);
}

TEST_F(QOracleToQCTest, AddsZeroWithoutGates) {
  constexpr StringRef body = R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %zero = arith.constant 0 : i4
      %next = arith.addi %vr, %zero : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(
      body, [](ArrayRef<uint64_t> in) -> RegisterValues { return {in[0]}; });
  auto moduleOp = parseBody(body);
  ASSERT_TRUE(moduleOp) << diagnostics;
  ASSERT_TRUE(runPass(*moduleOp)) << diagnostics;
  size_t gates = 0;
  moduleOp->walk([&](Operation* op) {
    if (isa<qc::XOp, qc::CtrlOp, qc::SWAPOp, qc::AllocOp>(op)) {
      ++gates;
    }
  });
  EXPECT_EQ(gates, 0U) << print(*moduleOp);
}

// ChaCha's key setting starts each of words 0 to 3 as `constant + key`.
TEST_F(QOracleToQCTest, CreatesSumOfConstantAndKeptRegister) {
  constexpr StringRef body = R"mlir(
    %k = memref.alloc() : memref<5x!qc.qubit>
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle kept(%k : memref<5x!qc.qubit> as %vk)
                   consumed(zero %r : memref<5x!qc.qubit> as %vr) {
      %c = arith.constant 6 : i5
      %s = arith.addi %c, %vk : i5
      qoracle.yield %s : i5
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[0] + 6U};
  });
  EXPECT_EQ(countToffolis(body), 4U);
  EXPECT_LE(countHelpers(body), 3U);
}

// One constant read by two additions, as in ChaCha's feed-forward.
TEST_F(QOracleToQCTest, AddsSharedConstantIntoTwoRegisters) {
  constexpr StringRef body = R"mlir(
    %x = memref.alloc() : memref<4x!qc.qubit>
    %y = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%x : memref<4x!qc.qubit> as %vx,
                            %y : memref<4x!qc.qubit> as %vy) {
      %c = arith.constant 5 : i4
      %nx = arith.addi %vx, %c : i4
      %ny = arith.addi %vy, %c : i4
      qoracle.yield %nx, %ny : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0] + 5U, in[1] + 5U};
  });
  EXPECT_EQ(countToffolis(body), 8U);
}

TEST_F(QOracleToQCTest, XorsSumWithConstantIntoRegister) {
  expectLowersTo(R"mlir(
    %k = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%k : memref<4x!qc.qubit> as %vk)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c = arith.constant 3 : i4
      %s = arith.addi %vk, %c : i4
      %next = arith.xori %vr, %s : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ (in[0] + 3U)};
                 });
}

TEST_F(QOracleToQCTest, AddsConstantInsideLoop) {
  expectLowersTo(R"mlir(
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle consumed(%r : memref<5x!qc.qubit> as %vr) {
      %lb = arith.constant 0 : index
      %ub = arith.constant 3 : index
      %step = arith.constant 1 : index
      %c = arith.constant 7 : i5
      %out = scf.for %i = %lb to %ub step %step iter_args(%acc = %vr) -> (i5) {
        %next = arith.addi %acc, %c : i5
        scf.yield %next : i5
      }
      qoracle.yield %out : i5
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0] + (3U * 7U)};
                 });
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

// A rotation by a constant moves no data; the consumed register is sorted
// into bit order at the end of the oracle.
TEST_F(QOracleToQCTest, RotatesConsumedRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %t = arith.xori %vd, %va : i4
      %hi = arith.shli %t, %c1 : i4
      %lo = arith.shrui %t, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      qoracle.yield %rot : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], rotl(in[1] ^ in[0], 1, 4)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, RecognizesRotationWithSwappedOperands) {
  expectLowersTo(
      R"mlir(
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vd, %c3 : i4
      %lo = arith.shrui %vd, %c1 : i4
      %rot = arith.ori %lo, %hi : i4
      qoracle.yield %rot : i4
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues {
        return {rotl(in[0], 3, 4)};
      });
}

// A rotation by half the width splits into two cycles of qubits, as ChaCha's
// rotation by 16 does on 32 bits.
TEST_F(QOracleToQCTest, RotatesByHalfTheWidth) {
  expectLowersTo(
      R"mlir(
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c2 = arith.constant 2 : i4
      %hi = arith.shli %vd, %c2 : i4
      %lo = arith.shrui %vd, %c2 : i4
      %rot = arith.ori %hi, %lo : i4
      qoracle.yield %rot : i4
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues {
        return {rotl(in[0], 2, 4)};
      });
}

TEST_F(QOracleToQCTest, ComposesRotations) {
  expectLowersTo(
      R"mlir(
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %c3 = arith.constant 3 : i4
      %hi1 = arith.shli %vd, %c1 : i4
      %lo1 = arith.shrui %vd, %c3 : i4
      %rot1 = arith.ori %hi1, %lo1 : i4
      %hi2 = arith.shli %rot1, %c2 : i4
      %lo2 = arith.shrui %rot1, %c2 : i4
      %rot2 = arith.ori %hi2, %lo2 : i4
      qoracle.yield %rot2 : i4
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues {
        return {rotl(in[0], 3, 4)};
      });
}

// Addition is sensitive to bit order, so it must read the rotated register
// in rotated order.
TEST_F(QOracleToQCTest, AddsIntoRotatedRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vd, %c1 : i4
      %lo = arith.shrui %vd, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %next = arith.addi %rot, %va : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], rotl(in[1], 1, 4) + in[0]};
                 });
}

// The shape of a ChaCha quarter-round step: the rotated register is the
// operand of an addition into another register.
TEST_F(QOracleToQCTest, AddsRotatedRegisterIntoOtherRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %c = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%c : memref<4x!qc.qubit> as %vc,
                            %d : memref<4x!qc.qubit> as %vd) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %t = arith.xori %vd, %va : i4
      %hi = arith.shli %t, %c1 : i4
      %lo = arith.shrui %t, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %nextC = arith.addi %vc, %rot : i4
      qoracle.yield %nextC, %rot : i4, i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const auto rotated = rotl(in[2] ^ in[0], 1, 4);
                   return {in[0], in[1] + rotated, rotated};
                 });
}

// The rotation is not the last reader of `d`, so it is a temporary that
// shares the qubits of `d`.
TEST_F(QOracleToQCTest, ReadsRegisterThroughRotationBeforeUpdatingIt) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd,
                            %r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vd, %c1 : i4
      %lo = arith.shrui %vd, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %nextR = arith.xori %vr, %rot : i4
      %nextD = arith.xori %vd, %va : i4
      qoracle.yield %nextD, %nextR : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ in[0], in[2] ^ rotl(in[1], 1, 4)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, ReadsRotatedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %next = arith.xori %vr, %rot : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ rotl(in[0], 1, 4)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, RotatesTemporary) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %ab = arith.andi %va, %vb : i4
      %hi = arith.shli %ab, %c1 : i4
      %lo = arith.shrui %ab, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %next = arith.xori %vr, %rot : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ rotl(in[0] & in[1], 1, 4)};
                 });
}

// The borrowed register is updated in rotated order; R6 restores it in place.
TEST_F(QOracleToQCTest, BorrowsRotatedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %sum = arith.addi %rot, %vb : i4
      %next = arith.xori %vr, %sum : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (rotl(in[0], 1, 4) + in[1])};
  });
  // The addition and its inverse each need one carry helper.
  EXPECT_EQ(countHelpers(body), 2U);
}

// The iteration argument must be in bit order again at the end of each
// iteration.
TEST_F(QOracleToQCTest, RotatesInsideLoop) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %three = arith.constant 3 : index
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %out = scf.for %i = %c0 to %three step %one
          iter_args(%acc = %vd) -> (i4) {
        %t = arith.xori %acc, %va : i4
        %hi = arith.shli %t, %c1 : i4
        %lo = arith.shrui %t, %c3 : i4
        %rot = arith.ori %hi, %lo : i4
        scf.yield %rot : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   auto value = in[1];
                   for (int i = 0; i < 3; ++i) {
                     value = rotl(value ^ in[0], 1, 4);
                   }
                   return {in[0], value};
                 });
}

TEST_F(QOracleToQCTest, RejectsShiftOutsideRotation) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %next = arith.shli %vr, %c1 : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

TEST_F(QOracleToQCTest, RejectsOrOutsideRotation) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %next = arith.ori %vr, %va : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

TEST_F(QOracleToQCTest, RejectsShiftAmountsThatDoNotFormRotation) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %hi = arith.shli %vr, %c1 : i4
      %lo = arith.shrui %vr, %c2 : i4
      %next = arith.ori %hi, %lo : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

// `x >> 4` is poison for an i4, so this is not a rotation by 0.
TEST_F(QOracleToQCTest, RejectsShiftByBitWidth) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : i4
      %c4 = arith.constant 4 : i4
      %hi = arith.shli %vr, %c0 : i4
      %lo = arith.shrui %vr, %c4 : i4
      %next = arith.ori %hi, %lo : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

TEST_F(QOracleToQCTest, RejectsShiftsOfDifferentValues) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vr, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %next = arith.ori %hi, %lo : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

// `r ^= rotl(r, 1)` cannot be computed in place: the rotation reads the
// register that the update overwrites.
TEST_F(QOracleToQCTest, RejectsXorWithOwnRotation) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vr, %c1 : i4
      %lo = arith.shrui %vr, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %next = arith.xori %vr, %rot : i4
      qoracle.yield %next : i4
    })mlir",
                 "a temporary depends on the value to be overwritten");
}

// The rotation of `d` is read after `d` is overwritten.
TEST_F(QOracleToQCTest, RejectsRotationOfValueOverwrittenEarlier) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd,
                            %r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vd, %c1 : i4
      %lo = arith.shrui %vd, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %nextD = arith.xori %vd, %va : i4
      %nextR = arith.xori %vr, %rot : i4
      qoracle.yield %nextD, %nextR : i4, i4
    })mlir",
                 "overwritten in a previous step");
}

// A shift with a second user computes more than the rotation.
TEST_F(QOracleToQCTest, RejectsShiftReadOutsideRotation) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %r1 = arith.xori %vr, %rot : i4
      %r2 = arith.xori %r1, %hi : i4
      qoracle.yield %r2 : i4
    })mlir",
                 "contains invalid operations");
}

// A right shift by a constant is a relabeling: bit i of `a >> 1` is bit i + 1
// of `a`, and the top bit is 0, so XOR reads it without gates or helpers.
TEST_F(QOracleToQCTest, XorsShiftedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %sh = arith.shrui %va, %c1 : i4
      %next = arith.xori %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ (in[0] >> 1)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_EQ(countToffolis(body), 0U);
}

// Unlike a shift inside a rotation, a plain shift may have several readers.
TEST_F(QOracleToQCTest, XorsOneShiftIntoTwoRegisters) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    %s = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr,
                            %s : memref<4x!qc.qubit> as %vs) {
      %c2 = arith.constant 2 : i4
      %sh = arith.shrui %va, %c2 : i4
      %nextR = arith.xori %vr, %sh : i4
      %nextS = arith.xori %vs, %sh : i4
      qoracle.yield %nextR, %nextS : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ (in[0] >> 2), in[2] ^ (in[0] >> 2)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, ShiftsTemporary) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %ab = arith.andi %va, %vb : i4
      %sh = arith.shrui %ab, %c1 : i4
      %next = arith.xori %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ ((in[0] & in[1]) >> 1)};
                 });
}

// Two relabelings compose: the shift reads the rotated bit order.
TEST_F(QOracleToQCTest, ShiftsRotatedRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %sh = arith.shrui %rot, %c2 : i4
      %next = arith.xori %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ (rotl(in[0], 1, 4) >> 2)};
                 });
}

// The pattern of SHA-256's message schedule: r += rotl(a, 1) ^ (a >> 2).
TEST_F(QOracleToQCTest, AddsXorOfRotationAndShift) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %sh = arith.shrui %va, %c2 : i4
      %sigma = arith.xori %rot, %sh : i4
      %next = arith.addi %vr, %sigma : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + (rotl(in[0], 1, 4) ^ (in[0] >> 2))};
                 });
}

// The zero top bits of the shift make the addition no more expensive than
// adding the whole register.
TEST_F(QOracleToQCTest, AddsShiftedKeptRegister) {
  constexpr StringRef shifted = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %sh = arith.shrui %va, %c1 : i4
      %next = arith.addi %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir";
  constexpr StringRef whole = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %next = arith.addi %vr, %va : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(shifted, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] + (in[0] >> 1)};
  });
  EXPECT_LE(countToffolis(shifted), countToffolis(whole));
}

TEST_F(QOracleToQCTest, ShiftsInsideLoop) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %three = arith.constant 3 : index
      %c1 = arith.constant 1 : i4
      %out = scf.for %i = %c0 to %three step %one
          iter_args(%acc = %vd) -> (i4) {
        %sh = arith.shrui %va, %c1 : i4
        %next = arith.addi %acc, %sh : i4
        scf.yield %next : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] + (3 * (in[0] >> 1))};
                 });
}

// The copy writes the shifted bits; the top bit of `t` stays 0.
TEST_F(QOracleToQCTest, CopiesShiftedValueIntoZeroRegister) {
  constexpr StringRef body = R"mlir(
    %x = memref.alloc() : memref<4x!qc.qubit>
    %t = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%x : memref<4x!qc.qubit> as %vx,
                            zero %t : memref<4x!qc.qubit> as %vt) {
      %c1 = arith.constant 1 : i4
      %sh = arith.shrui %vx, %c1 : i4
      qoracle.yield %vx, %sh : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[0] >> 1};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// `r >> 1` drops bit 0 of `r`, so it cannot be the new value of `r`.
TEST_F(QOracleToQCTest, RejectsYieldOfShiftedRegister) {
  expectRejected(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %sh = arith.shrui %vr, %c1 : i4
      qoracle.yield %sh : i4
    })mlir",
                 "yielded value 0 does not match");
}

// `x >> 4` is poison for an i4.
TEST_F(QOracleToQCTest, RejectsShiftByAtLeastBitWidth) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c4 = arith.constant 4 : i4
      %sh = arith.shrui %va, %c4 : i4
      %next = arith.xori %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

TEST_F(QOracleToQCTest, RejectsShiftByQuantumAmount) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %n = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %n : memref<4x!qc.qubit> as %vn)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %sh = arith.shrui %va, %vn : i4
      %next = arith.xori %vr, %sh : i4
      qoracle.yield %next : i4
    })mlir",
                 "contains invalid operations");
}

// A `zero` register starts at 0, so a value whose operands must all be kept is
// computed into it: a copy of `a`, then an in-place addition of `b`.
TEST_F(QOracleToQCTest, CreatesSumInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %s = arith.addi %va, %vb : i3
      qoracle.yield %s : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] + in[1]};
  });
  EXPECT_EQ(countHelpers(body), 1U);
}

TEST_F(QOracleToQCTest, CreatesXorInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %s = arith.xori %va, %vb : i3
      qoracle.yield %s : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] ^ in[1]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, CreatesAndInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %s = arith.andi %va, %vb : i3
      qoracle.yield %s : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] & in[1]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, CreatesConstantInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %five = arith.constant 5 : i3
      qoracle.yield %five : i3
    })mlir";
  expectLowersTo(body,
                 [](ArrayRef<uint64_t>) -> RegisterValues { return {5U}; });
  EXPECT_EQ(countHelpers(body), 0U);
}

TEST_F(QOracleToQCTest, CopiesKeptRegisterIntoZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      qoracle.yield %va : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[0]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// Inside the region, the alias of a `zero` register is 0, so an update of it
// is an ordinary in-place update.
TEST_F(QOracleToQCTest, UpdatesZeroAlias) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %next = arith.xori %vr, %ab : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[0] & in[1]};
                 });
}

TEST_F(QOracleToQCTest, UpdatesCreatedValueInPlace) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %s = arith.addi %va, %vb : i3
      %t = arith.xori %s, %vc : i3
      %u = arith.addi %t, %va : i3
      qoracle.yield %u : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2],
                           ((in[0] + in[1]) ^ in[2]) + in[0]};
                 });
}

// Once created, `%s` lives in `r`, so the later update of `x` reads it from
// there instead of recomputing it.
TEST_F(QOracleToQCTest, ReadsCreatedValueInLaterUpdate) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            zero %r : memref<2x!qc.qubit> as %vr) {
      %s = arith.andi %va, %vb : i2
      %next = arith.xori %vx, %s : i2
      qoracle.yield %next, %s : i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] & in[1]), in[0] & in[1]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// `%s` reads `x` before `%next` overwrites it, so `%s` must be created first.
TEST_F(QOracleToQCTest, CreatesValueBeforeItsOperandIsOverwritten) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %x = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%x : memref<3x!qc.qubit> as %vx,
                            zero %r : memref<3x!qc.qubit> as %vr) {
      %s = arith.addi %vx, %va : i3
      %next = arith.xori %vx, %va : i3
      qoracle.yield %next, %s : i3, i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ in[0], in[1] + in[0]};
                 });
}

TEST_F(QOracleToQCTest, CreatesValuesInSeveralZeroRegisters) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    %t = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(zero %r : memref<2x!qc.qubit> as %vr,
                            zero %t : memref<2x!qc.qubit> as %vt) {
      %sum = arith.addi %va, %vb : i2
      %xor = arith.xori %va, %vb : i2
      qoracle.yield %sum, %xor : i2, i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[0] + in[1], in[0] ^ in[1]};
                 });
}

// `%s` is created in `r`; the second `zero` register receives a copy.
TEST_F(QOracleToQCTest, CopiesValueIntoSecondZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    %t = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(zero %r : memref<2x!qc.qubit> as %vr,
                            zero %t : memref<2x!qc.qubit> as %vt) {
      %s = arith.addi %va, %vb : i2
      qoracle.yield %s, %s : i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] + in[1], in[0] + in[1]};
  });
  EXPECT_EQ(countHelpers(body), 1U);
}

// The value of `x` stays in `x`; the `zero` register receives a copy.
TEST_F(QOracleToQCTest, CopiesUpdatedRegisterIntoZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %x = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va)
                   consumed(%x : memref<3x!qc.qubit> as %vx,
                            zero %r : memref<3x!qc.qubit> as %vr) {
      %next = arith.addi %vx, %va : i3
      qoracle.yield %next, %next : i3, i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] + in[0], in[1] + in[0]};
  });
  EXPECT_EQ(countHelpers(body), 1U);
}

// The copy follows the relabeling, so `t` holds the rotated bits in order.
TEST_F(QOracleToQCTest, CopiesRotatedValueIntoZeroRegister) {
  constexpr StringRef body = R"mlir(
    %x = memref.alloc() : memref<4x!qc.qubit>
    %t = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%x : memref<4x!qc.qubit> as %vx,
                            zero %t : memref<4x!qc.qubit> as %vt) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %hi = arith.shli %vx, %c1 : i4
      %lo = arith.shrui %vx, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      qoracle.yield %rot, %rot : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {rotl(in[0], 1, 4), rotl(in[0], 1, 4)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// The pattern of ChaCha with a kept state: create a sum, then rotate it.
TEST_F(QOracleToQCTest, RotatesCreatedValue) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(zero %r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %s = arith.addi %va, %vb : i4
      %hi = arith.shli %s, %c1 : i4
      %lo = arith.shrui %s, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      qoracle.yield %rot : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], rotl(in[0] + in[1], 1, 4)};
  });
  EXPECT_EQ(countHelpers(body), 1U);
}

// Without the marker, the register holds an unknown value.
TEST_F(QOracleToQCTest, RejectsCreationInRegisterWithoutZeroMarker) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %s = arith.addi %va, %vb : i3
      qoracle.yield %s : i3
    })mlir",
                 "yielded value 0 does not match");
}

// `%t` reads the 0 in `r` after `%s` is created there.
TEST_F(QOracleToQCTest, RejectsReadOfZeroAliasThatReceivesCreatedValue) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            zero %r : memref<2x!qc.qubit> as %vr) {
      %s = arith.addi %va, %vb : i2
      %t = arith.andi %vr, %va : i2
      %next = arith.xori %vx, %t : i2
      qoracle.yield %next, %s : i2, i2
    })mlir",
                 "is read while a value is created in its register");
}

TEST_F(QOracleToQCTest, RejectsCreationOfUnsupportedOperation) {
  expectRejected(
      R"mlir(
    %k = qc.alloc : !qc.qubit
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%k : !qc.qubit as %vk)
                   consumed(zero %r : memref<3x!qc.qubit> as %vr) {
      %e = arith.extui %vk : i1 to i3
      qoracle.yield %e : i3
    })mlir",
      "an unsupported operation result is created in a zero register");
}

// `%t` has two users, so it is computed once into a helper register and
// cleaned up once at the end: 2w Toffolis instead of 2w per user.
TEST_F(QOracleToQCTest, CreatesSharedAndInHelperRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %x = memref.alloc() : memref<3x!qc.qubit>
    %y = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%x : memref<3x!qc.qubit> as %vx,
                            %y : memref<3x!qc.qubit> as %vy) {
      %t = arith.andi %va, %vb : i3
      %x1 = arith.xori %vx, %t : i3
      %y1 = arith.xori %vy, %t : i3
      qoracle.yield %x1, %y1 : i3, i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] & in[1]), in[3] ^ (in[0] & in[1])};
  });
  EXPECT_EQ(countToffolis(body), 6U);
  EXPECT_EQ(countHelpers(body), 3U);
}

// A temporary sum costs one addition per step that reads it; a helper register
// costs one to create it and one to clean it up. So with three users the
// register saves an addition. `a` and `b` are read after `%t`, so neither can
// be borrowed for the sum.
TEST_F(QOracleToQCTest, CreatesSharedSumInHelperRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    %z = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy,
                            %z : memref<2x!qc.qubit> as %vz) {
      %t = arith.addi %va, %vb : i2
      %x1 = arith.xori %vx, %t : i2
      %y1 = arith.xori %vy, %t : i2
      %z1 = arith.xori %vz, %t : i2
      %x2 = arith.xori %x1, %va : i2
      %y2 = arith.xori %y1, %vb : i2
      qoracle.yield %x2, %y2, %z1 : i2, i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    auto sum = in[0] + in[1];
    return {in[0], in[1], in[2] ^ sum ^ in[0], in[3] ^ sum ^ in[1],
            in[4] ^ sum};
  });
  // Each 2-bit addition takes 2 Toffolis.
  EXPECT_EQ(countToffolis(body), 4U);
}

// After `x` reads `%t`, `%t2` updates the helper register in place; R6 undoes
// both steps in reverse.
TEST_F(QOracleToQCTest, UpdatesHelperRegisterInPlace) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %c : memref<2x!qc.qubit> as %vc)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %t = arith.andi %va, %vb : i2
      %x1 = arith.xori %vx, %t : i2
      %t2 = arith.xori %t, %vc : i2
      %y1 = arith.xori %vy, %t2 : i2
      qoracle.yield %x1, %y1 : i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    auto t = in[0] & in[1];
    return {in[0], in[1], in[2], in[3] ^ t, in[4] ^ t ^ in[2]};
  });
  EXPECT_EQ(countToffolis(body), 4U);
  EXPECT_EQ(countHelpers(body), 2U);
}

// The temporary `%u` of the second step reads the helper register, which no
// step overwrites.
TEST_F(QOracleToQCTest, ReadsHelperRegisterInTemporary) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %c = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb,
                        %c : memref<2x!qc.qubit> as %vc)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %t = arith.andi %va, %vb : i2
      %x1 = arith.xori %vx, %t : i2
      %u = arith.andi %t, %vc : i2
      %y1 = arith.xori %vy, %u : i2
      qoracle.yield %x1, %y1 : i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    auto t = in[0] & in[1];
    return {in[0], in[1], in[2], in[3] ^ t, in[4] ^ (t & in[2])};
  });
  // `%t` once and its cleanup, plus `%u` and its cleanup.
  EXPECT_EQ(countToffolis(body), 8U);
}

// R6 undoes the creation of `%t` before it restores the borrowed `a`, because
// the creation reads `%s` from `a`.
TEST_F(QOracleToQCTest, CleansUpHelperRegisterBeforeBorrowedRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy) {
      %s = arith.addi %va, %vb : i2
      %t = arith.andi %s, %vb : i2
      %x1 = arith.xori %vx, %t : i2
      %y1 = arith.xori %vy, %t : i2
      qoracle.yield %x1, %y1 : i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    auto t = (in[0] + in[1]) & in[1];
    return {in[0], in[1], in[2] ^ t, in[3] ^ t};
  });
  // The helper register of `%t` and one carry for each of the two additions
  // on `a`. Without borrowing, `%s` would need two more helpers.
  EXPECT_LE(countHelpers(body), 4U);
}

// Cleaning up `%t` would need the old value of `x`, which `%x1` overwrites. The
// pass falls back to recomputing `%t` in each step and reports nothing.
TEST_F(QOracleToQCTest, FallsBackToTemporariesWhenCleanupIsIllegal) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %x = memref.alloc() : memref<2x!qc.qubit>
    %y = memref.alloc() : memref<2x!qc.qubit>
    %z = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va)
                   consumed(%x : memref<2x!qc.qubit> as %vx,
                            %y : memref<2x!qc.qubit> as %vy,
                            %z : memref<2x!qc.qubit> as %vz) {
      %t = arith.andi %va, %vx : i2
      %y1 = arith.xori %vy, %t : i2
      %z1 = arith.xori %vz, %t : i2
      %x1 = arith.xori %vx, %va : i2
      qoracle.yield %x1, %y1, %z1 : i2, i2, i2
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    auto t = in[0] & in[1];
    return {in[0], in[1] ^ in[0], in[2] ^ t, in[3] ^ t};
  });
  EXPECT_EQ(diagnostics, "");
  EXPECT_EQ(countToffolis(body), 8U);
}

// Inverting a loop is not supported, so R6 cannot clean up a helper register
// that a loop updates. Without helper registers, `%t` has no qubits to loop
// over.
TEST_F(QOracleToQCTest, RejectsLoopOverHelperRegister) {
  expectRejected(R"mlir(
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
      %c2 = arith.constant 2 : index
      %t = arith.andi %va, %vb : i2
      %x1 = arith.xori %vx, %t : i2
      %out = scf.for %i = %c0 to %c2 step %c1
          iter_args(%acc = %t) -> (i2) {
        %next = arith.xori %acc, %va : i2
        scf.yield %next : i2
      }
      %y1 = arith.xori %vy, %out : i2
      qoracle.yield %x1, %y1 : i2, i2
    })mlir",
                 "a loop initial value is not stored in consumed qubits");
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

// A value read inside a loop is computed once into a helper register before
// the loop and cleaned up after the last step.
TEST_F(QOracleToQCTest, ReadsHelperRegisterInsideLoop) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<2x!qc.qubit>
    %b = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%a : memref<2x!qc.qubit> as %va,
                        %b : memref<2x!qc.qubit> as %vb)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %ab = arith.andi %va, %vb : i2
      %out = scf.for %i = %c0 to %c3 step %c1
          iter_args(%acc = %vr) -> (i2) {
        %next = arith.xori %acc, %ab : i2
        scf.yield %next : i2
      }
      qoracle.yield %out : i2
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ (in[0] & in[1])};
                 });
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
