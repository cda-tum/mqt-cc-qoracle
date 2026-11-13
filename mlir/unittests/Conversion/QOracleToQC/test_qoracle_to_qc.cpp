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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
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

#include <algorithm>
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
  /// Classical values, such as loop induction variables and the conditions of
  /// `scf.if` operations, are evaluated by folding.
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
  DenseMap<Value, Attribute> classical_;
  DenseMap<Operation*, uint64_t> initialValues_;
  DenseMap<Operation*, SmallVector<size_t>> inputBits_;

  [[nodiscard]] SmallVector<size_t> allocate(size_t width, uint64_t value);
  [[nodiscard]] FailureOr<size_t> bitOf(Value qubit);
  [[nodiscard]] int64_t indexOf(Value value) const;
  [[nodiscard]] LogicalResult evaluate(Operation& op);
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

int64_t BasisSimulator::indexOf(Value value) const {
  return cast<IntegerAttr>(classical_.at(value)).getInt();
}

LogicalResult BasisSimulator::evaluate(Operation& op) {
  SmallVector<Attribute> operands;
  for (auto operand : op.getOperands()) {
    operands.push_back(classical_.lookup(operand));
  }
  SmallVector<OpFoldResult> results;
  if (failed(op.fold(operands, results)) ||
      results.size() != op.getNumResults()) {
    ADD_FAILURE() << "cannot evaluate " << op.getName().getStringRef().str();
    return failure();
  }
  for (auto [result, folded] : llvm::zip_equal(op.getResults(), results)) {
    if (auto value = dyn_cast<Value>(folded)) {
      classical_[result] = classical_.lookup(value);
    } else {
      classical_[result] = cast<Attribute>(folded);
    }
  }
  return success();
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
      const auto upper = indexOf(loop.getUpperBound());
      const auto step = indexOf(loop.getStep());
      Builder builder(op.getContext());
      for (auto i = indexOf(loop.getLowerBound()); i < upper; i += step) {
        classical_[loop.getInductionVar()] = builder.getIndexAttr(i);
        if (failed(runBlock(*loop.getBody()))) {
          return failure();
        }
      }
    } else if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults() != 0) {
        ADD_FAILURE() << "branch returns values";
        return failure();
      }
      const auto taken = cast<IntegerAttr>(classical_.at(branch.getCondition()))
                             .getValue()
                             .getBoolValue();
      auto& region = taken ? branch.getThenRegion() : branch.getElseRegion();
      if (!region.empty() && failed(runBlock(region.front()))) {
        return failure();
      }
    } else if (isa<arith::ArithDialect, tensor::TensorDialect>(
                   op.getDialect())) {
      // Classical values, including constants captured by an oracle, hold no
      // qubits.
      if (failed(evaluate(op))) {
        return failure();
      }
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      const auto index = indexOf(load.getIndices().front());
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
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                mqt::MQTDialect, qc::QCDialect, qoracle::QOracleDialect,
                scf::SCFDialect, tensor::TensorDialect>();
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
    return countControlledX(body, 2);
  }

  /// Lower `body` and return the number of CNOT gates in the circuit, with
  /// each loop body counted once.
  [[nodiscard]] size_t countCnots(StringRef body) {
    return countControlledX(body, 1);
  }

  [[nodiscard]] size_t countControlledX(StringRef body, size_t controls) {
    auto moduleOp = parseBody(body);
    if (!moduleOp || !runPass(*moduleOp)) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    size_t gates = 0;
    moduleOp->walk([&](qc::CtrlOp ctrl) {
      if (ctrl.getNumControls() == controls) {
        ++gates;
      }
    });
    return gates;
  }

  /// Lower `body` and return the largest number of helper qubits allocated at
  /// once, with each loop body counted once.
  [[nodiscard]] size_t countPeakHelpers(StringRef body) {
    auto moduleOp = parseBody(body);
    if (!moduleOp) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    const auto inputs = countAllocations(*moduleOp);
    if (!runPass(*moduleOp)) {
      ADD_FAILURE() << diagnostics;
      return 0;
    }
    size_t live = 0;
    size_t peak = 0;
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<qc::AllocOp>(op)) {
        peak = std::max(peak, ++live);
      } else if (isa<qc::DeallocOp>(op)) {
        --live;
      }
    });
    return peak - inputs;
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
      %next = arith.muli %vr, %vk : i1
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

// The temporary `t` is evaluated in the step of `y1`, so it still reads `x`
// after `x1`. Without reordering, `x1` cannot overwrite `x` in place.
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
                 "yielded value 0 does not match");
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

// `r | a` loses information about `r`, so OR never overwrites an operand.
TEST_F(QOracleToQCTest, RejectsOrIntoItsOperand) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %next = arith.ori %vr, %va : i4
      qoracle.yield %next : i4
    })mlir",
                 "yielded value 0 does not match");
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

// The rotation of `d` is evaluated in the step of `nextR`, so it still reads
// `d` after `nextD`. Without reordering, `nextD` cannot overwrite `d` in place.
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
                 "yielded value 0 does not match");
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

// NOT is `xori x, -1`. A NOT of a kept register is applied in place with X
// gates and undone at the end, as for any borrowed register, so it needs no
// helpers.
TEST_F(QOracleToQCTest, XorsInvertedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %ones = arith.constant -1 : i4
      %na = arith.xori %va, %ones : i4
      %next = arith.xori %vr, %na : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ ~in[0]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// The form of Ch in FIPS 180-4: (e & f) ^ (~e & g).
TEST_F(QOracleToQCTest, AndInvertedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %e = memref.alloc() : memref<4x!qc.qubit>
    %g = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%e : memref<4x!qc.qubit> as %ve,
                        %g : memref<4x!qc.qubit> as %vg)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %ones = arith.constant -1 : i4
      %ne = arith.xori %ve, %ones : i4
      %and = arith.andi %ne, %vg : i4
      %next = arith.xori %vr, %and : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (~in[0] & in[1])};
  });
  EXPECT_EQ(countHelpers(body), 4U);
}

// The NOT folds into the update, so only the AND needs helpers.
TEST_F(QOracleToQCTest, XorsNandOfKeptRegisters) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %ones = arith.constant -1 : i4
      %and = arith.andi %va, %vb : i4
      %nand = arith.xori %and, %ones : i4
      %next = arith.xori %vr, %nand : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ ~(in[0] & in[1])};
  });
  EXPECT_EQ(countHelpers(body), 4U);
}

// r + ~a = r - a - 1, so the carries must see the inverted bits.
TEST_F(QOracleToQCTest, AddsInvertedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %ones = arith.constant -1 : i4
      %na = arith.xori %va, %ones : i4
      %next = arith.addi %vr, %na : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] + ~in[0]};
  });
  EXPECT_EQ(countHelpers(body), 1U);
}

// The top bit of `a >> 1` is known to be 0, so the top bit of its inverse is 1.
// The inverse folds into the update, so no helpers are needed.
TEST_F(QOracleToQCTest, XorsInvertedShift) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %ones = arith.constant -1 : i4
      %sh = arith.shrui %va, %c1 : i4
      %nsh = arith.xori %sh, %ones : i4
      %next = arith.xori %vr, %nsh : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ ~(in[0] >> 1)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// A NOT of a kept register goes straight into the `zero` register: CNOTs, then
// X gates.
TEST_F(QOracleToQCTest, CreatesInvertedKeptRegisterInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %t = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(zero %t : memref<4x!qc.qubit> as %vt) {
      %ones = arith.constant -1 : i4
      %na = arith.xori %va, %ones : i4
      qoracle.yield %na : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], ~in[0]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// The form of SHA-256's sigma functions. A chain of XORs is one XOR of all its
// operands, so the update needs no helpers: one CNOT per operand bit, and none
// for the known-0 top bit of the shift. All operands read `a`, so borrowing
// `a` is impossible.
TEST_F(QOracleToQCTest, XorsRotationsAndShiftWithoutHelpers) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %c3 = arith.constant 3 : i4
      %hi1 = arith.shli %va, %c1 : i4
      %lo1 = arith.shrui %va, %c3 : i4
      %rot1 = arith.ori %hi1, %lo1 : i4
      %hi2 = arith.shli %va, %c2 : i4
      %lo2 = arith.shrui %va, %c2 : i4
      %rot2 = arith.ori %hi2, %lo2 : i4
      %sh = arith.shrui %va, %c1 : i4
      %x = arith.xori %rot1, %rot2 : i4
      %sigma = arith.xori %x, %sh : i4
      %next = arith.xori %vr, %sigma : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0],
            in[1] ^ rotl(in[0], 1, 4) ^ rotl(in[0], 2, 4) ^ (in[0] >> 1)};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_EQ(countCnots(body), 11U);
}

// The NOT of a XOR folds in as well: r ^= rotl(a, 1) ^ a, then X gates on `r`.
TEST_F(QOracleToQCTest, XorsInvertedXorWithoutHelpers) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %ones = arith.constant -1 : i4
      %hi = arith.shli %va, %c1 : i4
      %lo = arith.shrui %va, %c3 : i4
      %rot = arith.ori %hi, %lo : i4
      %x = arith.xori %rot, %va : i4
      %nx = arith.xori %x, %ones : i4
      %next = arith.xori %vr, %nx : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ ~(rotl(in[0], 1, 4) ^ in[0])};
  });
  EXPECT_EQ(countHelpers(body), 0U);
}

// Both registers are read again after the sum, so neither can be borrowed and
// the sum is a temporary. Its adder writes the copy of `a` in place, so that
// copy is not a finished XOR and must not be fused into its reader.
TEST_F(QOracleToQCTest, XorsTemporarySumOfRegistersReadLater) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %s = arith.addi %va, %vb : i4
      %x = arith.xori %vr, %s : i4
      %y = arith.xori %x, %va : i4
      %next = arith.xori %y, %vb : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1],
                           in[2] ^ (in[0] + in[1]) ^ in[0] ^ in[1]};
                 });
}

// The pattern of SHA-256's h + S1(e). The adder reads its operand more than
// once, so the XOR chain needs a temporary for each bit but the top one, which
// only the top sum bit reads. One more helper holds a carry.
TEST_F(QOracleToQCTest, AddsSigmaWithOneTemporary) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %c2 = arith.constant 2 : i4
      %c3 = arith.constant 3 : i4
      %hi1 = arith.shli %va, %c1 : i4
      %lo1 = arith.shrui %va, %c3 : i4
      %rot1 = arith.ori %hi1, %lo1 : i4
      %hi2 = arith.shli %va, %c2 : i4
      %lo2 = arith.shrui %va, %c2 : i4
      %rot2 = arith.ori %hi2, %lo2 : i4
      %sh = arith.shrui %va, %c1 : i4
      %x = arith.xori %rot1, %rot2 : i4
      %sigma = arith.xori %x, %sh : i4
      %next = arith.addi %vr, %sigma : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0],
            in[1] + (rotl(in[0], 1, 4) ^ rotl(in[0], 2, 4) ^ (in[0] >> 1))};
  });
  EXPECT_EQ(countHelpers(body), 4U);
}

// OR needs one Toffoli per bit, and one more to uncompute the temporary.
TEST_F(QOracleToQCTest, XorsOrOfKeptRegisters) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %or = arith.ori %va, %vb : i4
      %next = arith.xori %vr, %or : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] | in[1])};
  });
  EXPECT_EQ(countHelpers(body), 4U);
  EXPECT_EQ(countToffolis(body), 8U);
}

// The form of round 4 of MD5: c ^ (b | ~d). The NOT of `d` is applied in
// place, so only the OR needs helpers.
TEST_F(QOracleToQCTest, XorsOrWithInvertedKeptRegister) {
  constexpr StringRef body = R"mlir(
    %b = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    %c = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%b : memref<4x!qc.qubit> as %vb,
                        %d : memref<4x!qc.qubit> as %vd)
                   consumed(%c : memref<4x!qc.qubit> as %vc) {
      %ones = arith.constant -1 : i4
      %nd = arith.xori %vd, %ones : i4
      %or = arith.ori %vb, %nd : i4
      %next = arith.xori %vc, %or : i4
      qoracle.yield %next : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] | ~in[1])};
  });
  EXPECT_EQ(countHelpers(body), 4U);
  EXPECT_EQ(countToffolis(body), 8U);
}

TEST_F(QOracleToQCTest, AddsOrOfKeptRegisters) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %or = arith.ori %va, %vb : i4
      %next = arith.addi %vr, %or : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] + (in[0] | in[1])};
                 });
}

// A `zero` register starts at 0, so the OR is computed straight into it. As
// t ^= ~(~a & ~b), it needs one Toffoli per bit and X gates, but no CNOTs.
TEST_F(QOracleToQCTest, CreatesOrInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %t = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(zero %t : memref<4x!qc.qubit> as %vt) {
      %or = arith.ori %va, %vb : i4
      qoracle.yield %or : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] | in[1]};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_EQ(countToffolis(body), 4U);
  EXPECT_EQ(countCnots(body), 0U);
}

// The top bit of `a >> 1` is known to be 0, so the top bit of the OR is `b3`.
TEST_F(QOracleToQCTest, OrsShiftedKeptRegister) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i4
      %sh = arith.shrui %va, %c1 : i4
      %or = arith.ori %sh, %vb : i4
      %next = arith.xori %vr, %or : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] ^ ((in[0] >> 1) | in[1])};
                 });
}

TEST_F(QOracleToQCTest, OrsConstant) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%r : memref<4x!qc.qubit> as %vr) {
      %five = arith.constant 5 : i4
      %or = arith.ori %va, %five : i4
      %next = arith.xori %vr, %or : i4
      qoracle.yield %next : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ (in[0] | 5U)};
                 });
}

// An OR with two readers goes into a helper register: it is computed once and
// cleaned up once, instead of twice as a temporary.
TEST_F(QOracleToQCTest, CreatesSharedOrInHelperRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    %s = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%r : memref<3x!qc.qubit> as %vr,
                            %s : memref<3x!qc.qubit> as %vs) {
      %or = arith.ori %va, %vb : i3
      %nextR = arith.xori %vr, %or : i3
      %nextS = arith.xori %vs, %or : i3
      qoracle.yield %nextR, %nextS : i3, i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2] ^ (in[0] | in[1]), in[3] ^ (in[0] | in[1])};
  });
  EXPECT_EQ(countToffolis(body), 6U);
}

TEST_F(QOracleToQCTest, OrsInsideLoop) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %three = arith.constant 3 : index
      %out = scf.for %i = %c0 to %three step %one
          iter_args(%acc = %vd) -> (i4) {
        %or = arith.ori %va, %vb : i4
        %next = arith.addi %acc, %or : i4
        scf.yield %next : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2] + (3 * (in[0] | in[1]))};
                 });
}

// The majority of `a`, `b` and `c`, as mlir_emit writes it. Recognized as one
// operation, it is computed in place in one of its operands with one Toffoli
// per bit, and undone with one more. Three separate ANDs cost far more.
TEST_F(QOracleToQCTest, XorsMajority) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %or = arith.ori %ab, %ac : i3
      %maj = arith.ori %or, %bc : i3
      %next = arith.xori %vr, %maj : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            in[3] ^ ((in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2]))};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_LE(countToffolis(body), 6U);
}

// The form of Maj in FIPS 180-4: (a & b) ^ (a & c) ^ (b & c).
TEST_F(QOracleToQCTest, XorsMajorityWrittenWithXor) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %x = arith.xori %ab, %ac : i3
      %maj = arith.xori %x, %bc : i3
      %next = arith.xori %vr, %maj : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            in[3] ^ ((in[0] & in[1]) ^ (in[0] & in[2]) ^ (in[1] & in[2]))};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_LE(countToffolis(body), 6U);
}

// The pairs may come in any order, with their operands swapped and either side
// nested, and `|` and `^` may be mixed: no two of the three ANDs are 1 at once.
TEST_F(QOracleToQCTest, RecognizesMajorityInAnyOrder) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %cb = arith.andi %vc, %vb : i3
      %ba = arith.andi %vb, %va : i3
      %ca = arith.andi %vc, %va : i3
      %x = arith.xori %ba, %ca : i3
      %maj = arith.ori %cb, %x : i3
      %next = arith.xori %vr, %maj : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            in[3] ^ ((in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2]))};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_LE(countToffolis(body), 6U);
}

// The prototype needs one carry helper and 10 Toffolis: 6 for the adder and
// 4 for the majority and its inverse.
TEST_F(QOracleToQCTest, AddsMajority) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %or = arith.ori %ab, %ac : i3
      %maj = arith.ori %or, %bc : i3
      %next = arith.addi %vr, %maj : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            in[3] + ((in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2]))};
  });
  EXPECT_LE(countHelpers(body), 1U);
  EXPECT_LE(countToffolis(body), 10U);
}

// The pattern of SHA-256's r + S0(a) + Maj(a, b, c), regrouped as R4 does: the
// rotation reads `a` too, so the majority must be computed in place in `b` or
// `c`. Each addition allocates its own carry helper and frees it again.
TEST_F(QOracleToQCTest, AddsRotationAndMajorityOfSameRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %c1 = arith.constant 1 : i3
      %c2 = arith.constant 2 : i3
      %hi = arith.shli %va, %c1 : i3
      %lo = arith.shrui %va, %c2 : i3
      %rot = arith.ori %hi, %lo : i3
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %or = arith.ori %ab, %ac : i3
      %maj = arith.ori %or, %bc : i3
      %sum = arith.addi %vr, %rot : i3
      %next = arith.addi %sum, %maj : i3
      qoracle.yield %next : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    const auto maj = (in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2]);
    return {in[0], in[1], in[2], in[3] + rotl(in[0], 1, 3) + maj};
  });
  EXPECT_EQ(countHelpers(body), 2U);
  EXPECT_LE(countToffolis(body), 14U);
}

// A `zero` register starts at 0, so the majority is computed straight into
// it, with one Toffoli per bit.
TEST_F(QOracleToQCTest, CreatesMajorityInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %t = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(zero %t : memref<3x!qc.qubit> as %vt) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %or = arith.ori %ab, %ac : i3
      %maj = arith.ori %or, %bc : i3
      qoracle.yield %maj : i3
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            (in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2])};
  });
  EXPECT_EQ(countHelpers(body), 0U);
  EXPECT_EQ(countToffolis(body), 3U);
}

// `b & d` breaks the pattern, so this is not a majority. A recognizer that
// checks only the shape would compute the wrong function.
TEST_F(QOracleToQCTest, LowersNearMajority) {
  expectLowersTo(
      R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %d = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc,
                        %d : memref<3x!qc.qubit> as %vd)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bd = arith.andi %vb, %vd : i3
      %x = arith.xori %ab, %ac : i3
      %y = arith.xori %x, %bd : i3
      %next = arith.xori %vr, %y : i3
      qoracle.yield %next : i3
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues {
        return {in[0], in[1], in[2], in[3],
                in[4] ^ ((in[0] & in[1]) ^ (in[0] & in[2]) ^ (in[1] & in[3]))};
      });
}

// The inner OR reads a register directly, not an AND, so this is no majority.
TEST_F(QOracleToQCTest, LowersOrOfRegisterAndPairs) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr) {
      %ab = arith.andi %va, %vb : i3
      %bc = arith.andi %vb, %vc : i3
      %x = arith.ori %va, %ab : i3
      %y = arith.ori %x, %bc : i3
      %next = arith.xori %vr, %y : i3
      qoracle.yield %next : i3
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1], in[2],
                           in[3] ^ (in[0] | (in[0] & in[1]) | (in[1] & in[2]))};
                 });
}

// `a & b` has a second reader, so the pattern cannot absorb it. The circuit
// is still correct.
TEST_F(QOracleToQCTest, LowersMajorityWithSharedPair) {
  expectLowersTo(
      R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %r = memref.alloc() : memref<3x!qc.qubit>
    %s = memref.alloc() : memref<3x!qc.qubit>
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%r : memref<3x!qc.qubit> as %vr,
                            %s : memref<3x!qc.qubit> as %vs) {
      %ab = arith.andi %va, %vb : i3
      %ac = arith.andi %va, %vc : i3
      %bc = arith.andi %vb, %vc : i3
      %or = arith.ori %ab, %ac : i3
      %maj = arith.ori %or, %bc : i3
      %nextR = arith.xori %vr, %maj : i3
      %nextS = arith.xori %vs, %ab : i3
      qoracle.yield %nextR, %nextS : i3, i3
    })mlir",
      [](ArrayRef<uint64_t> in) -> RegisterValues {
        const auto maj = (in[0] & in[1]) | (in[0] & in[2]) | (in[1] & in[2]);
        return {in[0], in[1], in[2], in[3] ^ maj, in[4] ^ (in[0] & in[1])};
      });
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

// An XOR target is a consumed register yielded as `y ^ f(x)`: every
// intermediate value of `f` must be cleaned up. The programs below compute `f`
// from kept registers that are read again at the end, as in ChaCha's
// feed-forward, so no input can be borrowed.

namespace {
/// `rounds` rounds of `a += b; b = rotl(b ^ a, 1)` on kept 4-bit `a` and `b`,
/// then `y ^= (a + a0) ^ (b ^ b0)` with the inputs `a0` and `b0`.
std::string feedForwardBody(unsigned rounds) {
  std::string body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %y = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %a0,
                        %b : memref<4x!qc.qubit> as %b0)
                   consumed(%y : memref<4x!qc.qubit> as %vy) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
)mlir";
  for (unsigned i = 0; i < rounds; ++i) {
    const auto a = "%a" + std::to_string(i);
    const auto b = "%b" + std::to_string(i);
    const auto next = std::to_string(i + 1);
    const auto t = "%t" + std::to_string(i);
    body += "      %a" + next + " = arith.addi " + a + ", " + b + " : i4\n";
    body += "      " + t + " = arith.xori " + b + ", %a" + next + " : i4\n";
    body += "      %hi" + next + " = arith.shli " + t + ", %c1 : i4\n";
    body += "      %lo" + next + " = arith.shrui " + t + ", %c3 : i4\n";
    body += "      %b" + next + " = arith.ori %hi" + next + ", %lo" + next +
            " : i4\n";
  }
  const auto last = std::to_string(rounds);
  body += "      %fa = arith.addi %a" + last + ", %a0 : i4\n";
  body += "      %fb = arith.xori %b" + last + ", %b0 : i4\n";
  body += R"mlir(      %f = arith.xori %fa, %fb : i4
      %next = arith.xori %vy, %f : i4
      qoracle.yield %next : i4
    })mlir";
  return body;
}

RegisterValues feedForward(ArrayRef<uint64_t> in, unsigned rounds) {
  uint64_t a = in[0];
  uint64_t b = in[1];
  for (unsigned i = 0; i < rounds; ++i) {
    a = (a + b) & 0xF;
    b = rotl(b ^ a, 1, 4);
  }
  return {in[0], in[1], in[2] ^ ((a + in[0]) ^ b ^ in[1])};
}

/// `rounds` rounds of `x = rotl((x + m1) ^ m2, 1)` from the constant `x = 5`,
/// with kept `m1` and `m2`, then `y ^= x + 5`: the state chain never runs in an
/// input's qubits, as in a hash that starts from its IV.
std::string fromConstantBody(unsigned rounds) {
  std::string body = R"mlir(
    %m1 = memref.alloc() : memref<4x!qc.qubit>
    %m2 = memref.alloc() : memref<4x!qc.qubit>
    %y = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%m1 : memref<4x!qc.qubit> as %vm1,
                        %m2 : memref<4x!qc.qubit> as %vm2)
                   consumed(%y : memref<4x!qc.qubit> as %vy) {
      %c1 = arith.constant 1 : i4
      %c3 = arith.constant 3 : i4
      %x0 = arith.constant 5 : i4
)mlir";
  for (unsigned i = 0; i < rounds; ++i) {
    const auto x = "%x" + std::to_string(i);
    const auto n = std::to_string(i + 1);
    body += "      %s" + n + " = arith.addi " + x + ", %vm1 : i4\n";
    body += "      %t" + n + " = arith.xori %s" + n + ", %vm2 : i4\n";
    body += "      %hi" + n + " = arith.shli %t" + n + ", %c1 : i4\n";
    body += "      %lo" + n + " = arith.shrui %t" + n + ", %c3 : i4\n";
    body += "      %x" + n + " = arith.ori %hi" + n + ", %lo" + n + " : i4\n";
  }
  body += "      %f = arith.addi %x" + std::to_string(rounds) + ", %x0 : i4\n";
  body += R"mlir(      %next = arith.xori %vy, %f : i4
      qoracle.yield %next : i4
    })mlir";
  return body;
}

RegisterValues fromConstant(ArrayRef<uint64_t> in, unsigned rounds) {
  uint64_t x = 5;
  for (unsigned i = 0; i < rounds; ++i) {
    x = rotl(((x + in[0]) & 0xF) ^ in[1], 1, 4);
  }
  return {in[0], in[1], in[2] ^ (x + 5)};
}
} // namespace

TEST_F(QOracleToQCTest, XorsFeedForwardChainIntoTarget) {
  expectLowersTo(feedForwardBody(1),
                 [](ArrayRef<uint64_t> in) { return feedForward(in, 1); });
}

// The state words are updated in place in helper registers, so more rounds
// cost more gates but no more qubits. The prototype needs 15 helpers.
TEST_F(QOracleToQCTest, KeepsHelpersConstantAcrossRounds) {
  expectLowersTo(feedForwardBody(3),
                 [](ArrayRef<uint64_t> in) { return feedForward(in, 3); });
  const auto oneRound = countPeakHelpers(feedForwardBody(1));
  const auto threeRounds = countPeakHelpers(feedForwardBody(3));
  EXPECT_EQ(threeRounds, oneRound);
  EXPECT_LE(threeRounds, 15U);
}

// A chain that starts at a constant runs in one helper register. The
// prototype needs 7 helpers.
TEST_F(QOracleToQCTest, ComputesChainFromConstantInOneRegister) {
  expectLowersTo(fromConstantBody(3),
                 [](ArrayRef<uint64_t> in) { return fromConstant(in, 3); });
  // With one round, `m1` and `m2` are read once and can be borrowed instead.
  const auto twoRounds = countPeakHelpers(fromConstantBody(2));
  const auto threeRounds = countPeakHelpers(fromConstantBody(3));
  EXPECT_EQ(threeRounds, twoRounds);
  EXPECT_LE(threeRounds, 7U);
}

// A predicate oracle: one bit of `f` is XORed into a single target qubit.
TEST_F(QOracleToQCTest, XorsPredicateIntoTargetQubit) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %a0,
                        %b : memref<4x!qc.qubit> as %b0)
                   consumed(%y : !qc.qubit as %vy) {
      %a1 = arith.addi %a0, %b0 : i4
      %t = arith.xori %b0, %a1 : i4
      %fa = arith.addi %a1, %a0 : i4
      %f = arith.xori %fa, %t : i4
      %bit = arith.trunci %f : i4 to i1
      %next = arith.xori %vy, %bit : i1
      qoracle.yield %next : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const auto a1 = (in[0] + in[1]) & 0xF;
                   const auto f = ((a1 + in[0]) ^ (in[1] ^ a1)) & 1;
                   return {in[0], in[1], in[2] ^ f};
                 });
}

// A predicate compares words with constants and XORs the result into a target
// qubit. `a == 5` is the AND of the bits of `a`, with the bits where 5 has a 0
// inverted: three Toffolis for 4 bits, and as many to uncompute.
TEST_F(QOracleToQCTest, XorsComparisonWithConstantIntoTarget) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%y : !qc.qubit as %vy) {
      %c5 = arith.constant 5 : i4
      %eq = arith.cmpi eq, %va, %c5 : i4
      %next = arith.xori %vy, %eq : i1
      qoracle.yield %next : i1
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1] ^ static_cast<uint64_t>(in[0] == 5)};
  });
  EXPECT_LE(countToffolis(body), 6U);
  EXPECT_LE(countPeakHelpers(body), 3U);
}

// Comparing with 0 inverts every bit; comparing with all ones inverts none.
TEST_F(QOracleToQCTest, ComparesWithZeroAndAllOnes) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    %z = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%y : !qc.qubit as %vy, %z : !qc.qubit as %vz) {
      %c0 = arith.constant 0 : i4
      %ones = arith.constant -1 : i4
      %isZero = arith.cmpi eq, %va, %c0 : i4
      %isOnes = arith.cmpi eq, %va, %ones : i4
      %nextY = arith.xori %vy, %isZero : i1
      %nextZ = arith.xori %vz, %isOnes : i1
      qoracle.yield %nextY, %nextZ : i1, i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], in[1] ^ static_cast<uint64_t>(in[0] == 0),
                           in[2] ^ static_cast<uint64_t>(in[0] == 15)};
                 });
}

// The predicate of several words is the AND of their comparisons.
TEST_F(QOracleToQCTest, AndComparisonsOfTwoRegisters) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%y : !qc.qubit as %vy) {
      %c5 = arith.constant 5 : i3
      %c2 = arith.constant 2 : i3
      %eqA = arith.cmpi eq, %va, %c5 : i3
      %eqB = arith.cmpi eq, %vb, %c2 : i3
      %f = arith.andi %eqA, %eqB : i1
      %next = arith.xori %vy, %f : i1
      qoracle.yield %next : i1
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1],
            in[2] ^ static_cast<uint64_t>(in[0] == 5 && in[1] == 2)};
  });
  // Each comparison runs in its own step and is undone again by R6: three
  // Toffolis each time, plus two for the AND into the target.
  EXPECT_LE(countToffolis(body), 14U);
}

// The compared word is computed first; its register is cleaned up afterwards.
TEST_F(QOracleToQCTest, ComparesSumWithConstant) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<3x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<3x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb)
                   consumed(%y : !qc.qubit as %vy) {
      %c3 = arith.constant 3 : i3
      %s = arith.addi %va, %vb : i3
      %eq = arith.cmpi eq, %s, %c3 : i3
      %next = arith.xori %vy, %eq : i1
      qoracle.yield %next : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const auto sum = (in[0] + in[1]) & 7;
                   return {in[0], in[1],
                           in[2] ^ static_cast<uint64_t>(sum == 3)};
                 });
}

// The shape of the ChaCha key-search predicate: a round with feed-forward, then
// a comparison of the result with a constant.
TEST_F(QOracleToQCTest, ComparesFeedForwardChainWithConstant) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %a0,
                        %b : memref<4x!qc.qubit> as %b0)
                   consumed(%y : !qc.qubit as %vy) {
      %c6 = arith.constant 6 : i4
      %a1 = arith.addi %a0, %b0 : i4
      %t = arith.xori %b0, %a1 : i4
      %fa = arith.addi %a1, %a0 : i4
      %f = arith.xori %fa, %t : i4
      %eq = arith.cmpi eq, %f, %c6 : i4
      %next = arith.xori %vy, %eq : i1
      qoracle.yield %next : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const auto a1 = (in[0] + in[1]) & 0xF;
                   const auto f = ((a1 + in[0]) ^ (in[1] ^ a1)) & 0xF;
                   return {in[0], in[1], in[2] ^ static_cast<uint64_t>(f == 6)};
                 });
}

// Created in a `zero` qubit, the comparison is an Update; its AND tree and the
// flips of the bits where 5 has a 0 are still temporaries and must be undone.
TEST_F(QOracleToQCTest, CreatesComparisonInZeroQubit) {
  expectLowersTo(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %t = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(zero %t : !qc.qubit as %vt) {
      %c5 = arith.constant 5 : i4
      %eq = arith.cmpi eq, %va, %c5 : i4
      qoracle.yield %eq : i1
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0], static_cast<uint64_t>(in[0] == 5)};
                 });
}

// Each comparison gets its own step and a one-qubit result, so only one AND
// tree is alive at a time: two tree helpers and up to three results. Computing
// all trees at once would need 10 helpers.
TEST_F(QOracleToQCTest, ComparesWordsOneAtATime) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %c = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb,
                        %c : memref<4x!qc.qubit> as %vc)
                   consumed(%y : !qc.qubit as %vy) {
      %c5 = arith.constant 5 : i4
      %c9 = arith.constant 9 : i4
      %c3 = arith.constant 3 : i4
      %eqA = arith.cmpi eq, %va, %c5 : i4
      %eqB = arith.cmpi eq, %vb, %c9 : i4
      %eqC = arith.cmpi eq, %vc, %c3 : i4
      %ab = arith.andi %eqA, %eqB : i1
      %f = arith.andi %ab, %eqC : i1
      %next = arith.xori %vy, %f : i1
      qoracle.yield %next : i1
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[2],
            in[3] ^
                static_cast<uint64_t>(in[0] == 5 && in[1] == 9 && in[2] == 3)};
  });
  EXPECT_LE(countPeakHelpers(body), 5U);
}

// R6 undoes the steps in reverse order, so each helper register is 0 once the
// inverse of its creation has run, and is freed then. When R6 recomputes the
// first comparison, with its six tree helpers, the later registers are gone.
TEST_F(QOracleToQCTest, FreesHelperRegistersDuringCleanup) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<8x!qc.qubit>
    %b = memref.alloc() : memref<3x!qc.qubit>
    %c = memref.alloc() : memref<3x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<8x!qc.qubit> as %va,
                        %b : memref<3x!qc.qubit> as %vb,
                        %c : memref<3x!qc.qubit> as %vc)
                   consumed(%y : !qc.qubit as %vy) {
      %k = arith.constant 90 : i8
      %c3 = arith.constant 3 : i3
      %eq1 = arith.cmpi eq, %va, %k : i8
      %s = arith.addi %vb, %vc : i3
      %eq2 = arith.cmpi eq, %s, %c3 : i3
      %f = arith.andi %eq1, %eq2 : i1
      %next = arith.xori %vy, %f : i1
      qoracle.yield %next : i1
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    const auto s = (in[1] + in[2]) & 7;
    return {in[0], in[1], in[2],
            in[3] ^ static_cast<uint64_t>(in[0] == 90 && s == 3)};
  });
  EXPECT_LE(countPeakHelpers(body), 7U);
}

// An addition with carry-out: the operands are zero-extended, and the sum is
// created in the wider `zero` register. An extension cannot be updated in
// place, so the creation starts at the addition: a copy of `a`, then `b` added
// in place, with one carry helper.
TEST_F(QOracleToQCTest, CreatesSumWithCarryOutInZeroRegister) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %s = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(zero %s : memref<5x!qc.qubit> as %vs) {
      %ea = arith.extui %va : i4 to i5
      %eb = arith.extui %vb : i4 to i5
      %sum = arith.addi %ea, %eb : i5
      qoracle.yield %sum : i5
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    return {in[0], in[1], in[0] + in[1]};
  });
  EXPECT_LE(countToffolis(body), 8U);
}

TEST_F(QOracleToQCTest, RejectsOrderedComparison) {
  expectRejected(R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %y = qc.alloc : !qc.qubit
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%y : !qc.qubit as %vy) {
      %c5 = arith.constant 5 : i4
      %lt = arith.cmpi ult, %va, %c5 : i4
      %next = arith.xori %vy, %lt : i1
      qoracle.yield %next : i1
    })mlir",
                 "contains invalid operations");
}

// Helpers of steps inside a loop are allocated before the loop. Each step
// leaves its helpers at 0, so the steps of one body share them: three
// additions need one carry helper, not three.
TEST_F(QOracleToQCTest, ReusesCarryHelpersInsideLoop) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    %e = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va)
                   consumed(%d : memref<4x!qc.qubit> as %vd,
                            %e : memref<4x!qc.qubit> as %ve) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %out:2 = scf.for %i = %c0 to %two step %one
          iter_args(%x = %vd, %y = %ve) -> (i4, i4) {
        %x1 = arith.addi %x, %va : i4
        %y1 = arith.addi %y, %x1 : i4
        %x2 = arith.addi %x1, %y1 : i4
        scf.yield %x2, %y1 : i4, i4
      }
      qoracle.yield %out#0, %out#1 : i4, i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    uint64_t x = in[1];
    uint64_t y = in[2];
    for (int i = 0; i < 2; ++i) {
      x = (x + in[0]) & 0xF;
      y = (y + x) & 0xF;
      x = (x + y) & 0xF;
    }
    return {in[0], x, y};
  });
  EXPECT_EQ(countPeakHelpers(body), 1U);
}

// Steps with different numbers of helpers share them as well: the first step
// needs four for the AND and one carry, the second step reuses that carry.
TEST_F(QOracleToQCTest, SharesHelpersOfDifferentStepsInsideLoop) {
  constexpr StringRef body = R"mlir(
    %a = memref.alloc() : memref<4x!qc.qubit>
    %b = memref.alloc() : memref<4x!qc.qubit>
    %d = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%a : memref<4x!qc.qubit> as %va,
                        %b : memref<4x!qc.qubit> as %vb)
                   consumed(%d : memref<4x!qc.qubit> as %vd) {
      %c0 = arith.constant 0 : index
      %one = arith.constant 1 : index
      %two = arith.constant 2 : index
      %out = scf.for %i = %c0 to %two step %one
          iter_args(%x = %vd) -> (i4) {
        %ab = arith.andi %va, %vb : i4
        %x1 = arith.addi %x, %ab : i4
        %x2 = arith.addi %x1, %va : i4
        scf.yield %x2 : i4
      }
      qoracle.yield %out : i4
    })mlir";
  expectLowersTo(body, [](ArrayRef<uint64_t> in) -> RegisterValues {
    uint64_t x = in[2];
    for (int i = 0; i < 2; ++i) {
      x = (x + (in[0] & in[1]) + in[0]) & 0xF;
    }
    return {in[0], in[1], x};
  });
  EXPECT_LE(countPeakHelpers(body), 5U);
}

// A loop body may read a different constant in each iteration, such as the
// round constant of a hash function. The constant is extracted from a constant
// tensor at the induction variable.
TEST_F(QOracleToQCTest, XorsPerIterationConstantInsideLoop) {
  expectLowersTo(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %k = arith.constant dense<[3, 5, 12]> : tensor<3xi4>
      %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %vr) -> (i4) {
        %ki = tensor.extract %k[%i] : tensor<3xi4>
        %next = arith.xori %acc, %ki : i4
        scf.yield %next : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0] ^ 3U ^ 5U ^ 12U};
                 });
}

TEST_F(QOracleToQCTest, AddsPerIterationConstantInsideLoop) {
  expectLowersTo(R"mlir(
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle consumed(%r : memref<5x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %k = arith.constant dense<[7, 12, 25, 0]> : tensor<4xi5>
      %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %vr) -> (i5) {
        %ki = tensor.extract %k[%i] : tensor<4xi5>
        %next = arith.addi %acc, %ki : i5
        scf.yield %next : i5
      }
      qoracle.yield %out : i5
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0] + 7U + 12U + 25U};
                 });
}

// A body that covers several iterations of the source loop, such as 16 rounds
// of SHA-256, reads the constants at offsets of the induction variable.
TEST_F(QOracleToQCTest, ReadsConstantsAtOffsetsOfInductionVariable) {
  expectLowersTo(R"mlir(
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle consumed(%r : memref<5x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %k = arith.constant dense<[7, 12, 25, 3]> : tensor<4xi5>
      %out = scf.for %i = %c0 to %c4 step %c2 iter_args(%acc = %vr) -> (i5) {
        %j = arith.addi %i, %c1 : index
        %ki = tensor.extract %k[%i] : tensor<4xi5>
        %kj = tensor.extract %k[%j] : tensor<4xi5>
        %sum = arith.addi %acc, %ki : i5
        %next = arith.xori %sum, %kj : i5
        scf.yield %next : i5
      }
      qoracle.yield %out : i5
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   const std::array<uint64_t, 4> k{7, 12, 25, 3};
                   auto x = in[0];
                   for (size_t i = 0; i < k.size(); i += 2) {
                     x = ((x + k[i]) & 31U) ^ k[i + 1];
                   }
                   return {x};
                 });
}

TEST_F(QOracleToQCTest, AndPerIterationConstantIntoConsumedRegister) {
  expectLowersTo(R"mlir(
    %x = memref.alloc() : memref<4x!qc.qubit>
    %y = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle kept(%x : memref<4x!qc.qubit> as %vx)
                   consumed(%y : memref<4x!qc.qubit> as %vy) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %k = arith.constant dense<[3, 10, 6]> : tensor<3xi4>
      %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %vy) -> (i4) {
        %ki = tensor.extract %k[%i] : tensor<3xi4>
        %masked = arith.andi %vx, %ki : i4
        %next = arith.xori %acc, %masked : i4
        scf.yield %next : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0],
                           in[1] ^ (in[0] & 3U) ^ (in[0] & 10U) ^ (in[0] & 6U)};
                 });
}

// Any classical value computed from the induction variable is a per-iteration
// constant, not only a tensor element.
TEST_F(QOracleToQCTest, XorsInductionVariableInsideLoop) {
  expectLowersTo(R"mlir(
    %r = memref.alloc() : memref<4x!qc.qubit>
    qoracle.oracle consumed(%r : memref<4x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %vr) -> (i4) {
        %v = arith.index_cast %i : index to i4
        %next = arith.xori %acc, %v : i4
        scf.yield %next : i4
      }
      qoracle.yield %out : i4
    })mlir",
                 [](ArrayRef<uint64_t> in) -> RegisterValues {
                   return {in[0] ^ 0U ^ 1U ^ 2U};
                 });
}

// Adding a per-iteration constant costs no more than adding the most expensive
// constant known at compile time, the one with every bit set. The lowering
// does not first load the constant into helper qubits.
TEST_F(QOracleToQCTest, AddsPerIterationConstantLikeFixedConstant) {
  constexpr StringRef perIteration = R"mlir(
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle consumed(%r : memref<5x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %k = arith.constant dense<[31, 0, 21]> : tensor<3xi5>
      %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %vr) -> (i5) {
        %ki = tensor.extract %k[%i] : tensor<3xi5>
        %next = arith.addi %acc, %ki : i5
        scf.yield %next : i5
      }
      qoracle.yield %out : i5
    })mlir";
  constexpr StringRef fixed = R"mlir(
    %r = memref.alloc() : memref<5x!qc.qubit>
    qoracle.oracle consumed(%r : memref<5x!qc.qubit> as %vr) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %k = arith.constant 31 : i5
      %out = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %vr) -> (i5) {
        %next = arith.addi %acc, %k : i5
        scf.yield %next : i5
      }
      qoracle.yield %out : i5
    })mlir";
  EXPECT_LE(countPeakHelpers(perIteration), countPeakHelpers(fixed));
  EXPECT_LE(countToffolis(perIteration), countToffolis(fixed));
}

TEST_F(QOracleToQCTest, RejectsTensorIndexedByQuantumValue) {
  expectRejected(R"mlir(
    %k = memref.alloc() : memref<2x!qc.qubit>
    %r = memref.alloc() : memref<2x!qc.qubit>
    qoracle.oracle kept(%k : memref<2x!qc.qubit> as %vk)
                   consumed(%r : memref<2x!qc.qubit> as %vr) {
      %table = arith.constant dense<[0, 1, 3, 2]> : tensor<4xi2>
      %index = arith.index_cast %vk : i2 to index
      %entry = tensor.extract %table[%index] : tensor<4xi2>
      %next = arith.xori %vr, %entry : i2
      qoracle.yield %next : i2
    })mlir",
                 "contains invalid operations");
}
