// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QORACLE_IR_QORACLEOPS_TD
#define MLIR_DIALECT_QORACLE_IR_QORACLEOPS_TD

include "mqt/Dialect/QOracle/IR/QOracleDialect.td"
include "mqt/Dialect/QC/IR/QCTypes.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Operation Classes
//===----------------------------------------------------------------------===//

class QOracleOp<string mnemonic, list<Trait> traits = []>
    : Op<QOracleDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//
def StaticQubitMemRef
    : ConfinedType<MemRefRankOf<[QubitType], [1]>,
                   [HasStaticShapePred,
                    CPred<"::llvm::cast<::mlir::MemRefType>($_self)"
                          ".getDimSize(0) <= ::mlir::IntegerType::kMaxWidth">],
                   "static 1-D memref of qubits">;
def OracleOperandType : AnyTypeOf<[QubitType, StaticQubitMemRef],
                                  "qubit or static 1-D memref of qubits">;

//===----------------------------------------------------------------------===//
// Oracle Definition Operations
//===----------------------------------------------------------------------===//

def OracleOp : QOracleOp<"oracle", traits = [AttrSizedOperandSegments]> {
  let summary = "Define and apply a quantum oracle";
  let description = [{
        Applies a classical function, given by the op's region, to qubits as
        a reversible quantum oracle.

        The operands are qubits or static-size memrefs of qubits, in
        two groups. Kept operands are not changed directly by the operation.
        Consumed operands are overwritten: after the op they hold the values
        yielded by the region.

        Inside the region, each operand is aliased to a classical integer:
        a qubit to i1, a memref of n qubits to i<n>, element 0 being bit 0.
        The region computes on these integers with standard dialects (e.g., arith, scf)
        and may use classical values defined outside the op, but no qubits from
        outside and no operations with side effects.
        Its terminator yields one integer per consumed operand, of the
        same type as that operand's alias.

        Semantics: on computational basis states the op maps
        |k⟩|c⟩ to |k⟩|f(k, c)⟩, where k are the kept operands, c the consumed
        ones and f the function computed by the region. Every auxiliary qubit
        introduced by the lowering is returned to |0⟩, so no garbage remains
        entangled with the operands. f must be invertible in c not marked as `zero`
        for every fixed k, otherwise the op is not unitary.
        Passing the same qubit multiple times is not allowed.

        Consumed operands marked  `zero` must be |0⟩ on entry.
        Inside the region, their alias has a value of 0.

        Example:
        ```mlir
        %ctrl = qc.alloc : !qc.qubit
        %memref = memref.alloc() : memref<8x!qc.qubit>
        qoracle.oracle kept(%ctrl : !qc.qubit as %k) consumed(%memref : memref<8x!qc.qubit> as %int) {
          %step = arith.extui %k : i1 to i8
          %new_int = arith.addi %int, %step : i8
          qoracle.yield %new_int : i8
        }
        ```
    }];

  let arguments = (ins Arg<Variadic<OracleOperandType>,
                           "the kept target qubits", [MemRead]>:$kept_qubits,
      Arg<Variadic<OracleOperandType>,
          "the consumed target qubits", [MemRead, MemWrite]>:$consumed_qubits,
      OptionalAttr<DenseBoolArrayAttr>:$consumed_zeros);
  let regions = (region SizedRegion<1>:$region);
  let assemblyFormat = [{
          custom<OracleAliasing>($kept_qubits, type($kept_qubits), $consumed_qubits, type($consumed_qubits), $consumed_zeros, $region)
          attr-dict
      }];

  let hasVerifier = 1;       // aliases match operands (count and widths)
  let hasRegionVerifier = 1; // no captured qubits; terminator is a yield

  let extraClassDeclaration = [{
    /// i1 for a qubit, i<n> for memref<n x qubit>.
    static ::mlir::IntegerType getAliasType(::mlir::Type operandType);
    /// Entry-block arguments of the body, split by operand group.
    ::mlir::Block::BlockArgListType getKeptAliases();
    ::mlir::Block::BlockArgListType getConsumedAliases();
    bool ownsValue(::mlir::Value value);
    bool isKeptAlias(::mlir::Value value);
    bool isConsumedAlias(::mlir::Value value);
    bool isZeroAlias(::mlir::Value value);
  }];
}

def YieldOp : QOracleOp<"yield", [Pure, Terminator, HasParent<"OracleOp">]> {
  let summary = "Return the new values of the consumed operands";
  let arguments = (ins Variadic<AnySignlessInteger>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
  let hasVerifier = 1; // one value per consumed operand, same alias type
}

#endif // MLIR_DIALECT_QORACLE_IR_QORACLEOPS_TD
