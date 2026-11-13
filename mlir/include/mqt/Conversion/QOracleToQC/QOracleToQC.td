// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

include "mlir/Pass/PassBase.td"

def QOracleToQC : Pass<"qoracle-to-qc", "mlir::ModuleOp"> {
  let summary = "Lower the `qoracle` dialect to `qc`.";

  let description = [{
    Replaces each `qoracle.oracle` with `qc` gates that apply the function its
    region computes, at the position of the oracle.

    On computational basis states, the kept qubits keep their values and the
    consumed qubits end up holding the values the region yields. Intermediate
    values live in helper qubits, which are returned to |0⟩ and deallocated
    after the oracle.

    The region may use `arith.xori`, `arith.andi`, `arith.addi`, integer
    constants, `arith.extui` and `arith.trunci`. Helper qubits are allocated
    dynamically, so programs with `qc.static` qubits are not supported yet.
    Other operations, and yielded values that cannot end up in the consumed
    qubits, are reported as errors before the IR is changed.
  }];

  let dependentDialects = ["mlir::qc::QCDialect", "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::tensor::TensorDialect"];
}
