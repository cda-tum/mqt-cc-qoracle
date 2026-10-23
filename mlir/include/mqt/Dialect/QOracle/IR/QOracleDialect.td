// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QORACLE_IR_QORACLEDIALECT_TD
#define MLIR_DIALECT_QORACLE_IR_QORACLEDIALECT_TD

include "mlir/IR/DialectBase.td"

def QOracleDialect : Dialect {
  let name = "qoracle";

  let summary = "The oracle dialect for quantum computing.";

  let description = [{
        The qoracle dialect enables the definition of
        quantum oracles by the classical function they compute.
        A `qoracle.oracle` operation applies a reversible function, written
        with `arith` and `scf` operations on integer aliases of qubits, to QC qubits.
        Lowering passes synthesize the circuit, optimize it, and uncompute any auxiliary qubits.
    }];

  let cppNamespace = "::mlir::qoracle";
  let dependentDialects = ["::mlir::arith::ArithDialect",
                           "::mlir::scf::SCFDialect", "::mlir::qc::QCDialect"];
}

#endif // MLIR_DIALECT_QORACLE_IR_QORACLEDIALECT_TD
