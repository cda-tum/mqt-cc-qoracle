// Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
// Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
// All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Licensed under the MIT License

#ifndef MLIR_DIALECT_QORACLE_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_QORACLE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def ScheduleOracle : Pass<"qoracle-schedule", "mlir::ModuleOp"> {
  let summary =
      "Reorder oracle bodies so that more values can be updated in place";
  let description = [{
    Reorders the operations in each block of every `qoracle.oracle` region so
    that more values can be overwritten in place when the oracle is lowered. Only
    the last reader of a value can overwrite it. The pass therefore moves readers
    that cannot overwrite their operand, such as `arith.andi`, before readers that
    can, such as an `arith.addi` on a consumed alias.

    The pass moves only side-effect-free operations, only within their block, and
    keeps each operation after the operations that define its operands. The
    computed function is unchanged. Ties keep the original order, so the output is
    deterministic. Operations outside oracles are not changed.
  }];
}

def ReassociateOracle : Pass<"qoracle-reassociate", "mlir::ModuleOp"> {
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let summary = "Regroup additions in oracle bodies so that more sums can be "
                "computed in place";
  let description = [{
    Regroups chains of `arith.addi` in `qoracle.oracle` regions so that each sum
    starts at an operand the lowering may overwrite: one that the sum is the
    last to read and that is not a kept alias. Operands with qubits of their
    own, such as consumed aliases and earlier sums, take priority over
    single-use bitwise temporaries, which the lowering computes into fresh
    helpers. For example, with kept `a` and `b` and consumed `c`,
    `(a + b) + c` becomes `a + (b + c)`, which can be computed in the qubits of
    `c` without a fresh register.

    Only intermediate sums without other uses are regrouped. Constant
    terms are folded into one. Operations outside oracles are not changed.
  }];
}

#endif // MLIR_DIALECT_QORACLE_TRANSFORMS_PASSES_TD
