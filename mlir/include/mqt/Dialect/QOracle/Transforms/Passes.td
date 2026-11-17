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
    the last reader of a value can overwrite it. The pass keeps the original
    order, except for an update whose operands in qubits are all still needed:
    a ready operation that still needs one of them, and that is not blocked
    itself, moves before the update. For example, with `e = d + t1` written
    after `a = t1 + t2`, the pass moves `e` first, so that `a` can overwrite
    `t1`. A temporary that the lowering computes in the step of its reader
    counts as a read in that step.

    The pass moves only side-effect-free operations, only within their block, and
    keeps each operation after the operations that define its operands. The
    computed function is unchanged, and the output is deterministic. Operations
    outside oracles are not changed.
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
