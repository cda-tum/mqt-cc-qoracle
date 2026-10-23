/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::qoracle;

LogicalResult YieldOp::verify() {
  auto oracleOp = cast<OracleOp>(getOperation()->getParentOp());
  auto consumedQubits = oracleOp.getConsumedQubits();
  if (consumedQubits.size() != getOperands().size()) {
    return emitOpError() << "must yield the same number of values ("
                         << getOperands().size()
                         << ") as consumed by the parent qoracle.oracle ("
                         << consumedQubits.size() << ")";
  }
  for (auto [index, values] :
       llvm::enumerate(llvm::zip_equal(getOperands(), consumedQubits))) {
    auto [yielded, consumed] = values;
    auto expectedType = OracleOp::getAliasType(consumed.getType());
    if (yielded.getType() != expectedType) {
      return emitOpError()
             << "type of yielded value " << index << " (" << yielded.getType()
             << ") does not match the expected alias type of consumed qubit "
             << index << " (" << expectedType << ")";
    }
  }

  return success();
}
