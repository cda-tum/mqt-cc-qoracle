/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QC/IR/QCDialect.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleDialect.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::qoracle;

LogicalResult OracleOp::verify() {
  auto blockArgs = getRegion().getArguments();
  if (blockArgs.size() != getKeptQubits().size() + getConsumedQubits().size()) {
    return emitOpError() << "must have the same number of block arguments ("
                         << blockArgs.size() << ") as the sum of kept qubits ("
                         << getKeptQubits().size() << ") and consumed qubits ("
                         << getConsumedQubits().size() << ")";
  }
  DenseSet<Value> arguments;
  for (auto [index, values] :
       llvm::enumerate(llvm::zip_equal(blockArgs, getOperands()))) {
    auto [arg, argValue] = values;
    Type expectedType = getAliasType(argValue.getType());
    arguments.insert(argValue);

    if (arg.getType() != expectedType) {
      return emitOpError() << "type of block argument " << index << " ("
                           << arg.getType()
                           << ") does not match the expected alias type ("
                           << expectedType << ")";
    }
  }
  if (arguments.size() != blockArgs.size()) {
    return emitOpError() << "has duplicate operands";
  }
  auto zeros = getConsumedZerosAttr();
  if (zeros &&
      static_cast<size_t>(zeros.size()) != getConsumedQubits().size()) {
    return emitOpError() << "must have one zero flag per consumed operand";
  }
  return success();
}

static bool hasQuantumType(TypeRange types) {
  return llvm::any_of(types, [](Type type) -> bool {
    return isa<qc::QubitType, qco::QubitType>(getElementTypeOrSelf(type));
  });
}

LogicalResult OracleOp::verifyRegions() {
  auto* terminator = getRegion().front().getTerminator();
  if (!isa<YieldOp>(terminator)) {
    return emitOpError() << "must have a qoracle.yield operation as the "
                            "terminator of its region";
  }
  auto result = getRegion().front().walk([&](Operation* op) {
    if (hasQuantumType(op->getOperandTypes()) ||
        hasQuantumType(op->getResultTypes())) {
      return WalkResult::interrupt();
    }
    if (!isMemoryEffectFree(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    return emitOpError()
           << "must not use qubit operands or have side effects in its region";
  }

  return success();
}

IntegerType OracleOp::getAliasType(Type operandType) {
  if (isa<qc::QubitType>(operandType)) {
    return IntegerType::get(operandType.getContext(), 1);
  }
  if (auto memref = dyn_cast<MemRefType>(operandType);
      memref && memref.getRank() == 1 && memref.hasStaticShape() &&
      isa<qc::QubitType>(memref.getElementType())) {
    if (memref.getDimSize(0) > IntegerType::kMaxWidth) {
      return {};
    }
    return IntegerType::get(operandType.getContext(), memref.getDimSize(0));
  }
  return {};
}

Block::BlockArgListType OracleOp::getKeptAliases() {
  return getRegion().getArguments().take_front(getKeptQubits().size());
}

Block::BlockArgListType OracleOp::getConsumedAliases() {
  return getRegion().getArguments().drop_front(getKeptQubits().size());
}

bool OracleOp::ownsValue(Value value) {
  if (isConsumedAlias(value)) {
    return true;
  }
  if (isKeptAlias(value)) {
    return false;
  }
  if (getRegion().isAncestor(value.getParentRegion())) {
    return true;
  }
  return false;
}

bool OracleOp::isKeptAlias(Value value) {
  return llvm::is_contained(getKeptAliases(), value);
}

bool OracleOp::isConsumedAlias(Value value) {
  return llvm::is_contained(getConsumedAliases(), value);
}

bool OracleOp::isZeroAlias(Value value) {
  auto zeros = getConsumedZerosAttr();
  if (!zeros) {
    return false;
  }
  auto aliases = getConsumedAliases();
  const auto* it = llvm::find(aliases, value);
  return it != aliases.end() && zeros[it - aliases.begin()];
}
