/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QOracle/IR/QOracleOps.h"

#include "mqt/Dialect/QOracle/IR/QOracleDialect.h" // IWYU pragma: associated

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep (template instantiations)
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep (template instantiations)

using namespace mlir;
using namespace mlir::qoracle;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "mqt/Dialect/QOracle/IR/QOracleOpsDialect.cpp.inc"

void QOracleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mqt/Dialect/QOracle/IR/QOracleOps.cpp.inc"

      >();
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

[[nodiscard]] static ParseResult
parseOracleAliasGroup(OpAsmParser& parser,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand>& operands,
                      SmallVectorImpl<Type>& types,
                      SmallVectorImpl<OpAsmParser::Argument>& aliases) {
  return parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::UnresolvedOperand operand;
        if (failed(parser.parseOperand(operand)) ||
            failed(parser.parseColon())) {
          return failure();
        }
        const auto typeLoc = parser.getCurrentLocation();
        Type type;
        OpAsmParser::Argument alias;
        if (failed(parser.parseType(type)) ||
            failed(parser.parseKeyword("as")) ||
            failed(parser.parseArgument(alias))) {
          return failure();
        }
        alias.type = OracleOp::getAliasType(type);
        if (!alias.type) {
          return parser.emitError(typeLoc, "expected a qubit or a static 1-D "
                                           "memref of qubits, got ")
                 << type;
        }
        operands.push_back(operand);
        types.push_back(type);
        aliases.push_back(alias);
        return success();
      });
}

/// Parse an oracle's aliased qubit operands.
[[nodiscard]] static ParseResult parseOracleAliasing(
    OpAsmParser& parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& keptQubits,
    SmallVectorImpl<Type>& keptTypes,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand>& consumedQubits,
    SmallVectorImpl<Type>& consumedTypes, Region& body) {
  SmallVector<OpAsmParser::Argument> aliases;

  if (succeeded(parser.parseOptionalKeyword("kept"))) {
    if (failed(parseOracleAliasGroup(parser, keptQubits, keptTypes, aliases))) {
      return failure();
    }
  }
  if (succeeded(parser.parseOptionalKeyword("consumed"))) {
    if (failed(parseOracleAliasGroup(parser, consumedQubits, consumedTypes,
                                     aliases))) {
      return failure();
    }
  }
  return parser.parseRegion(body, aliases, /*enableNameShadowing=*/false);
}

/// Print one `kept(...)` or `consumed(...)` group, or nothing if it is empty.
static void printOracleAliasGroup(OpAsmPrinter& printer, StringRef keyword,
                                  OperandRange operands, TypeRange types,
                                  Block::BlockArgListType aliases) {
  if (operands.empty()) {
    return;
  }
  printer << keyword << '(';
  llvm::interleaveComma(llvm::zip_equal(operands, types, aliases), printer,
                        [&](auto entry) {
                          auto [operand, type, alias] = entry;
                          printer << operand << " : " << type << " as ";
                          printer.printOperand(alias);
                        });
  printer << ") ";
}

static void printOracleAliasing(OpAsmPrinter& printer, Operation* op,
                                OperandRange keptQubits, TypeRange keptTypes,
                                OperandRange consumedQubits,
                                TypeRange consumedTypes, Region& body) {
  auto oracle = cast<OracleOp>(op);
  printOracleAliasGroup(printer, "kept", keptQubits, keptTypes,
                        oracle.getKeptAliases());
  printOracleAliasGroup(printer, "consumed", consumedQubits, consumedTypes,
                        oracle.getConsumedAliases());
  printer.printRegion(body, /*printEntryBlockArgs=*/false);
}

#define GET_OP_CLASSES
#include "mqt/Dialect/QOracle/IR/QOracleOps.cpp.inc"
