/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/MQT/IR/QubitLayout.h"
#include "mqt/Dialect/QOracle/IR/QOracleOps.h"
#include "mqt/Dialect/QOracle/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace mlir::qoracle {

#define GEN_PASS_DEF_REASSOCIATEORACLE
#include "mqt/Dialect/QOracle/Transforms/Passes.h.inc"

namespace {

struct Sum {
  arith::AddIOp root;
  llvm::SmallVector<arith::AddIOp> additions;
  llvm::SmallVector<Value> terms;
};

bool isOwned(Value value, OracleOp oracle) {
  if (llvm::find(oracle.getConsumedAliases(), value) !=
      oracle.getConsumedAliases().end()) {
    return true;
  }
  if (llvm::find(oracle.getKeptAliases(), value) !=
      oracle.getKeptAliases().end()) {
    return false;
  }
  if (oracle.getRegion().isAncestor(value.getParentRegion())) {
    return true;
  }
  return false;
}

bool dependsOn(Operation* op, Operation* root) {
  auto* block = root->getBlock();
  llvm::SmallVector<Operation*> toVisit{op};
  llvm::SmallPtrSet<Operation*, 16> visited{op};
  while (!toVisit.empty()) {
    auto* current = toVisit.pop_back_val();
    if (current == root) {
      return true;
    }
    current->walk([&](Operation* nested) {
      for (auto operand : nested->getOperands()) {
        auto* definingOp = operand.getDefiningOp();
        if (definingOp == nullptr) {
          continue;
        }
        // Operations outside the block or before `root` cannot read its
        // result.
        auto* ancestor = block->findAncestorOpInBlock(*definingOp);
        if (ancestor == nullptr || ancestor->isBeforeInBlock(root)) {
          continue;
        }
        if (visited.insert(ancestor).second) {
          toVisit.push_back(ancestor);
        }
      }
    });
  }
  return false;
}

bool canBeLastReader(Sum& sum, Value value) {
  auto* block = sum.root->getBlock();
  for (auto* user : value.getUsers()) {
    auto* ancestor = block->findAncestorOpInBlock(*user);
    if (ancestor == nullptr) {
      return false;
    }
    if (llvm::is_contained(sum.additions, ancestor)) {
      continue;
    }
    if (dependsOn(ancestor, sum.root)) {
      return false;
    }
  }
  return true;
}

bool isSameChain(ArrayRef<Value> current, ArrayRef<Value> desired) {
  if (current.size() != desired.size()) {
    return false;
  }
  if (current == desired) {
    return true;
  }
  return current.size() >= 2 && current[0] == desired[1] &&
         current[1] == desired[0] &&
         current.drop_front(2) == desired.drop_front(2);
}

struct ReassociateOraclePattern final : OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp addOp,
                                PatternRewriter& rewriter) const override {

    if (addOp->hasOneUse() && isa<arith::AddIOp>(*addOp->getUsers().begin()) &&
        addOp->getUsers().begin()->getBlock() == addOp->getBlock()) {
      return failure();
    }

    Sum sum = {.root = addOp, .additions = {addOp}};
    llvm::SmallVector<Value> terms;
    llvm::SmallVector<Value> toVisit{addOp.getRhs(), addOp.getLhs()};

    while (!toVisit.empty()) {
      auto value = toVisit.pop_back_val();
      auto nestedAdd = value.getDefiningOp<arith::AddIOp>();
      if (nestedAdd && nestedAdd->getBlock() == addOp->getBlock() &&
          value.hasOneUse()) {
        sum.additions.push_back(nestedAdd);
        toVisit.push_back(nestedAdd.getRhs());
        toVisit.push_back(nestedAdd.getLhs());
      } else {
        terms.push_back(value);
      }
    }

    auto constantTerms = llvm::SmallVector<Operation*>();
    for (auto term : terms) {
      if (auto constant = term.getDefiningOp<arith::ConstantOp>()) {
        constantTerms.push_back(constant);
      } else {
        sum.terms.push_back(term);
      }
    }

    llvm::SmallVector<Value> reorderedTerms;
    bool foundHost = false;
    for (auto term : sum.terms) {
      if (!foundHost && isOwned(term, addOp->getParentOfType<OracleOp>()) &&
          canBeLastReader(sum, term)) {
        reorderedTerms.insert(reorderedTerms.begin(), term);
        foundHost = true;
      } else {
        reorderedTerms.push_back(term);
      }
    }

    if (constantTerms.size() >= 2) {
      auto totalConstant = std::accumulate(
          constantTerms.begin(), constantTerms.end(),
          APInt(addOp.getType().getIntOrFloatBitWidth(), 0),
          [](APInt a, Operation* b) -> APInt {
            return std::move(a) +
                   cast<IntegerAttr>(cast<arith::ConstantOp>(b).getValue())
                       .getValue();
          });
      if (!totalConstant.isZero()) {
        auto constantOp = arith::ConstantOp::create(
            rewriter, addOp.getLoc(), addOp.getType(),
            IntegerAttr::get(addOp.getType(), totalConstant));
        sum.terms.push_back(constantOp);
        reorderedTerms.push_back(constantOp);
      }
    } else if (constantTerms.size() == 1) {
      sum.terms.push_back(constantTerms.front()->getResult(0));
      reorderedTerms.push_back(constantTerms.front()->getResult(0));
    }

    auto isLeftDeep = llvm::none_of(sum.additions, [&](arith::AddIOp addition) {
      return llvm::is_contained(sum.additions,
                                addition.getRhs().getDefiningOp());
    });
    if (isLeftDeep && isSameChain(terms, reorderedTerms)) {
      return failure();
    }

    auto acc = reorderedTerms.empty()
                   ? arith::ConstantOp::create(
                         rewriter, addOp.getLoc(), addOp.getType(),
                         IntegerAttr::get(addOp.getType(), 0))
                   : reorderedTerms[0];
    for (auto term : llvm::drop_begin(reorderedTerms, 1)) {
      acc = arith::AddIOp::create(rewriter, addOp.getLoc(), acc, term);
    }
    rewriter.replaceOp(addOp, acc);
    return success();
  }
};

struct ReassociateOracle final
    : impl::ReassociateOracleBase<ReassociateOracle> {
protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReassociateOraclePattern>(&getContext());
    const FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Region simplification could change the oracle body beyond the sums,
    // so only the pattern and the driver's folding and dead-code removal run.
    auto config =
        GreedyRewriteConfig{}
            .setUseTopDownTraversal()
            .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);

    // Collect first: rewriting the regions would invalidate a running walk.
    SmallVector<OracleOp> oracles;
    getOperation()->walk([&](OracleOp oracle) { oracles.push_back(oracle); });

    // Scoping the driver to each oracle region leaves the rest of the
    // program unchanged.
    for (auto oracle : oracles) {
      if (failed(applyPatternsGreedily(oracle.getRegion(), frozenPatterns,
                                       config))) {
        oracle.emitOpError(
            "re-association of additions in the region did not converge");
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::qoracle
