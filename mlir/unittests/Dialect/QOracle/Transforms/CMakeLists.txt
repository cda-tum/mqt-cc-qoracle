# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qoracle_transforms_target mqt-core-mlir-unittest-qoracle-transforms)
add_executable(${qoracle_transforms_target} test_qoracle_transforms.cpp)
target_link_libraries(
  ${qoracle_transforms_target}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRFuncDialect
          MLIRIR
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRQCDialect
          MLIRQOracleDialect
          MLIRQOracleTransforms
          MLIRSCFDialect
          MLIRSupport)
mqt_mlir_configure_unittest_target(${qoracle_transforms_target})

gtest_discover_tests(${qoracle_transforms_target} PROPERTIES LABELS mqt-mlir-unittests
                                                             DISCOVERY_TIMEOUT 60)
