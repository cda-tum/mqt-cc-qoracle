# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(qoracle_to_qc_target mqt-core-mlir-unittest-qoracle-to-qc)
add_executable(${qoracle_to_qc_target} test_qoracle_to_qc.cpp)
target_link_libraries(
  ${qoracle_to_qc_target}
  PRIVATE GTest::gtest_main
          MLIRArithDialect
          MLIRFuncDialect
          MLIRIR
          MLIRMemRefDialect
          MLIRParser
          MLIRPass
          MLIRQCDialect
          MLIRQOracleDialect
          MLIRQOracleToQC
          MLIRSupport)
mqt_mlir_configure_unittest_target(${qoracle_to_qc_target})

gtest_discover_tests(${qoracle_to_qc_target} PROPERTIES LABELS mqt-mlir-unittests DISCOVERY_TIMEOUT
                                                        60)
