# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(include/mqt/Dialect/QOracle)
add_subdirectory(include/mqt/Conversion/QOracleToQC)
add_subdirectory(lib/Dialect/QOracle)
add_subdirectory(lib/Conversion/QOracleToQC)

if(BUILD_MQT_CORE_QORACLE_TESTS)
  # mqt_mlir_configure_unittest_target reads this variable from the calling scope.
  set(MQT_MLIR_UNITTESTS_DIR "${MQT_CORE_SOURCE_DIR}/mlir/unittests")
  add_subdirectory(unittests/Dialect/QOracle)
  add_subdirectory(unittests/Conversion/QOracleToQC)
endif()
