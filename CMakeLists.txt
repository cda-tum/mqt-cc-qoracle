# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

cmake_minimum_required(VERSION 3.29...4.5)

project(
  mqt-core-qoracle
  LANGUAGES C CXX
  DESCRIPTION "QOracle dialect and lowering for the MQT Core compiler")

option(BUILD_MQT_CORE_QORACLE_TESTS "Build the QOracle unit tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_ALL_MQT_CORE_TARGETS "Build all MQT Core targets, including enabled core tests" OFF)

set(MQT_CORE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external/mqt-core")
set(MQT_CORE_BINARY_DIR "${PROJECT_BINARY_DIR}/external/mqt-core")

if(NOT EXISTS "${MQT_CORE_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "external/mqt-core is empty. Run git submodule update --init.")
endif()

# The patches register QOracle in the compiler. Apply them here so that every build, CI included,
# uses the patched core. Patches that are already applied are skipped.
find_package(Git REQUIRED)
file(
  GLOB MQT_CORE_QORACLE_PATCHES
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/patches/*.patch")
foreach(patch IN LISTS MQT_CORE_QORACLE_PATCHES)
  cmake_path(GET patch FILENAME patch_name)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${patch}"
    WORKING_DIRECTORY "${MQT_CORE_SOURCE_DIR}"
    RESULT_VARIABLE patch_missing
    OUTPUT_QUIET ERROR_QUIET)
  if(patch_missing)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" apply "${patch}"
      WORKING_DIRECTORY "${MQT_CORE_SOURCE_DIR}"
      RESULT_VARIABLE patch_failed
      ERROR_VARIABLE patch_error)
    if(patch_failed)
      message(FATAL_ERROR "${patch_name} does not apply to external/mqt-core:\n${patch_error}")
    endif()
    message(STATUS "Applied ${patch_name} to external/mqt-core")
  endif()
endforeach()

# Find the SDK before adding mqt-core so that core reuses it instead of selecting its own.
find_package(MLIR 23.1 CONFIG REQUIRED)

if(BUILD_MQT_CORE_QORACLE_TESTS)
  # Reuse the GoogleTest setup and MLIR unit-test helpers of mqt-core. Core tests require its
  # bundled QDMI devices.
  set(BUILD_MQT_CORE_TESTS ON)
  set(BUILD_MQT_CORE_QDMI_DDSIM_DEVICE ON)
  set(BUILD_MQT_CORE_QDMI_SC_DEVICE ON)
  enable_testing()
  include(GoogleTest)
endif()

if(BUILD_ALL_MQT_CORE_TARGETS)
  add_subdirectory("${MQT_CORE_SOURCE_DIR}" "${MQT_CORE_BINARY_DIR}")
else()
  # Only targets that the default target needs are built from mqt-core.
  add_subdirectory("${MQT_CORE_SOURCE_DIR}" "${MQT_CORE_BINARY_DIR}" EXCLUDE_FROM_ALL)
endif()

# Match the MLIR setup that mqt-core applies to its own directory scope.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(MQT_MLIR_SOURCE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/mlir/include")
set(MQT_MLIR_BUILD_INCLUDE_DIR "${PROJECT_BINARY_DIR}/mlir/include")
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories("${MQT_CORE_SOURCE_DIR}/mlir/include" "${MQT_CORE_BINARY_DIR}/mlir/include"
                    "${MQT_MLIR_SOURCE_INCLUDE_DIR}" "${MQT_MLIR_BUILD_INCLUDE_DIR}")
separate_arguments(MQT_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${MQT_LLVM_DEFINITIONS})
set(MLIR_BINARY_DIR "${PROJECT_BINARY_DIR}")

add_subdirectory(mlir)

add_custom_target(mqt-core-qoracle ALL)
add_dependencies(mqt-core-qoracle mqt-core)
