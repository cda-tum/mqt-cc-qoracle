# MQT Core QOracle

The QOracle dialect, its passes, and its lowering to QC, built into the MQT Core
compiler (`mqt-core compile`). This repository holds only the QOracle code. It
pins [MQT Core](https://github.com/munich-quantum-toolkit/core) as a git
submodule and carries a small patch that registers QOracle in the compiler.

## Layout

- `external/mqt-core/`: the pinned MQT Core commit.
- `patches/`: changes to MQT Core, applied to the submodule's working tree.
  They only register QOracle in core targets. All QOracle code lives here.
- `mlir/`: QOracle sources and unit tests, in the same layout as MQT Core's
  `mlir/` tree.
- `docs/glossary.md`: terms that QOracle adds to the MQT Core glossary.

## Build

The build needs LLVM/MLIR 23.1. Unlike a standalone MQT Core build, it does not
download the SDK, so pass `MLIR_DIR`.

```bash
git clone --recurse-submodules https://github.com/cda-tum/mqt-cc-qoracle.git
cd mqt-cc-qoracle
scripts/apply-patches.sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=<llvm-prefix>/lib/cmake/mlir -DLLVM_DIR=<llvm-prefix>/lib/cmake/llvm
cmake --build build
ctest --test-dir build -L qoracle
```

The default target builds the `mqt-core` executable and the QOracle libraries
and tests. Other MQT Core targets build only on request, for example
`cmake --build build --target mqt-core-mlir-unittests-compiler`. `ctest`
without `-L qoracle` also lists those core tests and reports them as not built.

The build needs CMake 3.29 or newer, as MQT Core does. If the system CMake is
older, run `uvx cmake` instead.

Configuration fails if a patch is not applied, so the compiler cannot silently
build without QOracle.

## Update MQT Core

```bash
git -C external/mqt-core checkout -- .
git -C external/mqt-core fetch origin
git -C external/mqt-core checkout <new-commit>
scripts/apply-patches.sh
```

If a patch no longer applies, apply it by hand in `external/mqt-core`, fix the
conflicts, and regenerate it with `git -C external/mqt-core diff`. Keep the
description at the top of the patch file. Then rebuild and fix any QOracle code
that the core API changes broke. Commit the new submodule commit together with
those fixes.

`.gitmodules` sets `ignore = dirty`, so the applied patches do not show up in
`git status`. Use `git -C external/mqt-core status` to see them.
