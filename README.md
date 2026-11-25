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
- `docs/creating-an-extension.md`: how this repository was set up, as a recipe
  for other MQT Core extensions.
- `AGENTS.md`: rules for coding agents in this repository.

## Build

The build needs LLVM/MLIR 23.1. Unlike a standalone MQT Core build, it does not
download the SDK, so pass `MLIR_DIR`.

```bash
git clone --recurse-submodules https://github.com/cda-tum/mqt-cc-qoracle.git
cd mqt-cc-qoracle
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

Configuration applies the patches to `external/mqt-core` and skips those that
are already applied. It fails if a patch does not apply, so the compiler cannot
silently build without QOracle.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull
request. It checks out the pinned MQT Core, applies the patches, builds all MQT
Core and QOracle targets, and runs all tests of both. The two jobs match the
Ubuntu jobs of MQT Core's CI: Debug on x86-64 and Release on ARM, both with GCC.

To run the same build locally, use the presets. They build all of MQT Core,
which takes much longer than the build above.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

## Update MQT Core

```bash
git -C external/mqt-core checkout -- .
git -C external/mqt-core fetch origin
git -C external/mqt-core checkout <new-commit>
cmake -S . -B build
```

If a patch no longer applies, apply it by hand in `external/mqt-core`, fix the
conflicts, and regenerate it with `git -C external/mqt-core diff`. Keep the
description at the top of the patch file. Then rebuild and fix any QOracle code
that the core API changes broke. Commit the new submodule commit together with
those fixes.

`.gitmodules` sets `ignore = dirty`, so the applied patches do not show up in
`git status`. Use `git -C external/mqt-core status` to see them.
