# MQT Core QOracle Agent Guide

This repository extends the MQT Core compiler with the QOracle dialect. It pins
MQT Core in `external/mqt-core/` and patches it only to register QOracle. The
[README](README.md) describes the layout, build, and update steps.
[Create an MQT Core extension repository](docs/creating-an-extension.md)
explains how the repository was set up.

## Inherited rules

Follow `external/mqt-core/AGENTS.md` and the policies it links for code, tests,
documentation, prose, commits, and public contributions. Its paths refer to
the core tree. The same rules apply to the matching paths here, for example
`mlir/lib/Dialect/QOracle/`. Add QOracle terms to `docs/glossary.md` in this
repository, not to core's glossary.

## Extension rules

- Change core only through `patches/`. Do not commit changes inside
  `external/mqt-core/`; `ignore = dirty` hides them from `git status`, so check
  `git -C external/mqt-core status` before you finish.
- Keep the patch limited to registering QOracle in core targets. Put new code,
  tests, and build logic in this repository. After you change the patch, check
  that `scripts/apply-patches.sh` applies it to a clean submodule.
- When you move the submodule to a newer commit, update the patch and fix the
  QOracle code in the same commit as the new pin.
- Give QOracle tests the `qoracle` label.

## Build and validation

| Task                         | Command                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Apply the patch              | `scripts/apply-patches.sh`                                                                                               |
| Configure                    | `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=<llvm-prefix>/lib/cmake/mlir -DLLVM_DIR=<llvm-prefix>/lib/cmake/llvm` |
| Build                        | `cmake --build build`                                                                                                    |
| QOracle tests                | `ctest --test-dir build -L qoracle`                                                                                      |
| QOracle passes in the driver | `build/bin/mqt-core compile --run-pipeline --pass-pipeline='builtin.module(qoracle-to-qc)' <input.mlir>`                 |

Use `uvx cmake` if the system CMake is older than 3.29. With GCC 13, add
`-DENABLE_IPO=OFF`; link-time optimization crashes the compiler when it links
`mqt-core`. Keep such machine settings out of the repository.
