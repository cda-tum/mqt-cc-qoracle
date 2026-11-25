# Create an MQT Core extension repository

This guide describes how this repository was split from MQT Core and how to do
the same for another compiler extension. QOracle is the worked example.

An extension repository holds work that belongs in the MQT Core compiler but
that the MQT Core maintainers do not want to keep up to date. It pins MQT Core
as a git submodule, builds the new sources out of tree, and carries a small
patch that registers the extension in core. This is the usual layout for MLIR
projects that add dialects downstream of another project.

## When the layout fits

The layout works when the extension is mostly new files in its own folders and
needs only a few lines in existing core files. Check this on the feature branch
before you start:

```bash
base=$(git merge-base HEAD origin/main)
git diff --stat "$base" HEAD
git diff --name-status "$base" HEAD | grep -v '^A'
```

The second command lists the core files that the branch changes. For QOracle,
the branch added 29 files and changed 13 core files by 1 to 3 lines each:
`add_subdirectory` lines, link lists, dialect and pass registration, and the
glossary. If the branch changes core logic, upstream that change first or keep
it in the patch, knowing that each line there can conflict on every update.

## Steps

### 1. Sort the changed core files

Split the changed core files into two groups:

- **Registration in core targets.** These lines must stay in core and go into
  the patch: the dialect in `createCompilerContext` (`mlir/lib/Compiler/`),
  passes in `registerMQTCompilerPasses` (`mlir/lib/Support/Passes.cpp`),
  conversions in the `compile` driver (`mlir/tools/compile/Compile.cpp`), and
  the link lines that these need.
- **Everything else.** `add_subdirectory` lines, tests, and glossary entries
  move into the extension repository and leave core unchanged.

### 2. Create the repository with the extension's history

`git filter-repo` keeps the commits that touch the new files and drops all
other changes:

```bash
git diff --name-status "$base" HEAD | awk '$1=="A"{print $2}' > new-files.txt
git clone --no-local --single-branch --branch <feature-branch> <core-checkout> extract
git -C extract filter-repo --paths-from-file "$PWD/new-files.txt"
git clone extract <extension-repo>
git -C <extension-repo> branch -m <feature-branch> main
git -C <extension-repo> remote remove origin
```

Keep the file paths as they are in core, for example
`mlir/lib/Dialect/QOracle/`. The extension's own `CMakeLists.txt` files then
work unchanged.

### 3. Pin MQT Core as a submodule

```bash
git submodule add --name mqt-core https://github.com/munich-quantum-toolkit/core.git external/mqt-core
git -C external/mqt-core checkout <commit>
git config -f .gitmodules submodule.mqt-core.ignore dirty
```

`ignore = dirty` hides the applied patch from `git status` in the extension
repository. Pin a recent `main` commit. Pinning the branch's old merge base
only delays the work of moving to `main`.

### 4. Write the patch

Make the registration edits in `external/mqt-core`, then save them:

```bash
git -C external/mqt-core diff > patches/0001-register-<name>-in-compiler.patch
```

Put a short description above the first `diff --git` line. `git apply` skips
that text. Copy `scripts/apply-patches.sh` from this repository unchanged.

Core targets in the patch can name extension targets, such as
`MLIRQOracleDialect`, before the extension defines them. CMake resolves the
names when it generates the build. The extension libraries pass their include
folders on through `FILE_SET HEADERS`, so core sources that include extension
headers compile without extra include paths.

### 5. Write the top-level build files

Copy `CMakeLists.txt` and `mlir/CMakeLists.txt` from this repository and change
the names and folders. The root file does these things, in order:

1. Sets `cmake_minimum_required` to the range that core uses.
2. Stops configuration if a patch is not applied. Otherwise core builds a
   compiler that silently lacks the extension.
3. Finds LLVM/MLIR before it adds core. Core's `cmake/SetupMLIR.cmake` then
   reuses that SDK. Because of this, the extension does not download the SDK
   and needs `MLIR_DIR`.
4. Sets `BUILD_MQT_CORE_TESTS` and both bundled QDMI device options when the
   extension's tests are enabled. The tests then reuse core's GoogleTest setup
   and `mqt_mlir_configure_unittest_target`.
5. Adds core with `EXCLUDE_FROM_ALL`, so the default target builds only the
   `mqt-core` executable and the extension.
6. Repeats the MLIR setup that core applies to its own folder: the
   `TableGen`, `AddLLVM`, `AddMLIR`, and `HandleLLVMOptions` modules, include
   folders for core's and the extension's source and generated headers,
   `LLVM_DEFINITIONS`, and the `MQT_MLIR_SOURCE_INCLUDE_DIR` and
   `MQT_MLIR_BUILD_INCLUDE_DIR` variables. CMake variables set in core's folder
   do not reach the parent folder, while functions do.

In `mlir/CMakeLists.txt`, set `MQT_MLIR_UNITTESTS_DIR` before adding the test
folders, because `mqt_mlir_configure_unittest_target` reads it from the
calling folder. Give the extension's tests their own `LABELS` value, so that
`ctest -L <label>` runs them without core's tests.

### 6. Add the remaining files

- `README.md` with the build and update steps.
- `docs/glossary.md` with the glossary entries that the branch added to core.
- `LICENSE`. The moved sources carry MIT headers.
- `AGENTS.md` that defers to core's guide and adds the extension rules.

### 7. Verify

Configure and build from a clean folder, then check four things:

1. The extension's tests pass: `ctest --test-dir build -L <label>`.
2. The compiler accepts the extension. `mqt-core compile --help` does not list
   passes, so run them:
   `mqt-core compile --run-pipeline --pass-pipeline='builtin.module(<passes>)' <input.mlir>`.
3. Configuration fails after you reverse the patch with
   `git -C external/mqt-core apply --reverse <patch>`.
4. `scripts/apply-patches.sh` applies the patch and skips it on a second run.

## Known problems

- **CMake version.** MQT Core needs CMake 3.29 or newer. If the system CMake is
  older, use `uvx cmake`.
- **GCC 13 and link-time optimization.** Release builds enable link-time
  optimization by default. With GCC 13, linking the `mqt-core` executable
  crashes the compiler in `lto-partition.cc`. Configure with
  `-DENABLE_IPO=OFF` on such machines.
- **A patch that no longer applies.** `git apply` changes nothing if one part
  fails. Port the edits by hand and save the patch again. This happened when
  `mqt-cc` became the `compile` subcommand of the `mqt-core` executable
  (munich-quantum-toolkit/core#2925).

To move an existing extension to a newer MQT Core commit, follow
[Update MQT Core](../README.md#update-mqt-core).
