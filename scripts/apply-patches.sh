#!/usr/bin/env bash
# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Checks out the pinned mqt-core and applies patches/*.patch to its working tree.
# Patches that are already applied are skipped, so the script is safe to rerun.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core="$root/external/mqt-core"

git -C "$root" submodule update --init external/mqt-core

for patch in "$root"/patches/*.patch; do
  name="$(basename "$patch")"
  if git -C "$core" apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $name"
  elif git -C "$core" apply --check "$patch"; then
    git -C "$core" apply "$patch"
    echo "applied: $name"
  else
    echo "error: $name does not apply to external/mqt-core; update the patch" >&2
    exit 1
  fi
done
