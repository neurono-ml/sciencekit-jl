#!/usr/bin/env bash
# Copy the authoritative `sciencekit.h` C ABI header from a checkout of
# `neurono-ml/sciencekit` into `include/sciencekit.h` and verify there is no
# drift when `--check` is requested.
#
# Usage:
#   scripts/sync-header.sh [--check] [path-to-sciencekit-checkout]
set -euo pipefail

check_only=0
if [[ "${1:-}" == "--check" ]]; then
  check_only=1
  shift
fi

binding_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_root="${1:-${SCIENCEKIT_REPOSITORY:-}}"

if [[ -z "$repository_root" ]]; then
  candidate="$binding_root/../sciencekit"
  if [[ -f "$candidate/Cargo.toml" ]]; then
    repository_root="$(cd "$candidate" && pwd)"
  fi
fi

if [[ -z "$repository_root" || ! -f "$repository_root/crates/sciencekit_ffi/include/sciencekit.h" ]]; then
  echo "error: sciencekit checkout with crates/sciencekit_ffi/include/sciencekit.h not found" >&2
  exit 2
fi

source_header="$repository_root/crates/sciencekit_ffi/include/sciencekit.h"
target_header="$binding_root/include/sciencekit.h"

mkdir -p "$binding_root/include"

if [[ "$check_only" -eq 1 ]]; then
  diff -u "$target_header" "$source_header"
  echo "include/sciencekit.h matches $source_header"
else
  cp "$source_header" "$target_header"
  echo "Synced $target_header from $source_header"
fi
