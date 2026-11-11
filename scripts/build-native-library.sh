#!/usr/bin/env bash
# Build the `sciencekit_ffi` shared library from a checkout of
# `neurono-ml/sciencekit` and install it into `lib/` so `ScienceKit.__init__()`
# discovers it automatically.
#
# Usage:
#   scripts/build-native-library.sh [path-to-sciencekit-checkout]
#
# The checkout defaults to $SCIENCEKIT_REPOSITORY, then to a sibling
# `../sciencekit` directory.
set -euo pipefail

binding_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_root="${1:-${SCIENCEKIT_REPOSITORY:-}}"

if [[ -z "$repository_root" ]]; then
  candidate="$binding_root/../sciencekit"
  if [[ -f "$candidate/Cargo.toml" ]]; then
    repository_root="$(cd "$candidate" && pwd)"
  fi
fi

if [[ -z "$repository_root" || ! -f "$repository_root/Cargo.toml" ]]; then
  cat >&2 <<'USAGE'
error: no sciencekit checkout found.

Pass the path to a checkout of https://github.com/neurono-ml/sciencekit:

  scripts/build-native-library.sh /path/to/sciencekit

or set SCIENCEKIT_REPOSITORY to the same path.
USAGE
  exit 2
fi

cd "$repository_root"

echo "Building sciencekit_ffi from $repository_root"
cargo build -p sciencekit_ffi --release

target_directory="${CARGO_TARGET_DIR:-$repository_root/target}"
library_directory="$target_directory/release"

case "$(uname -s)" in
  Darwin) library_name="libsciencekit_ffi.dylib" ;;
  MINGW* | MSYS* | CYGWIN*) library_name="sciencekit_ffi.dll" ;;
  *) library_name="libsciencekit_ffi.so" ;;
esac

mkdir -p "$binding_root/lib"
install -m 0755 "$library_directory/$library_name" "$binding_root/lib/$library_name"

echo "Installed $binding_root/lib/$library_name"
