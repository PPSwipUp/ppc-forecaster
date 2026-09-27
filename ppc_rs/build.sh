#!/usr/bin/env bash
# Local dev build: compile the Rust brain tuned for THIS machine's CPU and drop it next to the Python
# package (one abi3 file for py3.9+).  Published wheels are built portably by maturin in CI instead
# (see .github/workflows/CI.yml); don't ship this file.
set -euo pipefail
cd "$(dirname "$0")"
FLAGS="-C target-cpu=native"
[[ "$(uname)" == "Darwin" ]] && FLAGS="$FLAGS -C link-arg=-undefined -C link-arg=dynamic_lookup"
RUSTFLAGS="$FLAGS" cargo build --release
case "$(uname)" in
  Darwin) cp target/release/libppc_rs.dylib ../ppc/ppc_rs.abi3.so ;;
  Linux)  cp target/release/libppc_rs.so ../ppc/ppc_rs.abi3.so ;;
  *)      cp target/release/ppc_rs.dll ../ppc/ppc_rs.pyd ;;
esac
echo "installed ppc/ppc_rs extension"
