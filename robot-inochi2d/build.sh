#!/bin/sh
# [robot fork] Builds the Inochi2D renderer (this crate) to WebAssembly, inside Docker (nothing to
# install on the host), into ../src/open_llm_vtuber/robot/web/inochi2d/. Run it again after a
# change to this crate; the VTuber image then picks up the built files with its sources.
#   ./build.sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
out="$here/../src/open_llm_vtuber/robot/web/inochi2d"
mkdir -p "$out" "$here/target"
docker run --rm --user "$(id -u):$(id -g)" \
  -e CARGO_HOME=/src/target/cargo-home -e CARGO_TARGET_DIR=/src/target \
  -v "$here":/src -v "$out":/out -w /src rust:1-bookworm sh -euc '
    rustup target add wasm32-unknown-unknown
    cargo build --release --target wasm32-unknown-unknown
    # wasm-bindgen (the tool) must be the version of the wasm-bindgen crate in Cargo.lock
    version=$(sed -n "/^name = \"wasm-bindgen\"$/{n;s/^version = \"\(.*\)\"$/\1/p}" Cargo.lock)
    bin=/src/target/tools/bin/wasm-bindgen
    if ! [ -x "$bin" ] || ! "$bin" --version | grep -q " $version\$"; then
      cargo install --root /src/target/tools wasm-bindgen-cli --version "$version" --locked
    fi
    "$bin" --target web --no-typescript --out-dir /out --out-name inochi2d \
      /src/target/wasm32-unknown-unknown/release/robot_inochi2d.wasm
  '
ls -l "$out"
