#!/usr/bin/env bash
# Makes the LV2 bundle, target/bundled/CA-72.lv2, from the port table (crates/ca72-lv2): the
# cdylib renamed ca72.so beside the manifest.ttl and dsp.ttl the ca72-ttl bin writes from
# `PORTS`, so the plug-in and its description cannot drift.
#
#   scripts/lv2.sh                               for this machine (Linux)
#   scripts/lv2.sh aarch64-unknown-linux-gnu     for a Raspberry Pi 4 (Zynthian)
#
# The bundle is the whole plug-in: copy CA-72.lv2 into an LV2 directory the host searches
# (/usr/lib/lv2 or /usr/local/lib/lv2 on a Zynthian, ~/.lv2 for a user), and it is found.
# Cross-compiling needs the aarch64 C toolchain (gcc-aarch64-linux-gnu) and
# `rustup target add aarch64-unknown-linux-gnu`.
set -euo pipefail

case "$(uname -s)" in
Linux) ;;
*) echo "lv2.sh: the LV2 bundle is built for Linux (this is $(uname -s))" >&2; exit 1 ;;
esac

root="$(cd "$(dirname "$0")/.." && pwd)"
target="${1:-}"
[ "$#" -le 1 ] || { echo "lv2.sh: usage: lv2.sh [target-triple]" >&2; exit 1; }

libdir="$root/target/release"
build=(build -p ca72-lv2 --lib --release)
if [ -n "$target" ]; then
    build+=(--target "$target")
    libdir="$root/target/$target/release"
    case "$target" in
    aarch64-unknown-linux-gnu)
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER:-aarch64-linux-gnu-gcc}"
        ;;
    esac
fi

(cd "$root" && cargo "${build[@]}")

bundle="$root/target/bundled/CA-72.lv2"
rm -rf "$bundle"
mkdir -p "$bundle"
# The TTL generator is host-independent: it always runs on this machine, whatever the target.
(cd "$root" && cargo run -q -p ca72-lv2 --bin ca72-ttl -- "$bundle")
cp "$libdir/libca72_lv2.so" "$bundle/ca72.so"

echo "$bundle"
