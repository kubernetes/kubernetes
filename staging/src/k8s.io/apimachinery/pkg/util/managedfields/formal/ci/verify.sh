#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
artifacts="${ARTIFACTS:-${TMPDIR:-/tmp}/ssa-formal-artifacts-$$}"
mkdir -p "$artifacts"
# A separate shell preserves errexit inside bootstrap while the parent tees its output.
if [[ "${SSA_FORMAL_LOGGED:-0}" != 1 ]]; then
  SSA_FORMAL_LOGGED=1 ARTIFACTS="$artifacts" bash "$formal_root/ci/verify.sh" 2>&1 |
    tee "$artifacts/verify-ssa-formal.log"
  exit
fi

if [[ -z "${SSA_LAKE:-}" ]]; then
  version="$(sed -n 's/.*:v//p' "$formal_root/lean-toolchain")"
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64) platform=linux ;;
    Linux/aarch64 | Linux/arm64) platform=linux_aarch64 ;;
    Darwin/x86_64) platform=darwin ;;
    Darwin/arm64) platform=darwin_aarch64 ;;
    *) printf 'Unsupported Lean platform; set SSA_LAKE to a pinned installation.\n' >&2; exit 1 ;;
  esac
  cache="${SSA_TOOLCHAIN_CACHE:-${XDG_CACHE_HOME:-${TMPDIR:-/tmp}}/ssa-formal}"
  installation="$cache/lean-$version-$platform"
  if [[ ! -x "$installation/bin/lake" ]]; then
    archive="lean-$version-$platform.tar.zst"
    checksum="$(awk -v name="$archive" '$2 == name {print $1}' "$formal_root/ci/lean.sha256")"
    if [[ "${#checksum}" != 64 ]]; then
      printf 'No pinned SHA-256 for %s; update ci/lean.sha256 with the toolchain.\n' "$archive" >&2
      exit 1
    fi
    mkdir -p "$cache"
    download="$(mktemp -d "$cache/download.XXXXXX")"
    trap 'rm -rf "$download"' EXIT
    curl --fail --location --retry 3 \
      "https://github.com/leanprover/lean4/releases/download/v$version/$archive" \
      --output "$download/$archive"
    (
      cd "$download"
      if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check -
      else
        printf '%s  %s\n' "$checksum" "$archive" | shasum -a 256 --check -
      fi
      tar -xf "$archive"
    )
    mv "$download/lean-$version-$platform" "$installation"
  fi
  export SSA_LAKE="$installation/bin/lake"
fi

printf 'SSA verification artifacts: %s\n' "$artifacts"
bash "$formal_root/verify.sh"
