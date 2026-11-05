#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
# SPDX-License-Identifier: Apache-2.0
set -o errexit
set -o nounset
set -o pipefail

formal_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
kube_root="$(git -C "$formal_root" rev-parse --show-toplevel)"
if [[ ! -f "$kube_root/hack/lib/init.sh" ]]; then
  printf 'The HTTP regression requires a Kubernetes checkout; run ci/verify.sh for the standalone apimachinery proofs.\n' >&2
  exit 1
fi
artifacts="${ARTIFACTS:-${TMPDIR:-/tmp}/ssa-apiserver-artifacts-$$}"
mkdir -p "$artifacts"
if [[ "${SSA_APISERVER_LOGGED:-0}" != 1 ]]; then
  SSA_APISERVER_LOGGED=1 ARTIFACTS="$artifacts" bash "$formal_root/ci/verify-apiserver.sh" 2>&1 |
    tee "$artifacts/verify-ssa-apiserver.log"
  exit
fi

# Use the checkout's normal etcd version, installation and lifecycle helpers.
cd "$kube_root"
export PATH="$kube_root/third_party/etcd:$PATH"
source "$kube_root/hack/lib/init.sh"
if ! command -v etcd >/dev/null; then
  bash "$kube_root/hack/install-etcd.sh"
fi
trap kube::etcd::cleanup EXIT
kube::etcd::start
go test ./staging/src/k8s.io/apiextensions-apiserver/test/integration \
  -run '^TestSSAEmpty(AssociativeList|GranularMap)Presence$' -v -count=1 -timeout=240s
