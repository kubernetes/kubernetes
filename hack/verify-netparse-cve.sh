#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

# This script checks if the "net" stdlib IP and CIDR parsers are used
# instead of the ones forked in k8s.io/utils/net to parse IP addresses
# because of the compatibility break introduced in golang 1.17
# Reference: #100895
# Usage: `hack/verify-netparse-cve.sh`.

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

cd "${KUBE_ROOT}"

rc=0

find_files() {
  find . -not \( \
      \( \
        -wholename './.git' \
        -o -wholename './_output' \
        -o -wholename './release' \
        -o -wholename './target' \
        -o -wholename '*/third_party/*' \
        -o -wholename '*/vendor/*' \
      \) -prune \
    \) -name '*.go'
}

# find files using net.ParseIP()
netparseip_matches=$(find_files | xargs grep -nE "net.ParseIP\(.*\)" 2>/dev/null) || true
if [[ -n "${netparseip_matches}" ]]; then
  echo "net.ParseIP reject leading zeros in the dot-decimal notation of IPv4 addresses since golang 1.17:" >&2
  echo "${netparseip_matches}" >&2
  echo >&2
  echo "Use k8s.io/utils/net ParseIPSloppy() to parse IP addresses. Kubernetes #100895" >&2
  echo >&2
  echo "Run ./hack/update-netparse-cve.sh" >&2
  echo >&2
  rc=1
fi

# find files using net.ParseCIDR()
netparsecidrs_matches=$(find_files | xargs grep -nE "net.ParseCIDR\(.*\)" 2>/dev/null) || true
if [[ -n "${netparsecidrs_matches}" ]]; then
  echo "net.ParseCIDR reject leading zeros in the dot-decimal notation of IPv4 addresses since golang 1.17:" >&2
  echo "${netparsecidrs_matches}" >&2
  echo >&2
  echo "Use k8s.io/utils/net ParseCIDRSloppy() to parse network CIDRs. Kubernetes #100895" >&2
  echo >&2
  echo "Run ./hack/update-netparse-cve.sh" >&2
  echo >&2
  rc=1
fi

exit $rc
