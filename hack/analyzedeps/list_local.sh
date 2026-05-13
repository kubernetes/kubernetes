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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Map server targets to correct Go import paths
declare -a RESOLVED_SERVER_TARGETS=()
for t in "${KUBE_SERVER_TARGETS[@]}"; do
  if [[ "$t" =~ ^staging/src/ ]]; then
    RESOLVED_SERVER_TARGETS+=("${t#staging/src/}")
  elif [[ "$t" =~ ^(cmd|cluster)/ ]]; then
    RESOLVED_SERVER_TARGETS+=("k8s.io/kubernetes/$t")
  else
    RESOLVED_SERVER_TARGETS+=("$t")
  fi
done

# Map client targets to correct Go import paths
declare -a RESOLVED_CLIENT_TARGETS=()
for t in "${KUBE_CLIENT_TARGETS[@]}"; do
  if [[ "$t" =~ ^staging/src/ ]]; then
    RESOLVED_CLIENT_TARGETS+=("${t#staging/src/}")
  elif [[ "$t" =~ ^(cmd|cluster)/ ]]; then
    RESOLVED_CLIENT_TARGETS+=("k8s.io/kubernetes/$t")
  else
    RESOLVED_CLIENT_TARGETS+=("$t")
  fi
done

# Map node targets to correct Go import paths
declare -a RESOLVED_NODE_TARGETS=()
for t in "${KUBE_NODE_TARGETS[@]}"; do
  if [[ "$t" =~ ^staging/src/ ]]; then
    RESOLVED_NODE_TARGETS+=("${t#staging/src/}")
  elif [[ "$t" =~ ^(cmd|cluster)/ ]]; then
    RESOLVED_NODE_TARGETS+=("k8s.io/kubernetes/$t")
  else
    RESOLVED_NODE_TARGETS+=("$t")
  fi
done

export GOWORK=off
# 1. Linux (Server, Client, Node) - Static & Dynamic
GOOS=linux CGO_ENABLED=0 go list -tags="selinux,notest,grpcnotrace" -deps -json "${RESOLVED_SERVER_TARGETS[@]}" "${RESOLVED_CLIENT_TARGETS[@]}" "${RESOLVED_NODE_TARGETS[@]}"
GOOS=linux CGO_ENABLED=1 go list -tags="selinux,notest,grpcnotrace" -deps -json "${RESOLVED_SERVER_TARGETS[@]}" "${RESOLVED_CLIENT_TARGETS[@]}" "${RESOLVED_NODE_TARGETS[@]}"

# 2. Windows (Client, Node) - Static
GOOS=windows CGO_ENABLED=0 go list -tags="selinux,notest,grpcnotrace" -deps -json "${RESOLVED_CLIENT_TARGETS[@]}" "${RESOLVED_NODE_TARGETS[@]}"

# 3. Darwin (Client) - Static & Dynamic
GOOS=darwin CGO_ENABLED=0 go list -tags="selinux,notest,grpcnotrace" -deps -json "${RESOLVED_CLIENT_TARGETS[@]}"
GOOS=darwin CGO_ENABLED=1 go list -tags="selinux,notest,grpcnotrace" -deps -json "${RESOLVED_CLIENT_TARGETS[@]}"
