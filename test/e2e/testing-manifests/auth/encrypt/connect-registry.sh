#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# input and default values
KIND_NETWORK="${1:-kind}"
REGISTRY_NAME="${2:-kind-registry}"

if [ "${KIND_NETWORK}" != "bridge" ]; then
  # wait for the kind network to exist
  for i in $(seq 1 25); do
    if docker network ls | grep "${KIND_NETWORK}"; then
      break
    else
      sleep 1
    fi
  done
  containers=$(docker network inspect "${KIND_NETWORK}" -f "{{range .Containers}}{{.Name}} {{end}}")
  needs_connect="true"
  for c in $containers; do
    if [ "$c" = "${REGISTRY_NAME}" ]; then
      needs_connect="false"
    fi
  done
  if [ "${needs_connect}" = "true" ]; then
    echo "connecting ${KIND_NETWORK} network to ${REGISTRY_NAME}"
    docker network connect "${KIND_NETWORK}" "${REGISTRY_NAME}" || true
  fi
fi
