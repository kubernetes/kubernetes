#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../..

if [[ -n "${1:-}" ]]; then
  export KUBE_GCE_ZONE="${1}"
fi
if [[ -n "${2:-}" ]]; then
  export MULTIZONE="${2}"
fi
if [[ -n "${3:-}" ]]; then
  export KUBE_DELETE_NODES="${3}"
fi
if [[ -n "${4:-}" ]]; then
  export KUBE_USE_EXISTING_MASTER="${4}"
fi

source "${KUBE_ROOT}/hack/e2e-internal/e2e-down.sh"

