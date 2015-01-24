#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# Launches a container and verifies it can be reached. Assumes that
# we're being called by hack/e2e-test.sh (we use some env vars it sets up).

set -o errexit
set -o nounset
set -o pipefail

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/../..
source "${KUBE_ROOT}/cluster/kube-env.sh"
source "${KUBE_ROOT}/cluster/$KUBERNETES_PROVIDER/util.sh"

# Private image works only on GCE and GKE.
if [[ "${KUBERNETES_PROVIDER}" != "gce" ]] && [[ "${KUBERNETES_PROVIDER}" != "gke" ]]; then
  echo "WARNING: Skipping private.sh for cloud provider: ${KUBERNETES_PROVIDER}."
  exit 0
fi

# Run the basic.sh test, but using this image.
export POD_IMG_SRV="gcr.io/_b_k8s_test/serve_hostname:1.1"
source "${KUBE_ROOT}/hack/e2e-suite/basic.sh"
