#!/bin/sh

# Copyright 2025 The Kubernetes Authors.
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

set -ex

sudo rm -rf /var/run/kubernetes /var/run/cdi /var/lib/kubelet/plugins_registry /var/lib/kubelet/plugins /var/lib/kubelet/*_state /var/lib/kubelet/checkpoints /tmp/artifacts
sudo mkdir /var/run/kubernetes /var/lib/kubelet/plugins_registry /var/lib/kubelet/plugins /var/run/cdi
sudo chown "$(id -u)" /var/run/kubernetes /var/lib/kubelet/plugins_registry /var/lib/kubelet/plugins /var/run/cdi
ARTIFACTS=/tmp/artifacts
TMP_DIR=${ARTIFACTS}/tmp
LOG_DIR=${ARTIFACTS}/logs
TMPDIR=${TMP_DIR}
KUBERNETES_SERVER_BIN_DIR="$(pwd)/_output/local/bin/$(go env GOOS)/$(go env GOARCH)"
KUBERNETES_SERVER_CACHE_DIR="${KUBERNETES_SERVER_BIN_DIR}/cache-dir"
mkdir -p "${TMP_DIR}" "${LOG_DIR}"
export ARTIFACTS TMP_DIR LOG_DIR TMPDIR KUBERNETES_SERVER_BIN_DIR KUBERNETES_SERVER_CACHE_DIR

exec "$@"
