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

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../../../../../.. && pwd -P)"

# This script was moved. The old entry point is kept around for whoever
# might need it.
exec bash "${KUBE_ROOT}/testutils/testing-manifests/storage-csi/external-snapshotter/volume-group-snapshots/run_group_snapshot_e2e.sh" "${@}"
