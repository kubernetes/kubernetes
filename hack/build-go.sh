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

# This script sets up a go workspace locally and builds all go components.

set -o errexit
set -o nounset
set -o pipefail

hackdir=$(CDPATH="" cd $(dirname $0); pwd)

# Set the environment variables required by the build.
. "${hackdir}/config-go.sh"

# Go to the top of the tree.
cd "${KUBE_REPO_ROOT}"

# Update the version.
"${hackdir}/version-gen.sh"

if [[ $# == 0 ]]; then
  # Update $@ with the default list of targets to build.
  set -- cmd/proxy cmd/apiserver cmd/controller-manager cmd/kubelet cmd/kubecfg plugin/cmd/scheduler
fi

binaries=()
for arg; do
  binaries+=("${KUBE_GO_PACKAGE}/${arg}")
done

go install "${binaries[@]}"
