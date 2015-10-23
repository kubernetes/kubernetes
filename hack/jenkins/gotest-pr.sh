#!/bin/bash

# Copyright 2015 The Kubernetes Authors All rights reserved.
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
set -o xtrace

export REPO_DIR=${REPO_DIR:-$(pwd)}

# Produce a JUnit-style XML test report for Jenkins.
export KUBE_JUNIT_REPORT_DIR=${WORKSPACE}/_artifacts

# Run the kubekins container, mapping in docker (so we can launch containers)
# and the repo directory
docker run -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(which docker)":/bin/docker \
  -v "${REPO_DIR}":/go/src/k8s.io/kubernetes \
  -v "${KUBE_JUNIT_REPORT_DIR}":/workspace/artifacts \
  -it kubekins-test \
  bash -c "cd kubernetes && /workspace/run.sh"
