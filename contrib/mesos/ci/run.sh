#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# Runs the specified command in the test container (mesosphere/kubernetes-mesos-test).
#
# Prerequisite:
# ./cluster/mesos/docker/test/build.sh
#
# Example Usage:
# ./contrib/mesos/ci/run.sh make test

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

RUN_CMD="$@"
[ -z "${RUN_CMD:-}" ] && echo "No command supplied" && exit 1

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)

echo "Detecting docker client"
# Mount docker client binary to avoid client/compose/daemon version conflicts
if [ -n "${DOCKER_MACHINE_NAME:-}" ] && which docker-machine; then
  # On a Mac with docker-machine, use the binary in the VM, not the host binary
  DOCKER_BIN_PATH="$(docker-machine ssh "${DOCKER_MACHINE_NAME}" which docker)"
else
  DOCKER_BIN_PATH="$(which docker)"
fi
echo "${DOCKER_BIN_PATH}"

# Clean (k8s output & images) & Build
cd "${KUBE_ROOT}"
exec docker run \
  --rm \
  -v "${KUBE_ROOT}:/go/src/github.com/GoogleCloudPlatform/kubernetes" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -v "${DOCKER_BIN_PATH}:/usr/bin/docker" \
  -e "KUBERNETES_CONTRIB=mesos" \
  -e "USER=root" \
  -t $(tty &>/dev/null && echo "-i") \
  mesosphere/kubernetes-mesos-test \
  -ceux "${RUN_CMD}"
