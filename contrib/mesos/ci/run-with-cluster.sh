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

# Deploys a test cluster, runs the specified command, and destroys the test cluster.
# Runs all commands inside the mesosphere/kubernetes-mesos-test docker image (built on demand).
# Uses the mesos/docker cluster provider.
#
# Prerequisite:
# ./cluster/mesos/docker/test/build.sh
#
# Example Usage:
# ./contrib/mesos/ci/run-with-cluster.sh ./cluster/test-smoke.sh -v=2

set -o errexit
set -o nounset
set -o pipefail
set -o errtrace

RUN_CMD="$@"
[ -z "${RUN_CMD:-}" ] && echo "No command supplied" && exit 1

KUBERNETES_PROVIDER="mesos/docker"

MESOS_DOCKER_WORK_DIR="${MESOS_DOCKER_WORK_DIR:-${HOME}/tmp/kubernetes}"

KUBE_ROOT=$(cd "$(dirname "${BASH_SOURCE}")/../../.." && pwd)

# Clean (test artifacts)
echo "Cleaning work dir"
echo "${MESOS_DOCKER_WORK_DIR}"
rm -rf "${MESOS_DOCKER_WORK_DIR}"
mkdir -p "${MESOS_DOCKER_WORK_DIR}"

echo "Detecting docker client"
# Mount docker client binary to avoid client/compose/daemon version conflicts
if [ -n "${DOCKER_MACHINE_NAME:-}" ] && which docker-machine; then
  # On a Mac with docker-machine, use the binary in the VM, not the host binary
  DOCKER_BIN_PATH="$(docker-machine ssh "${DOCKER_MACHINE_NAME}" which docker)"
else
  DOCKER_BIN_PATH="$(which docker)"
fi
echo "${DOCKER_BIN_PATH}"

# Clean (k8s output & images), Build, Kube-Up, Test, Kube-Down
cd "${KUBE_ROOT}"
docker run \
  --rm \
  -v "${KUBE_ROOT}:/go/src/github.com/GoogleCloudPlatform/kubernetes" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -v "${DOCKER_BIN_PATH}:/usr/bin/docker" \
  -v "${MESOS_DOCKER_WORK_DIR}/auth:${MESOS_DOCKER_WORK_DIR}/auth" \
  -v "${MESOS_DOCKER_WORK_DIR}/log:${MESOS_DOCKER_WORK_DIR}/log" \
  -v "${MESOS_DOCKER_WORK_DIR}/mesosslave1/mesos:${MESOS_DOCKER_WORK_DIR}/mesosslave1/mesos" \
  -v "${MESOS_DOCKER_WORK_DIR}/mesosslave2/mesos:${MESOS_DOCKER_WORK_DIR}/mesosslave2/mesos" \
  -v "${MESOS_DOCKER_WORK_DIR}/overlay:${MESOS_DOCKER_WORK_DIR}/overlay" \
  -v "${MESOS_DOCKER_WORK_DIR}/reports:${MESOS_DOCKER_WORK_DIR}/reports" \
  $(test -d /teamcity/system/git && echo "-v /teamcity/system/git:/teamcity/system/git" || true) \
  -e "MESOS_DOCKER_WORK_DIR=${MESOS_DOCKER_WORK_DIR}" \
  -e "MESOS_DOCKER_IMAGE_DIR=/var/tmp/kubernetes" \
  -e "MESOS_DOCKER_OVERLAY_DIR=${MESOS_DOCKER_WORK_DIR}/overlay" \
  -e "KUBERNETES_CONTRIB=mesos" \
  -e "KUBERNETES_PROVIDER=mesos/docker" \
  -e "USER=root" \
  -e "E2E_REPORT_DIR=${MESOS_DOCKER_WORK_DIR}/reports" \
  -t $(tty &>/dev/null && echo "-i") \
  mesosphere/kubernetes-mesos-test \
  -ceux "\
    make clean all && \
    trap 'timeout 5m ./cluster/kube-down.sh' EXIT && \
    ./cluster/kube-down.sh && \
    ./cluster/kube-up.sh && \
    trap \"test \\\$? != 0 && export MESOS_DOCKER_DUMP_LOGS=true; cd \${PWD} && timeout 5m ./cluster/kube-down.sh\" EXIT && \
    ${RUN_CMD}
  "
