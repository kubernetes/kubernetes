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

# from hack/lib/golang.sh and build/common.sh
readonly KUBE_GO_PACKAGE=k8s.io/kubernetes

readonly LOCAL_OUTPUT_ROOT="${KUBE_ROOT}/_output"
readonly LOCAL_OUTPUT_SUBPATH="${LOCAL_OUTPUT_ROOT}/dockerized"
readonly LOCAL_OUTPUT_BINPATH="${LOCAL_OUTPUT_SUBPATH}/bin"

readonly OUTPUT_BINPATH="${CUSTOM_OUTPUT_BINPATH:-$LOCAL_OUTPUT_BINPATH}"

readonly REMOTE_OUTPUT_ROOT="/go/src/${KUBE_GO_PACKAGE}/_output"
readonly REMOTE_OUTPUT_SUBPATH="${REMOTE_OUTPUT_ROOT}/dockerized"
readonly REMOTE_OUTPUT_BINPATH="${REMOTE_OUTPUT_SUBPATH}/bin"

readonly DOCKER_MOUNT_ARGS=(--volume "${OUTPUT_BINPATH}:${REMOTE_OUTPUT_BINPATH}")

# Clean (k8s output & images) & Build
cd "${KUBE_ROOT}"
exec docker run \
  --rm \
  -v "${KUBE_ROOT}:/go/src/k8s.io/kubernetes" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -v "${DOCKER_BIN_PATH}:/usr/bin/docker" \
  "${DOCKER_MOUNT_ARGS[@]}" \
  $(test -d /teamcity/system/git && echo "-v /teamcity/system/git:/teamcity/system/git" || true) \
  -e "KUBERNETES_CONTRIB=mesos" \
  -e "USER=root" \
  -t $(tty &>/dev/null && echo "-i") \
  mesosphere/kubernetes-mesos-test \
  -ceux "${RUN_CMD}"
