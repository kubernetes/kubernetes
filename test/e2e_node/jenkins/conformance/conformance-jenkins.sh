#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# Script executed by jenkins to run node conformance test against gce
# Usage: test/e2e_node/jenkins/conformance-node-jenkins.sh <path to properties>

set -e
set -x

: "${1:?Usage test/e2e_node/jenkins/conformance-node-jenkins.sh <path to properties>}"

. "${1}"

make generated_files

WORKSPACE=${WORKSPACE:-"/tmp/"}
ARTIFACTS=${WORKSPACE}/_artifacts
TIMEOUT=${TIMEOUT:-"45m"}

mkdir -p "${ARTIFACTS}"

go run test/e2e_node/runner/remote/run_remote.go  --test-suite=conformance \
  --logtostderr --vmodule=*=4 --ssh-env="gce" --ssh-user="$GCE_USER" \
  --zone="$GCE_ZONE" --project="$GCE_PROJECT" --hosts="$GCE_HOSTS" \
  --images="$GCE_IMAGES" --image-project="$GCE_IMAGE_PROJECT" \
  --image-config-file="$GCE_IMAGE_CONFIG_PATH" --cleanup="$CLEANUP" \
  --results-dir="$ARTIFACTS" --test-timeout="$TIMEOUT" \
  --test_args="--kubelet-flags=\"$KUBELET_ARGS\"" \
  --instance-metadata="$GCE_INSTANCE_METADATA" \
  --system-spec-name="$SYSTEM_SPEC_NAME" \
  --extra-envs="$EXTRA_ENVS"
