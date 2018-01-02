#!/usr/bin/env bash

# Copyright 2016 Google Inc. All rights reserved.
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

set -e
set -x

BUILDER=${BUILDER:-false} # Whether this is running a PR builder job.

export GO_FLAGS="-race"
export GORACE="halt_on_error=1"

# Check whether assets need to be rebuilt.
FORCE=true build/assets.sh
if [[ ! -z "$(git diff --name-only pages)" ]]; then
  echo "Found changes to UI assets:"
  git diff --name-only pages
  echo "Run: `make assets FORCE=true`"
  exit 1
fi

# Build & test with go 1.8
docker run --rm \
       -w "/go/src/github.com/google/cadvisor" \
       -v "${GOPATH}/src/github.com/google/cadvisor:/go/src/github.com/google/cadvisor" \
       golang:1.8 make all test-runner

# Nodes that are currently stable. When tests fail on a specific node, and the failure is not remedied within a week, that node will be removed from this list.
golden_nodes=(
  e2e-cadvisor-ubuntu-trusty
  e2e-cadvisor-container-vm-v20151215
  e2e-cadvisor-container-vm-v20160127
  e2e-cadvisor-rhel-7
)

# TODO: Add test on GCI

# TODO: Add test for kubernetes default image
# e2e-cadvisor-container-vm-v20160321

# TODO: Temporarily disabled for #1344
# e2e-cadvisor-coreos-beta

# TODO: enable when docker 1.10  is working
# e2e-cadvisor-ubuntu-trusty-docker110

# TODO: Always fails with "Network tx and rx bytes should not be equal"
# e2e-cadvisor-centos-v7

max_retries=8

./runner --logtostderr --test-retry-count=$max_retries \
  --test-retry-whitelist=integration/runner/retrywhitelist.txt \
  --ssh-options "-i /var/lib/jenkins/gce_keys/google_compute_engine -o UserKnownHostsFile=/dev/null -o IdentitiesOnly=yes -o CheckHostIP=no -o StrictHostKeyChecking=no" \
  ${golden_nodes[*]}
