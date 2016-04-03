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

godep go build github.com/google/cadvisor/integration/runner

# Host Notes
# e2e-cadvisor-ubuntu-trusty-docker110
# - ubuntu 14.04
# - docker 1.10
# e2e-cadvisor-container-vm-v20160127-docker18
# - docker 1.8.3
# e2e-cadvisor-container-vm-v20151215-docker18
# - docker 1.8.3
# e2e-cadvisor-ubuntu-trusty-docker19
# - ubunty 14.04
# - docker 1.9.1
# e2e-cadvisor-coreos-beta-docker19
# - docker 1.9.1
# e2e-cadvisor-rhel-7-docker19
# - red hat 7
# - docker 1.9.1
# e2e-cadvisor-centos-v7
# - docker 1.9.1

# Nodes that are currently stable. When tests fail on a specific node, and the failure is not remedied within a week, that node will be removed from this list.
golden_nodes="e2e-cadvisor-ubuntu-trusty-docker19 e2e-cadvisor-coreos-beta-docker19 e2e-cadvisor-container-vm-v20151215-docker18 e2e-cadvisor-container-vm-v20160127-docker18 e2e-cadvisor-rhel-7-docker19
"
# TODO: enable when docker 1.10  is working
# e2e-cadvisor-ubuntu-trusty-docker110

# Always fails with "Network tx and rx bytes should not be equal"
failing_nodes="e2e-cadvisor-centos-v7"

max_retries=8

./runner --logtostderr --test-retry-count=$max_retries --test-retry-whitelist=integration/runner/retrywhitelist.txt $golden_nodes
