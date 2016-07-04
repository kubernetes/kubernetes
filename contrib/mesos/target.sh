#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

# The set of server targets that we are only building for Linux
# Used by hack/lib/golang.sh
kube::contrib::mesos::server_targets() {
  local -r targets=(
    contrib/mesos/cmd/k8sm-scheduler
    contrib/mesos/cmd/k8sm-executor
    contrib/mesos/cmd/k8sm-controller-manager
    contrib/mesos/cmd/km
  )
  echo "${targets[@]}"
}

# The set of test targets that we are building for all platforms
# Used by hack/lib/golang.sh
kube::contrib::mesos::test_targets() {
  true
}

# The set of source targets to include in the kube-build image
# Used by build/common.sh
kube::contrib::mesos::source_targets() {
  local -r targets=(
    contrib/mesos
  )
  echo "${targets[@]}"
}
