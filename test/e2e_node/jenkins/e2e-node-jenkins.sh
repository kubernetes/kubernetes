#!/bin/bash

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# Script executed by jenkins to run node e2e tests against gce
# Usage: test/e2e_node/jenkins/e2e-node-jenkins.sh <path to properties>
# Properties files:
# - test/e2e_node/jenkins/jenkins-ci.properties : for running jenkins ci
# - test/e2e_node/jenkins/jenkins-pull.properties : for running jenkins pull request builder
# - test/e2e_node/jenkins/template.properties : template for creating a properties file to run locally

set -e
set -x

: "${1:?Usage test/e2e_node/jenkins/e2e-node-jenkins.sh <path to properties>}"

. $1

if [ "$INSTALL_GODEP" = true ] ; then
  go get -u github.com/tools/godep
  go get -u github.com/onsi/ginkgo/ginkgo
  go get -u github.com/onsi/gomega
fi

godep go build test/e2e_node/environment/conformance.go
godep go run test/e2e_node/runner/run_e2e.go  --logtostderr --v="2" --ssh-env="gce" --zone="$GCE_ZONE" --project="$GCE_PROJECT"  --hosts="$GCE_HOSTS" --images="$GCE_IMAGES"
