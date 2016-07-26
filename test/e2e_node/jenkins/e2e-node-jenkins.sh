#!/bin/bash

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

go get -u github.com/jteeuwen/go-bindata/go-bindata
make generated_files

# TODO converge build steps with hack/build-go some day if possible.
go generate test/e2e/framework/gobindata_util.go
go build test/e2e_node/environment/conformance.go

PARALLELISM=${PARALLELISM:-8}
WORKSPACE=${WORKSPACE:-"/tmp/"}
ARTIFACTS=${WORKSPACE}/_artifacts

mkdir -p ${ARTIFACTS}

if [[ -f "${KUBEKINS_SERVICE_ACCOUNT_FILE:-}" ]]; then
  echo 'Activating service account...'  # No harm in doing this multiple times.
  gcloud auth activate-service-account --key-file="${KUBEKINS_SERVICE_ACCOUNT_FILE}"
  # https://developers.google.com/identity/protocols/application-default-credentials
  export GOOGLE_APPLICATION_CREDENTIALS="${KUBEKINS_SERVICE_ACCOUNT_FILE}"
  unset KUBEKINS_SERVICE_ACCOUNT_FILE
elif [[ -n "${KUBEKINS_SERVICE_ACCOUNT_FILE:-}" ]]; then
  echo "ERROR: cannot access service account file at: ${KUBEKINS_SERVICE_ACCOUNT_FILE}"
fi

go run test/e2e_node/runner/run_e2e.go  --logtostderr --vmodule=*=4 --ssh-env="gce" \
  --zone="$GCE_ZONE" --project="$GCE_PROJECT" --hosts="$GCE_HOSTS" \
  --images="$GCE_IMAGES" --image-project="$GCE_IMAGE_PROJECT" \
  --image-config-file="$GCE_IMAGE_CONFIG_PATH" --cleanup="$CLEANUP" \
  --results-dir="$ARTIFACTS" --ginkgo-flags="--nodes=$PARALLELISM $GINKGO_FLAGS" \
  --setup-node="$SETUP_NODE" --test_args="$TEST_ARGS" --instance-metadata="$GCE_INSTANCE_METADATA"
