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

# Save environment variables in $WORKSPACE/env.list and then run the Jenkins e2e
# test runner inside the kubekins-test Docker image.

set -o errexit
set -o nounset
set -o pipefail

export REPO_DIR=${REPO_DIR:-$(pwd)}
export HOST_ARTIFACTS_DIR=${WORKSPACE}/_artifacts
mkdir -p "${HOST_ARTIFACTS_DIR}"

env -u HOME -u PATH -u PWD -u WORKSPACE >${WORKSPACE}/env.list

# Add all uncommented lines for metadata.google.internal in /etc/hosts to the
# test container.
readonly METADATA_SERVER_ADD_HOST_ARGS=($(
  grep '^[0-9a-fA-F\.:]\+ \+metadata\.google\.internal' /etc/hosts |\
  cut -f1 -d' ' |\
  xargs -r printf -- '--add-host="metadata.google.internal:%s"\n'))

docker_extra_args=()
if [[ "${JENKINS_ENABLE_DOCKER_IN_DOCKER:-}" =~ ^[yY]$ ]]; then
    docker_extra_args+=(\
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v "$(which docker)":/bin/docker:ro \
      -v "${REPO_DIR}":/go/src/k8s.io/kubernetes \
      -e "REPO_DIR=${REPO_DIR}" \
      -e "HOST_ARTIFACTS_DIR=${HOST_ARTIFACTS_DIR}" \
    )
fi

docker run --rm=true -i \
  -v "${WORKSPACE}/_artifacts":/workspace/_artifacts \
  -v /etc/localtime:/etc/localtime:ro \
  -v /var/lib/jenkins/gce_keys:/workspace/.ssh:ro `# TODO(ixdy): remove when all jobs are using JENKINS_GCE_SSH_KEYFILE` \
  ${JENKINS_GCE_SSH_KEY_FILE:+-v "${JENKINS_GCE_SSH_KEY_FILE}:/workspace/.ssh/google_compute_engine:ro"} \
  ${JENKINS_AWS_SSH_KEY_FILE:+-v "${JENKINS_AWS_SSH_KEY_FILE}:/workspace/.ssh/kube_aws_rsa:ro"} \
  ${JENKINS_AWS_CREDENTIALS_FILE:+-v "${JENKINS_AWS_CREDENTIALS_FILE}:/workspace/.aws/credentials:ro"} \
  --env-file "${WORKSPACE}/env.list" \
  -e "HOME=/workspace" \
  -e "WORKSPACE=/workspace" \
  "${docker_extra_args[@]:+${docker_extra_args[@]}}" \
  "${METADATA_SERVER_ADD_HOST_ARGS[@]:+${METADATA_SERVER_ADD_HOST_ARGS[@]}}" \
  gcr.io/google_containers/kubekins-test:0.11 \
  bash -c "bash <(curl -fsS --retry 3 'https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh')"
