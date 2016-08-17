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

# Save environment variables in $WORKSPACE/env.list and then run the Jenkins e2e
# test runner inside the kubekins-test Docker image.

set -o errexit
set -o nounset
set -o pipefail

export REPO_DIR=${REPO_DIR:-$(pwd)}
export HOST_ARTIFACTS_DIR=${WORKSPACE}/_artifacts
mkdir -p "${HOST_ARTIFACTS_DIR}"

# TODO(ixdy): remove when all jobs are setting these vars using Jenkins credentials
: ${JENKINS_GCE_SSH_PRIVATE_KEY_FILE:='/var/lib/jenkins/gce_keys/google_compute_engine'}
: ${JENKINS_GCE_SSH_PUBLIC_KEY_FILE:='/var/lib/jenkins/gce_keys/google_compute_engine.pub'}

env \
  -u HOME \
  -u KUBEKINS_SERVICE_ACCOUNT_FILE \
  -u PATH \
  -u PWD \
  -u WORKSPACE \
  >${WORKSPACE}/env.list

docker_extra_args=()
if [[ "${JENKINS_ENABLE_DOCKER_IN_DOCKER:-}" =~ ^[yY]$ ]]; then
    docker_extra_args+=(\
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v "${REPO_DIR}":/go/src/k8s.io/kubernetes \
      -e "REPO_DIR=${REPO_DIR}" \
      -e "HOST_ARTIFACTS_DIR=${HOST_ARTIFACTS_DIR}" \
    )
fi

echo "Starting..."
docker run --rm=true -i \
  -v "${WORKSPACE}/_artifacts":/workspace/_artifacts \
  -v /etc/localtime:/etc/localtime:ro \
  ${JENKINS_GCE_SSH_PRIVATE_KEY_FILE:+-v "${JENKINS_GCE_SSH_PRIVATE_KEY_FILE}:/workspace/.ssh/google_compute_engine:ro"} \
  ${JENKINS_GCE_SSH_PUBLIC_KEY_FILE:+-v "${JENKINS_GCE_SSH_PUBLIC_KEY_FILE}:/workspace/.ssh/google_compute_engine.pub:ro"} \
  ${JENKINS_AWS_SSH_PRIVATE_KEY_FILE:+-v "${JENKINS_AWS_SSH_PRIVATE_KEY_FILE}:/workspace/.ssh/kube_aws_rsa:ro"} \
  ${JENKINS_AWS_SSH_PUBLIC_KEY_FILE:+-v "${JENKINS_AWS_SSH_PUBLIC_KEY_FILE}:/workspace/.ssh/kube_aws_rsa.pub:ro"} \
  ${JENKINS_AWS_CREDENTIALS_FILE:+-v "${JENKINS_AWS_CREDENTIALS_FILE}:/workspace/.aws/credentials:ro"} \
  ${KUBEKINS_SERVICE_ACCOUNT_FILE:+-v "${KUBEKINS_SERVICE_ACCOUNT_FILE}:/service-account.json:ro"} \
  --env-file "${WORKSPACE}/env.list" \
  -e "HOME=/workspace" \
  -e "WORKSPACE=/workspace" \
  ${KUBEKINS_SERVICE_ACCOUNT_FILE:+-e "KUBEKINS_SERVICE_ACCOUNT_FILE=/service-account.json"} \
  "${docker_extra_args[@]:+${docker_extra_args[@]}}" \
  gcr.io/google_containers/kubekins-test:go1.6.3-docker1.9.1-rev3 \
  bash -c "bash <(curl -fsS --retry 3 --keepalive-time 2 'https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh')"
