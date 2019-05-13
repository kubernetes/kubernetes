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

set -o errexit
set -o nounset
set -o pipefail

run_kubeadm_tests() {
  set -o nounset
  set -o errexit

  KUBEADM_PATH="${KUBEADM_PATH:=$(kube::realpath "${KUBE_ROOT}")/cluster/kubeadm.sh}"

  # If testing a different version of kubeadm than the current build, you can
  # comment this out to save yourself from needlessly building here.
  make -C "${KUBE_ROOT}" WHAT=cmd/kubeadm

  #TODO(runyontr): Remove the the KUBE_TIMEOUT override when 
  # kubernetes/kubeadm/issues/1430 is fixed
  make -C "${KUBE_ROOT}" test \
  WHAT=k8s.io/kubernetes/cmd/kubeadm/test/cmd \
  KUBE_TEST_ARGS="--kubeadm-path '${KUBEADM_PATH}'" \
  KUBE_TIMEOUT="--timeout 600s"
  set +o nounset
  set +o errexit
}