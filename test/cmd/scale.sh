#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

run_kubectl_scale_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl scale"

  create_and_use_new_namespace

  kube::log::status "Testing kubectl scale output for Deployment"
  kubectl create deployment test-scale-deploy --image=busybox
  output_message=$(kubectl scale deployment test-scale-deploy --replicas=2 --dry-run=client -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 2'
  kube::test::get_object_assert 'deployment test-scale-deploy' '{{.spec.replicas}}' '1'
  output_message=$(kubectl scale deployment test-scale-deploy --replicas=3 --dry-run=server -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 3'
  kube::test::get_object_assert 'deployment test-scale-deploy' '{{.spec.replicas}}' '1'
  output_message=$(kubectl scale deployment test-scale-deploy --replicas=2 -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 2'
  kube::test::get_object_assert 'deployment test-scale-deploy' '{{.spec.replicas}}' '2'
  kubectl delete deployment test-scale-deploy

  kube::log::status "Testing kubectl scale output for ReplicaSet"
  kubectl create -f hack/testdata/frontend-replicaset.yaml
  output_message=$(kubectl scale -f hack/testdata/frontend-replicaset.yaml --replicas=2 -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 2'
  kubectl delete rs frontend

  kube::log::status "Testing kubectl scale output for ReplicationController"
  kubectl create -f hack/testdata/frontend-controller.yaml
  output_message=$(kubectl scale rc frontend --replicas=3 -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 3'
  kubectl delete rc frontend

  kube::log::status "Testing kubectl scale output for StatefulSet"
  kubectl create -f hack/testdata/rollingupdate-statefulset.yaml
  output_message=$(kubectl scale statefulset nginx --replicas=2 -o yaml)
  kube::test::if_has_string "${output_message}" 'replicas: 2'
  kubectl delete statefulset nginx

  set +o nounset
  set +o errexit
}
