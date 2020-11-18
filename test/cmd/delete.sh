#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

# Runs tests related to kubectl delete --all-namespaces.
run_kubectl_delete_allnamespaces_tests() {
  set -o nounset
  set -o errexit

  ns_one="namespace-$(date +%s)-${RANDOM}"
  ns_two="namespace-$(date +%s)-${RANDOM}"
  kubectl create namespace "${ns_one}"
  kubectl create namespace "${ns_two}"

  kubectl create configmap "one" --namespace="${ns_one}"
  kubectl create configmap "two" --namespace="${ns_two}"
  kubectl label configmap "one" --namespace="${ns_one}" deletetest=true
  kubectl label configmap "two" --namespace="${ns_two}" deletetest=true

  # dry-run
  kubectl delete configmap --dry-run=client -l deletetest=true --all-namespaces
  kubectl delete configmap --dry-run=server -l deletetest=true --all-namespaces
  kubectl config set-context "${CONTEXT}" --namespace="${ns_one}"
  kube::test::get_object_assert 'configmap -l deletetest' "{{range.items}}{{${id_field:?}}}:{{end}}" 'one:'
  kubectl config set-context "${CONTEXT}" --namespace="${ns_two}"
  kube::test::get_object_assert 'configmap -l deletetest' "{{range.items}}{{${id_field:?}}}:{{end}}" 'two:'

  kubectl delete configmap -l deletetest=true --all-namespaces

  # no configmaps should be in either of those namespaces with label deletetest
  kubectl config set-context "${CONTEXT}" --namespace="${ns_one}"
  kube::test::get_object_assert 'configmap -l deletetest' "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kubectl config set-context "${CONTEXT}" --namespace="${ns_two}"
  kube::test::get_object_assert 'configmap -l deletetest' "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  set +o nounset
  set +o errexit
}
