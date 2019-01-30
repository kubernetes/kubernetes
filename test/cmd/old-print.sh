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

run_kubectl_old_print_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl get --server-print=false"
  ### Test retrieval of all types in discovery
  # Pre-condition: no resources exist
  output_message=$(kubectl get pods --server-print=false 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: Expect text indicating no resources were found
  kube::test::if_has_string "${output_message}" 'No resources found.'

  ### Test retrieval of pods against server-side printing
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get pod "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get pod --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  # Test printing objects with --use-openapi-print-columns
  actual_output=$(kubectl get namespaces --use-openapi-print-columns --v=7 "${KUBE_FLAGS[@]}" 2>&1)
  # it should request full objects (not server-side printing)
  kube::test::if_has_not_string "${actual_output}" 'application/json;as=Table'
  kube::test::if_has_string "${actual_output}"     'application/json'

  ### Test retrieval of daemonsets against server-side printing
  kubectl apply -f hack/testdata/rollingupdate-daemonset.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: daemonset is created
  kube::test::get_object_assert ds "{{range.items}}{{$ID_FIELD}}:{{end}}" 'bind:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get ds "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get ds --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  ### Test retrieval of replicationcontrollers against server-side printing
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get rc "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get rc --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  ### Test retrieval of replicasets against server-side printing
  kubectl create -f hack/testdata/frontend-replicaset.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: frontend replica set is created
  kube::test::get_object_assert rs "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get rs "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get rs --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  ### Test retrieval of jobs against server-side printing
  kubectl run pi --generator=job/v1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${KUBE_FLAGS[@]}"
  # Post-Condition: assertion object exists
  kube::test::get_object_assert jobs "{{range.items}}{{$ID_FIELD}}:{{end}}" 'pi:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get jobs/pi "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get jobs/pi --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  ### Test retrieval of clusterroles against server-side printing
  kubectl create "${KUBE_FLAGS[@]}" clusterrole sample-role --verb=* --resource=pods
  # Post-Condition: assertion object exists
  kube::test::get_object_assert clusterrole/sample-role "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  # Compare "old" output with experimental output and ensure both are the same
  # remove the last column, as it contains the object's AGE, which could cause a mismatch.
  expected_output=$(kubectl get clusterroles/sample-role "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get clusterroles/sample-role --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  ### Test retrieval of crds against server-side printing
  kubectl "${KUBE_FLAGS_WITH_TOKEN[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "foos.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "scope": "Namespaced",
    "names": {
      "plural": "foos",
      "kind": "Foo"
    }
  }
}
__EOF__

  # Post-Condition: assertion object exists
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{if eq $ID_FIELD \\\"foos.company.com\\\"}}{{$ID_FIELD}}:{{end}}{{end}}" 'foos.company.com:'

  # Test that we can list this new CustomResource
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Compare "old" output with experimental output and ensure both are the same
  expected_output=$(kubectl get foos "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  actual_output=$(kubectl get foos --server-print=false "${KUBE_FLAGS[@]}" | awk 'NF{NF--};1')
  kube::test::if_has_string "${actual_output}" "${expected_output}"

  # teardown
  kubectl delete customresourcedefinitions/foos.company.com "${KUBE_FLAGS_WITH_TOKEN[@]}"
  kubectl delete clusterroles/sample-role "${KUBE_FLAGS_WITH_TOKEN[@]}"
  kubectl delete jobs pi "${KUBE_FLAGS[@]}"
  kubectl delete rs frontend "${KUBE_FLAGS[@]}"
  kubectl delete rc frontend "${KUBE_FLAGS[@]}"
  kubectl delete ds bind "${KUBE_FLAGS[@]}"
  kubectl delete pod valid-pod "${KUBE_FLAGS[@]}"

  set +o nounset
  set +o errexit
}
