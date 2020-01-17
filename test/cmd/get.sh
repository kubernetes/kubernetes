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

run_kubectl_get_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl get"
  ### Test retrieval of non-existing pods
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]:?}")
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'

  ### Test retrieval of non-existing POD with output flag specified
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]}" -o name)
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'

  ### Test retrieval of pods when none exist with non-human readable output format flag specified
  # Pre-condition: no pods exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o json)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o yaml)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o name)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o jsonpath='{.items}')
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o go-template='{{.items}}')
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o custom-columns=NAME:.metadata.name)
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'

  ### Test retrieval of pods when none exist, with human-readable output format flag specified
  # Pre-condition: no pods exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get foobar 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should not be part of the output when an error occurs
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods --ignore-not-found 2>&1 "${kube_flags[@]}")
  # Post-condition: The text "No resources found" should not be part of the output
  kube::test::if_has_not_string "${output_message}" 'No resources found'
  # Command
  output_message=$(kubectl get pods 2>&1 "${kube_flags[@]}" -o wide)
  # Post-condition: The text "No resources found" should be part of the output
  kube::test::if_has_string "${output_message}" 'No resources found'

  ### Test retrieval of non-existing POD with json output flag specified
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl get pods abc 2>&1 "${kube_flags[@]}" -o json)
  # Post-condition: POD abc should error since it doesn't exist
  kube::test::if_has_string "${output_message}" 'pods "abc" not found'
  # Post-condition: make sure we don't display an empty List
  kube::test::if_has_not_string "${output_message}" 'List'

  ### Test kubectl get all
  output_message=$(kubectl --v=6 --namespace default get all --chunk-size=0 2>&1 "${kube_flags[@]}")
  # Post-condition: Check if we get 200 OK from all the url(s)
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/pods 200 OK"
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/replicationcontrollers 200 OK"
  kube::test::if_has_string "${output_message}" "/api/v1/namespaces/default/services 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/apps/v1/namespaces/default/daemonsets 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/apps/v1/namespaces/default/deployments 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/apps/v1/namespaces/default/replicasets 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/apps/v1/namespaces/default/statefulsets 200 OK"
  kube::test::if_has_string "${output_message}" "/apis/autoscaling/v1/namespaces/default/horizontalpodautoscalers 200"
  kube::test::if_has_string "${output_message}" "/apis/batch/v1/namespaces/default/jobs 200 OK"
  kube::test::if_has_not_string "${output_message}" "/apis/extensions/v1beta1/namespaces/default/daemonsets 200 OK"
  kube::test::if_has_not_string "${output_message}" "/apis/extensions/v1beta1/namespaces/default/deployments 200 OK"
  kube::test::if_has_not_string "${output_message}" "/apis/extensions/v1beta1/namespaces/default/replicasets 200 OK"

  ### Test kubectl get chunk size
  output_message=$(kubectl --v=6 get clusterrole --chunk-size=10 2>&1 "${kube_flags[@]}")
  # Post-condition: Check if we get a limit and continue
  kube::test::if_has_string "${output_message}" "/clusterroles?limit=10 200 OK"
  kube::test::if_has_string "${output_message}" "/v1/clusterroles?continue="

  ### Test kubectl get chunk size defaults to 500
  output_message=$(kubectl --v=6 get clusterrole 2>&1 "${kube_flags[@]}")
  # Post-condition: Check if we get a limit and continue
  kube::test::if_has_string "${output_message}" "/clusterroles?limit=500 200 OK"

  ### Test kubectl get accumulates pages
  output_message=$(kubectl get namespaces --chunk-size=1 --no-headers "${kube_flags[@]}")
  # Post-condition: Check we got multiple pages worth of namespaces
  kube::test::if_has_string "${output_message}" "default"
  kube::test::if_has_string "${output_message}" "kube-public"
  kube::test::if_has_string "${output_message}" "kube-system"

  ### Test kubectl get chunk size does not result in a --watch error when resource list is served in multiple chunks
  # Pre-condition: ConfigMap one two tree does not exist
  kube::test::get_object_assert 'configmaps' "{{range.items}}{{ if eq $id_field \\\"one\\\" }}found{{end}}{{end}}:" ':'
  kube::test::get_object_assert 'configmaps' "{{range.items}}{{ if eq $id_field \\\"two\\\" }}found{{end}}{{end}}:" ':'
  kube::test::get_object_assert 'configmaps' "{{range.items}}{{ if eq $id_field \\\"three\\\" }}found{{end}}{{end}}:" ':'

  # Post-condition: Create three configmaps and ensure that we can --watch them with a --chunk-size of 1
  kubectl create cm one "${kube_flags[@]}"
  kubectl create cm two "${kube_flags[@]}"
  kubectl create cm three "${kube_flags[@]}"
  output_message=$(kubectl get configmap --chunk-size=1 --watch --request-timeout=1s 2>&1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" "watch is only supported on individual resources"
  output_message=$(kubectl get configmap --chunk-size=1 --watch-only --request-timeout=1s 2>&1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" "watch is only supported on individual resources"

  ### Test --allow-missing-template-keys
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## check --allow-missing-template-keys defaults to true for jsonpath templates
  kubectl get "${kube_flags[@]}" pod valid-pod -o jsonpath='{.missing}'

  ## check --allow-missing-template-keys defaults to true for go templates
  kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{.missing}}'

  ## check --template flag causes go-template to be printed, even when no --output value is provided
  output_message=$(kubectl get "${kube_flags[@]}" pod valid-pod --template="{{$id_field}}:")
  kube::test::if_has_string "${output_message}" 'valid-pod:'

  ## check --allow-missing-template-keys=false results in an error for a missing key with jsonpath
  output_message=$(! kubectl get pod valid-pod --allow-missing-template-keys=false -o jsonpath='{.missing}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'missing is not found'

  ## check --allow-missing-template-keys=false results in an error for a missing key with go
  output_message=$(! kubectl get pod valid-pod --allow-missing-template-keys=false -o go-template='{{.missing}}' "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'map has no entry for key "missing"'

  ### Test kubectl get watch
  output_message=$(kubectl get pods -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'STATUS'    # headers
  kube::test::if_has_string "${output_message}" 'valid-pod' # pod details
  output_message=$(kubectl get pods/valid-pod -o name -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" 'STATUS' # no headers
  kube::test::if_has_string     "${output_message}" 'pod/valid-pod' # resource name
  output_message=$(kubectl get pods/valid-pod -o yaml -w --request-timeout=1 "${kube_flags[@]}")
  kube::test::if_has_not_string "${output_message}" 'STATUS'          # no headers
  kube::test::if_has_string     "${output_message}" 'name: valid-pod' # yaml
  output_message=$(! kubectl get pods/invalid-pod -w --request-timeout=1 "${kube_flags[@]}" 2>&1)
  kube::test::if_has_string "${output_message}" '"invalid-pod" not found'

  # cleanup
  kubectl delete pods valid-pod "${kube_flags[@]}"

  ### Test 'kubectl get -f <file> -o <non default printer>' prints all the items in the file's list
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: PODs redis-master and valid-pod exist

  # Check that all items in the list are printed
  output_message=$(kubectl get -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml -o jsonpath="{..metadata.name}" "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "redis-master valid-pod"

  # cleanup
  kubectl delete pods redis-master valid-pod "${kube_flags[@]}"

  ### Test 'kubectl get -k <dir>' prints all the items built from a kustomization directory
  # Pre-condition: no ConfigMap, Deployment, Service exist
  kube::test::get_object_assert configmaps "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl apply -k hack/testdata/kustomize
  # Post-condition: test-the-map, test-the-deployment, test-the-service exist

  # Check that all items in the list are printed
  output_message=$(kubectl get -k hack/testdata/kustomize -o jsonpath="{..metadata.name}" "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "test-the-map"
  kube::test::if_has_string "${output_message}" "test-the-deployment"
  kube::test::if_has_string "${output_message}" "test-the-service"

  # cleanup
  kubectl delete -k hack/testdata/kustomize

  # Check that all items in the list are deleted
  kube::test::get_object_assert configmaps "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_retrieve_multiple_tests() {
  set -o nounset
  set -o errexit

  # switch back to the default namespace
  kubectl config set-context "${CONTEXT}" --namespace=""
  kube::log::status "Testing kubectl(v1:multiget)"
  kube::test::get_object_assert 'nodes/127.0.0.1 service/kubernetes' "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:kubernetes:'

  set +o nounset
  set +o errexit
}

run_kubectl_sort_by_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --sort-by"

  ### sort-by should not panic if no pod exists
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl get pods --sort-by="{metadata.name}"
  kubectl get pods --sort-by="{metadata.creationTimestamp}"

  ### sort-by should works if pod exists
  # Create POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Check output of sort-by
  output_message=$(kubectl get pods --sort-by="{metadata.name}")
  kube::test::if_has_string "${output_message}" "valid-pod"
  # ensure sort-by receivers objects as Table
  output_message=$(kubectl get pods --v=8 --sort-by="{metadata.name}" 2>&1)
  kube::test::if_has_string "${output_message}" "as=Table"
  # ensure sort-by requests the full object
  kube::test::if_has_string "${output_message}" "includeObject=Object"
  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod valid-pod --grace-period=0 --force
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### sort-by should works by sorting by name
  # Create three PODs
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod1.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:'
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod2.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:'
  # Command
  kubectl create "${kube_flags[@]}" -f hack/testdata/sorted-pods/sorted-pod3.yaml
  # Post-condition: sorted-pod1 is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:sorted-pod3:'

  # Check output of sort-by '{metadata.name}'
  output_message=$(kubectl get pods --sort-by="{metadata.name}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod1:sorted-pod2:sorted-pod3:"

  # Check output of sort-by '{metadata.labels.name}'
  output_message=$(kubectl get pods --sort-by="{metadata.labels.name}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod3:sorted-pod2:sorted-pod1:"

  # if sorting, we should be able to use any field in our objects
  output_message=$(kubectl get pods --sort-by="{spec.containers[0].name}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod2:sorted-pod1:sorted-pod3:"

  # ensure sorting by creation timestamps works
  output_message=$(kubectl get pods --sort-by="{metadata.creationTimestamp}")
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod1:sorted-pod2:sorted-pod3:"

  # ensure sorting using fallback codepath still works
  output_message=$(kubectl get pods --sort-by="{spec.containers[0].name}" --server-print=false --v=8 2>&1)
  kube::test::if_sort_by_has_correct_order "${output_message}" "sorted-pod2:sorted-pod1:sorted-pod3:"
  kube::test::if_has_not_string "${output_message}" "Table"

  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'sorted-pod1:sorted-pod2:sorted-pod3:'
  # Command
  kubectl delete "${kube_flags[@]}" pod --grace-period=0 --force --all
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_kubectl_all_namespace_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --all-namespace"

  # Pre-condition: the "default" namespace exists
  kube::test::get_object_assert namespaces "{{range.items}}{{if eq $id_field \\\"default\\\"}}{{$id_field}}:{{end}}{{end}}" 'default:'

  ### Create POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Verify a specific namespace is ignored when all-namespaces is provided
  # Command
  kubectl get pods --all-namespaces --namespace=default

  ### Check --all-namespaces option shows namespaces
  # Create objects in multiple namespaces
  kubectl create "${kube_flags[@]}" namespace all-ns-test-1
  kubectl create "${kube_flags[@]}" serviceaccount test -n all-ns-test-1
  kubectl create "${kube_flags[@]}" namespace all-ns-test-2
  kubectl create "${kube_flags[@]}" serviceaccount test -n all-ns-test-2
  # Ensure listing across namespaces displays the namespace (--all-namespaces)
  output_message=$(kubectl get serviceaccounts --all-namespaces "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "all-ns-test-1"
  kube::test::if_has_string "${output_message}" "all-ns-test-2"
  # Ensure listing across namespaces displays the namespace (-A)
  output_message=$(kubectl get serviceaccounts -A "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "all-ns-test-1"
  kube::test::if_has_string "${output_message}" "all-ns-test-2"
  # Clean up
  kubectl delete "${kube_flags[@]}" namespace all-ns-test-1
  kubectl delete "${kube_flags[@]}" namespace all-ns-test-2

  ### Clean up
  # Pre-condition: valid-pod exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete "${kube_flags[@]}" pod valid-pod --grace-period=0 --force
  # Post-condition: valid-pod doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Verify flag all-namespaces is ignored for rootScoped resources
  # Pre-condition: node exists
  kube::test::get_object_assert nodes "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'
  # Command
  output_message=$(kubectl get nodes --all-namespaces 2>&1)
  # Post-condition: output with no NAMESPACE field
  kube::test::if_has_not_string "${output_message}" "NAMESPACE"

  set +o nounset
  set +o errexit
}
