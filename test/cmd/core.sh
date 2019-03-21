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

run_configmap_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing configmaps"
  kubectl create -f test/fixtures/doc-yaml/user-guide/configmap/configmap.yaml
  kube::test::get_object_assert 'configmap/test-configmap' "{{$id_field}}" 'test-configmap'
  kubectl delete configmap test-configmap "${kube_flags[@]}"

  ### Create a new namespace
  # Pre-condition: the test-configmaps namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-configmaps\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-configmaps
  # Post-condition: namespace 'test-configmaps' is created.
  kube::test::get_object_assert 'namespaces/test-configmaps' "{{$id_field}}" 'test-configmaps'

  ### Create a generic configmap in a specific namespace
  # Pre-condition: configmap test-configmap and test-binary-configmap does not exist
  kube::test::get_object_assert 'configmaps' '{{range.items}}{{ if eq $id_field \"test-configmap\" }}found{{end}}{{end}}:' ':'
  kube::test::get_object_assert 'configmaps' '{{range.items}}{{ if eq $id_field \"test-binary-configmap\" }}found{{end}}{{end}}:' ':'

  # Command
  kubectl create configmap test-configmap --from-literal=key1=value1 --namespace=test-configmaps
  kubectl create configmap test-binary-configmap --from-file <( head -c 256 /dev/urandom ) --namespace=test-configmaps
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-configmaps' "{{$id_field}}" 'test-configmap'
  kube::test::get_object_assert 'configmap/test-binary-configmap --namespace=test-configmaps' "{{$id_field}}" 'test-binary-configmap'
  [[ "$(kubectl get configmap/test-configmap --namespace=test-configmaps -o yaml "${kube_flags[@]}" | grep 'key1: value1')" ]]
  [[ "$(kubectl get configmap/test-binary-configmap --namespace=test-configmaps -o yaml "${kube_flags[@]}" | grep 'binaryData')" ]]
  # Clean-up
  kubectl delete configmap test-configmap --namespace=test-configmaps
  kubectl delete configmap test-binary-configmap --namespace=test-configmaps
  kubectl delete namespace test-configmaps

  set +o nounset
  set +o errexit
}

# Runs all pod related tests.
run_pod_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:pods)"

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  kube::test::get_object_assert 'pod valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pod/valid-pod' "{{$id_field}}" 'valid-pod'
  kube::test::get_object_assert 'pods/valid-pod' "{{$id_field}}" 'valid-pod'
  # Repeat above test using jsonpath template
  kube::test::get_object_jsonpath_assert pods "{.items[*]$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod valid-pod' "{$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod/valid-pod' "{$id_field}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pods/valid-pod' "{$id_field}" 'valid-pod'
  # Describe command should print detailed information
  kube::test::describe_object_assert pods 'valid-pod' "Name:" "Image:" "Node:" "Labels:" "Status:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert pods 'valid-pod'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert pods 'valid-pod' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert pods 'valid-pod' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:"

  # Describe command should print events information by default
  kube::test::describe_resource_events_assert pods
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert pods false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert pods true
  ### Validate Export ###
  kube::test::get_object_assert 'pods/valid-pod' "{{.metadata.namespace}} {{.metadata.name}}" '<no value> valid-pod' "--export=true"

  ### Dump current valid-pod POD
  output_pod=$(kubectl get pod valid-pod -o yaml "${kube_flags[@]}")

  ### Delete POD valid-pod by id
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete POD valid-pod by id with --now
  # Pre-condition: valid-pod POD exists
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${kube_flags[@]}" --now
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Delete POD valid-pod by id with --grace-period=0
  # Pre-condition: valid-pod POD exists
  kubectl create "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command succeeds without --force by waiting
  kubectl delete pod valid-pod "${kube_flags[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from dumped YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  echo "${output_pod}" | ${SED} '/namespace:/d' | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod from JSON
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod)' "${kube_flags[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  ### Create POD valid-pod from YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  output_message=$(kubectl get pods --field-selector metadata.name=valid-pod "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "valid-pod"
  # Command
  phase=$(kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{ .status.phase }}')
  output_message=$(kubectl get pods --field-selector status.phase="${phase}" "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" "valid-pod"

  ### Delete PODs with no parameter mustn't kill everything
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete pods "${kube_flags[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete PODs with --all and a label selector is not permitted
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete --all pods -l'name in (valid-pod)' "${kube_flags[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Delete all PODs
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete --all pods "${kube_flags[@]}" --grace-period=0 --force # --all remove all the pods
  # Post-condition: no POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$id_field}}:{{end}}' ''

  # Detailed tests for describe pod output
    ### Create a new namespace
  # Pre-condition: the test-secrets namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-kubectl-describe-pod\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-kubectl-describe-pod
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-kubectl-describe-pod' "{{$id_field}}" 'test-kubectl-describe-pod'

  ### Create a generic secret
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key-1=value1 --type=test-type --namespace=test-kubectl-describe-pod
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$secret_type}}" 'test-type'

  ### Create a generic configmap
  # Pre-condition: CONFIGMAP test-configmap does not exist
  #kube::test::get_object_assert 'configmap/test-configmap --namespace=test-kubectl-describe-pod' "{{$id_field}}" ''
  kube::test::get_object_assert 'configmaps --namespace=test-kubectl-describe-pod' '{{range.items}}{{ if eq $id_field \"test-configmap\" }}found{{end}}{{end}}:' ':'

  #kube::test::get_object_assert 'configmaps --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create configmap test-configmap --from-literal=key-2=value2 --namespace=test-kubectl-describe-pod
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-kubectl-describe-pod' "{{$id_field}}" 'test-configmap'

  ### Create a pod disruption budget with minAvailable
  # Command
  kubectl create pdb test-pdb-1 --selector=app=rails --min-available=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-1 --namespace=test-kubectl-describe-pod' "{{$pdb_min_available}}" '2'
  # Command
  kubectl create pdb test-pdb-2 --selector=app=rails --min-available=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-2 --namespace=test-kubectl-describe-pod' "{{$pdb_min_available}}" '50%'

  ### Create a pod disruption budget with maxUnavailable
  # Command
  kubectl create pdb test-pdb-3 --selector=app=rails --max-unavailable=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-3 --namespace=test-kubectl-describe-pod' "{{$pdb_max_unavailable}}" '2'
  # Command
  kubectl create pdb test-pdb-4 --selector=app=rails --max-unavailable=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-4 --namespace=test-kubectl-describe-pod' "{{$pdb_max_unavailable}}" '50%'

  ### Fail creating a pod disruption budget if both maxUnavailable and minAvailable specified
  ! kubectl create pdb test-pdb --selector=app=rails --min-available=2 --max-unavailable=3 --namespace=test-kubectl-describe-pod

  # Create a pod that consumes secret, configmap, and downward API keys as envs
  kube::test::get_object_assert 'pods --namespace=test-kubectl-describe-pod' "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod-with-api-env.yaml --namespace=test-kubectl-describe-pod

  kube::test::describe_object_assert 'pods --namespace=test-kubectl-describe-pod' 'env-test-pod' "TEST_CMD_1" "<set to the key 'key-1' in secret 'test-secret'>" "TEST_CMD_2" "<set to the key 'key-2' of config map 'test-configmap'>" "TEST_CMD_3" "env-test-pod (v1:metadata.name)"
  # Describe command (resource only) should print detailed information about environment variables
  kube::test::describe_resource_assert 'pods --namespace=test-kubectl-describe-pod' "TEST_CMD_1" "<set to the key 'key-1' in secret 'test-secret'>" "TEST_CMD_2" "<set to the key 'key-2' of config map 'test-configmap'>" "TEST_CMD_3" "env-test-pod (v1:metadata.name)"

  # Clean-up
  kubectl delete pod env-test-pod --namespace=test-kubectl-describe-pod
  kubectl delete secret test-secret --namespace=test-kubectl-describe-pod
  kubectl delete configmap test-configmap --namespace=test-kubectl-describe-pod
  kubectl delete pdb/test-pdb-1 pdb/test-pdb-2 pdb/test-pdb-3 pdb/test-pdb-4 --namespace=test-kubectl-describe-pod
  kubectl delete namespace test-kubectl-describe-pod

  ### Create two PODs
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/kubectl/redis-master-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod and redis-master PODs are created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-master PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete pods valid-pod redis-master "${kube_flags[@]}" --grace-period=0 --force # delete multiple pods at once
  # Post-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  kubectl label pods valid-pod new-name=new-valid-pod "${kube_flags[@]}"
  # Post-condition: valid-pod is labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

  ### Label the valid-pod POD with empty label value
  # Pre-condition: valid-pod does not have label "emptylabel"
  kube::test::get_object_assert 'pod valid-pod' "{{range$labels_field}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'
  # Command
  kubectl label pods valid-pod emptylabel="" "${kube_flags[@]}"
  # Post-condition: valid pod contains "emptylabel" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.emptylabel}}" ''

  ### Annotate the valid-pod POD with empty annotation value
  # Pre-condition: valid-pod does not have annotation "emptyannotation"
  kube::test::get_object_assert 'pod valid-pod' "{{${annotations_field}.emptyannotation}}" '<no value>'
  # Command
  kubectl annotate pods valid-pod emptyannotation="" "${kube_flags[@]}"
  # Post-condition: valid pod contains "emptyannotation" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${annotations_field}.emptyannotation}}" ''

  ### Record label change
  # Pre-condition: valid-pod does not have record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range.items}}{{$annotations_field}}:{{end}}" ''
  # Command
  kubectl label pods valid-pod record-change=true --record=true "${kube_flags[@]}"
  # Post-condition: valid-pod has record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*--record=true.*"

  ### Do not record label change
  # Command
  kubectl label pods valid-pod no-record-change=true --record=false "${kube_flags[@]}"
  # Post-condition: valid-pod's record annotation still contains command with --record=true
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*--record=true.*"

  ### Record label change with specified flag and previous change already recorded
  ### we are no longer tricked by data from another user into revealing more information about our client
  # Command
  kubectl label pods valid-pod new-record-change=true --record=true "${kube_flags[@]}"
  # Post-condition: valid-pod's record annotation contains new change
  kube::test::get_object_assert 'pod valid-pod' "{{range$annotations_field}}{{.}}:{{end}}" ".*new-record-change=true.*"


  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -lnew-name=new-valid-pod --grace-period=0 --force "${kube_flags[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create pod-with-precision POD
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/pod-with-precision.json "${kube_flags[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'pod-with-precision:'

  ## Patch preserves precision
  # Command
  kubectl patch "${kube_flags[@]}" pod pod-with-precision -p='{"metadata":{"annotations":{"patchkey": "patchvalue"}}}'
  # Post-condition: pod-with-precision POD has patched annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${annotations_field}.patchkey}}" 'patchvalue'
  # Command
  kubectl label pods pod-with-precision labelkey=labelvalue "${kube_flags[@]}"
  # Post-condition: pod-with-precision POD has label
  kube::test::get_object_assert 'pod pod-with-precision' "{{${labels_field}.labelkey}}" 'labelvalue'
  # Command
  kubectl annotate pods pod-with-precision annotatekey=annotatevalue "${kube_flags[@]}"
  # Post-condition: pod-with-precision POD has annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${annotations_field}.annotatekey}}" 'annotatevalue'
  # Cleanup
  kubectl delete pod pod-with-precision "${kube_flags[@]}"

  ### Annotate POD YAML file locally without effecting the live pod.
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  # Command
  kubectl annotate -f hack/testdata/pod.yaml annotatekey=annotatevalue "${kube_flags[@]}"

  # Pre-condition: annotationkey is annotationvalue
  kube::test::get_object_assert 'pod test-pod' "{{${annotations_field}.annotatekey}}" 'annotatevalue'

  # Command
  output_message=$(kubectl annotate --local -f hack/testdata/pod.yaml annotatekey=localvalue -o yaml "${kube_flags[@]}")
  echo $output_message

  # Post-condition: annotationkey is still annotationvalue in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod test-pod' "{{${annotations_field}.annotatekey}}" 'annotatevalue'
  kube::test::if_has_string "${output_message}" "localvalue"

  # Cleanup
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"

  ### Create valid-pod POD
  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  ## kubectl create --edit can update the label filed of multiple resources. tmp-editor.sh is a fake editor
  TEMP=$(mktemp /tmp/tmp-editor-XXXXXXXX.sh)
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/mock/modified/g\" \$1" > ${TEMP}
  chmod +x ${TEMP}
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-json.json "${kube_flags[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  # Clean up
  kubectl delete service/modified "${kube_flags[@]}"
  kubectl delete rc/modified "${kube_flags[@]}"

  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-list.json "${kube_flags[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'modified:'
  # Clean up
  rm ${TEMP}
  kubectl delete service/modified "${kube_flags[@]}"
  kubectl delete rc/modified "${kube_flags[@]}"

  ## kubectl create --edit won't create anything if user makes no changes
  [ "$(EDITOR=cat kubectl create --edit -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -o json 2>&1 | grep 'Edit cancelled')" ]

  ## Create valid-pod POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## Patch can modify a local object
  kubectl patch --local -f pkg/kubectl/validation/testdata/v1/validPod.yaml --patch='{"spec": {"restartPolicy":"Never"}}' -o yaml | grep -q "Never"

  ## Patch fails with type restore error and exit code 1
  output_message=$(! kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"metadata":{"labels":"invalid"}}' 2>&1)
  kube::test::if_has_string "${output_message}" 'cannot restore map from string'

  ## Patch exits with error message "patched (no change)" and exit code 0 when no-op occurs
  output_message=$(kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"metadata":{"labels":{"name":"valid-pod"}}}' 2>&1)
  kube::test::if_has_string "${output_message}" 'patched (no change)'

  ## Patch pod can change image
  # Command
  kubectl patch "${kube_flags[@]}" pod valid-pod --record -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]}}'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  # Post-condition: valid-pod has the record annotation
  kube::test::get_object_assert pods "{{range.items}}{{$annotations_field}}:{{end}}" "${change_cause_annotation}"
  # prove that patch can use different types
  kubectl patch "${kube_flags[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx2"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx2:'
  # prove that patch can use different types
  kubectl patch "${kube_flags[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  # prove that yaml input works too
  YAML_PATCH=$'spec:\n  containers:\n  - name: kubernetes-serve-hostname\n    image: changed-with-yaml\n'
  kubectl patch "${kube_flags[@]}" pod valid-pod -p="${YAML_PATCH}"
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'changed-with-yaml:'
  ## Patch pod from JSON can change image
  # Command
  kubectl patch "${kube_flags[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "k8s.gcr.io/pause:3.1"}]}}'
  # Post-condition: valid-pod POD has expected image
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'k8s.gcr.io/pause:3.1:'

  ## If resourceVersion is specified in the patch, it will be treated as a precondition, i.e., if the resourceVersion is different from that is stored in the server, the Patch should be rejected
  ERROR_FILE="${KUBE_TEMP}/conflict-error"
  ## If the resourceVersion is the same as the one stored in the server, the patch will be applied.
  # Command
  # Needs to retry because other party may change the resource.
  for count in {0..3}; do
    resourceVersion=$(kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
    kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
    if grep -q "the object has been modified" "${ERROR_FILE}"; then
      kube::log::status "retry $1, error: $(cat ${ERROR_FILE})"
      rm "${ERROR_FILE}"
      sleep $((2**count))
    else
      rm "${ERROR_FILE}"
      kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
      break
    fi
  done

  ## If the resourceVersion is the different from the one stored in the server, the patch will be rejected.
  resourceVersion=$(kubectl get "${kube_flags[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
  ((resourceVersion+=100))
  # Command
  kubectl patch "${kube_flags[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
  # Post-condition: should get an error reporting the conflict
  if grep -q "please apply your changes to the latest version and try again" "${ERROR_FILE}"; then
    kube::log::status "\"kubectl patch with resourceVersion $resourceVersion\" returns error as expected: $(cat ${ERROR_FILE})"
  else
    kube::log::status "\"kubectl patch with resourceVersion $resourceVersion\" returns unexpected error or non-error: $(cat ${ERROR_FILE})"
    exit 1
  fi
  rm "${ERROR_FILE}"

  ## --force replace pod can change other field, e.g., spec.container.name
  # Command
  kubectl get "${kube_flags[@]}" pod valid-pod -o json | ${SED} 's/"kubernetes-serve-hostname"/"replaced-k8s-serve-hostname"/g' > /tmp/tmp-valid-pod.json
  kubectl replace "${kube_flags[@]}" --force -f /tmp/tmp-valid-pod.json
  # Post-condition: spec.container.name = "replaced-k8s-serve-hostname"
  kube::test::get_object_assert 'pod valid-pod' "{{(index .spec.containers 0).name}}" 'replaced-k8s-serve-hostname'

  ## check replace --grace-period requires --force
  output_message=$(! kubectl replace "${kube_flags[@]}" --grace-period=1 -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-grace-period must have \-\-force specified'

  ## check replace --timeout requires --force
  output_message=$(! kubectl replace "${kube_flags[@]}" --timeout=1s -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-timeout must have \-\-force specified'

  #cleaning
  rm /tmp/tmp-valid-pod.json

  ## replace of a cluster scoped resource can succeed
  # Pre-condition: a node exists
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-v1-test"
  }
}
__EOF__
  kubectl replace -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-v1-test",
    "annotations": {"a":"b"},
    "resourceVersion": "0"
  }
}
__EOF__

  # Post-condition: the node command succeeds
  kube::test::get_object_assert "node node-v1-test" "{{.metadata.annotations.a}}" 'b'
  kubectl delete node node-v1-test "${kube_flags[@]}"

  ## kubectl edit can update the image field of a POD. tmp-editor.sh is a fake editor
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/nginx/k8s.gcr.io\/serve_hostname/g\" \$1" > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  # Pre-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'nginx:'
  [[ "$(EDITOR=/tmp/tmp-editor.sh kubectl edit "${kube_flags[@]}" pods/valid-pod --output-patch=true | grep Patch:)" ]]
  # Post-condition: valid-pod POD has image k8s.gcr.io/serve_hostname
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'k8s.gcr.io/serve_hostname:'
  # cleaning
  rm /tmp/tmp-editor.sh

  ## kubectl edit should work on Windows
  [ "$(EDITOR=cat kubectl edit pod/valid-pod 2>&1 | grep 'Edit cancelled')" ]
  [ "$(EDITOR=cat kubectl edit pod/valid-pod | grep 'name: valid-pod')" ]
  [ "$(EDITOR=cat kubectl edit --windows-line-endings pod/valid-pod | file - | grep CRLF)" ]
  [ ! "$(EDITOR=cat kubectl edit --windows-line-endings=false pod/valid-pod | file - | grep CRLF)" ]
  [ "$(EDITOR=cat kubectl edit ns | grep 'kind: List')" ]

  ### Label POD YAML file locally without effecting the live pod.
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  output_message=$(kubectl label --local --overwrite -f hack/testdata/pod.yaml name=localonlyvalue -o yaml "${kube_flags[@]}")
  echo $output_message
  # Post-condition: name is still valid-pod in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  kube::test::if_has_string "${output_message}" "localonlyvalue"

  ### Overwriting an existing label is not permitted
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  ! kubectl label pods valid-pod name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is still valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'

  ### --overwrite must be used to overwrite existing label, can be applied to all resources
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod'
  # Command
  kubectl label --overwrite pods --all name=valid-pod-super-sayan "${kube_flags[@]}"
  # Post-condition: name is valid-pod-super-sayan
  kube::test::get_object_assert 'pod valid-pod' "{{${labels_field}.name}}" 'valid-pod-super-sayan'

  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod-super-sayan)' --grace-period=0 --force "${kube_flags[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two PODs from 1 yaml file
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete two PODs from 1 yaml file
  # Pre-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${kube_flags[@]}"
  # Post-condition: no PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  ## kubectl apply should update configuration annotations only if apply is already called
  ## 1. kubectl create doesn't set the annotation
  # Pre-Condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-label'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 2. kubectl replace doesn't set the annotation
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | ${SED} 's/test-pod-label/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 3. kubectl apply does set the annotation
  # Command: apply the pod "test-pod"
  kubectl apply -f hack/testdata/pod-apply.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is applied
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-applied'
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration
  ## 4. kubectl replace updates an existing annotation
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | ${SED} 's/test-pod-applied/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" has configuration annotation, and it's updated (different from the annotation when it's applied)
  [[ "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${kube_flags[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration-replaced
  ! [[ $(diff -q "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced > /dev/null) ]]
  # Clean up
  rm "${KUBE_TEMP}"/test-pod-replace.yaml "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced
  kubectl delete pods test-pod "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

# runs specific kubectl create tests
run_create_secret_tests() {
    set -o nounset
    set -o errexit

    ### Create generic secret with explicit namespace
    # Pre-condition: secret 'mysecret' does not exist
    output_message=$(! kubectl get secrets mysecret 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" 'secrets "mysecret" not found'
    # Command
    output_message=$(kubectl create "${kube_flags[@]}" secret generic mysecret --dry-run --from-literal=foo=bar -o jsonpath='{.metadata.namespace}' --namespace=user-specified)
    # Post-condition: mysecret still not created since --dry-run was used
    # Output from 'create' command should contain the specified --namespace value
    failure_message=$(! kubectl get secrets mysecret 2>&1 "${kube_flags[@]}")
    kube::test::if_has_string "${failure_message}" 'secrets "mysecret" not found'
    kube::test::if_has_string "${output_message}" 'user-specified'
    # Command
    output_message=$(kubectl create "${kube_flags[@]}" secret generic mysecret --dry-run --from-literal=foo=bar -o jsonpath='{.metadata.namespace}')
    # Post-condition: jsonpath for .metadata.namespace should be empty for object since --namespace was not explicitly specified
    kube::test::if_empty_string "${output_message}"

    kubectl create configmap tester-create-cm -o json --dry-run | kubectl create "${kube_flags[@]}" --raw /api/v1/namespaces/default/configmaps -f -
    kubectl delete -ndefault "${kube_flags[@]}" configmap tester-create-cm

    set +o nounset
    set +o errexit
}

run_secrets_test() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing secrets"

  # Ensure dry run succeeds and includes kind, apiVersion and data, and doesn't require a server connection
  output_message=$(kubectl create secret generic test --from-literal=key1=value1 --dry-run -o yaml --server=example.com --v=6)
  kube::test::if_has_string "${output_message}" 'kind: Secret'
  kube::test::if_has_string "${output_message}" 'apiVersion: v1'
  kube::test::if_has_string "${output_message}" 'key1: dmFsdWUx'
  kube::test::if_has_not_string "${output_message}" 'example.com'

  ### Create a new namespace
  # Pre-condition: the test-secrets namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-secrets\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-secrets
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-secrets' "{{$id_field}}" 'test-secrets'

  ### Create a generic secret in a specific namespace
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key1=value1 --type=test-type --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'test-type'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${kube_flags[@]}" | grep 'key1: dmFsdWUx')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a docker-registry secret in a specific namespace
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret docker-registry test-secret --docker-username=test-user --docker-password=test-password --docker-email='test-user@test.com' --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'kubernetes.io/dockerconfigjson'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${kube_flags[@]}" | grep '.dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ0ZXN0LXVzZXIiLCJwYXNzd29yZCI6InRlc3QtcGFzc3dvcmQiLCJlbWFpbCI6InRlc3QtdXNlckB0ZXN0LmNvbSIsImF1dGgiOiJkR1Z6ZEMxMWMyVnlPblJsYzNRdGNHRnpjM2R2Y21RPSJ9fX0=')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a tls secret
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create secret tls test-secret --namespace=test-secrets --key=hack/testdata/tls.key --cert=hack/testdata/tls.crt
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'kubernetes.io/tls'
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  # Command with process substitution
  kubectl create secret tls test-secret --namespace=test-secrets --key <(cat hack/testdata/tls.key) --cert <(cat hack/testdata/tls.crt)
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$id_field}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$secret_type}}" 'kubernetes.io/tls'
    # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  # Create a secret using stringData
  kubectl create --namespace=test-secrets -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Secret",
  "apiVersion": "v1",
  "metadata": {
    "name": "secret-string-data"
  },
  "data": {
    "k1":"djE=",
    "k2":""
  },
  "stringData": {
    "k2":"v2"
  }
}
__EOF__
  # Post-condition: secret-string-data secret is created with expected data, merged/overridden data from stringData, and a cleared stringData field
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.data}}' '.*k1:djE=.*'
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.data}}' '.*k2:djI=.*'
  kube::test::get_object_assert 'secret/secret-string-data --namespace=test-secrets ' '{{.stringData}}' '<no value>'
  # Clean up
  kubectl delete secret secret-string-data --namespace=test-secrets

  ### Create a secret using output flags
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  fi
  # Pre-condition: no secret exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  [[ "$(kubectl create secret generic test-secret --namespace=test-secrets --from-literal=key1=value1 --output=go-template --template=\"{{.metadata.name}}:\" | grep 'test-secret:')" ]]
  ## Clean-up
  kubectl delete secret test-secret --namespace=test-secrets
  # Clean up
  kubectl delete namespace test-secrets

  set +o nounset
  set +o errexit
}

run_service_accounts_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing service accounts"

  ### Create a new namespace
  # Pre-condition: the test-service-accounts namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"test-service-accounts\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-service-accounts
  # Post-condition: namespace 'test-service-accounts' is created.
  kube::test::get_object_assert 'namespaces/test-service-accounts' "{{$id_field}}" 'test-service-accounts'

  ### Create a service account in a specific namespace
  # Command
  kubectl create serviceaccount test-service-account --namespace=test-service-accounts
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'serviceaccount/test-service-account --namespace=test-service-accounts' "{{$id_field}}" 'test-service-account'
  # Clean-up
  kubectl delete serviceaccount test-service-account --namespace=test-service-accounts
  # Clean up
  kubectl delete namespace test-service-accounts

  set +o nounset
  set +o errexit
}

run_service_tests() {
  set -o nounset
  set -o errexit

  # switch back to the default namespace
  kubectl config set-context "${CONTEXT}" --namespace=""
  kube::log::status "Testing kubectl(v1:services)"

  ### Create redis-master service from JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml "${kube_flags[@]}"
  # Post-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Describe command should print detailed information
  kube::test::describe_object_assert services 'redis-master' "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert services 'redis-master'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert services 'redis-master' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert services 'redis-master' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert services "Name:" "Labels:" "Selector:" "IP:" "Port:" "Endpoints:" "Session Affinity:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert services
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert services false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert services true

  ### set selector
  # prove role=master
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"

  # Set selector of a local file without talking to the server
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan --local -o yaml "${kube_flags[@]}"
  ! kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan --dry-run -o yaml "${kube_flags[@]}"
  # Set command to change the selector.
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan
  # prove role=padawan
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "padawan:"
  # Set command to reset the selector back to the original one.
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml app=redis,role=master,tier=backend
  # prove role=master
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"
  # Show dry-run works on running selector
  kubectl set selector services redis-master role=padawan --dry-run -o yaml "${kube_flags[@]}"
  ! kubectl set selector services redis-master role=padawan --local -o yaml "${kube_flags[@]}"
  kube::test::get_object_assert 'services redis-master' "{{range$service_selector_field}}{{.}}:{{end}}" "redis:master:backend:"

  ### Dump current redis-master service
  output_service=$(kubectl get service redis-master -o json "${kube_flags[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  echo "${output_service}" | kubectl create -f - "${kube_flags[@]}"
  # Post-condition: redis-master service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'

  ### Create redis-master-v1-test service
  # Pre-condition: redis-master-service service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "Service",
  "apiVersion": "v1",
  "metadata": {
    "name": "service-v1-test"
  },
  "spec": {
    "ports": [
      {
        "protocol": "TCP",
        "port": 80,
        "targetPort": 80
      }
    ]
  }
}
__EOF__
  # Post-condition: service-v1-test service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'

  ### Identity
  kubectl get service "${kube_flags[@]}" service-v1-test -o json | kubectl replace "${kube_flags[@]}" -f -

  ### Delete services by id
  # Pre-condition: service-v1-test exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'
  # Command
  kubectl delete service redis-master "${kube_flags[@]}"
  kubectl delete service "service-v1-test" "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create two services
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-slave-service.yaml "${kube_flags[@]}"
  # Post-condition: redis-master and redis-slave services are created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:redis-slave:'

  ### Custom columns can be specified
  # Pre-condition: generate output using custom columns
  output_message=$(kubectl get services -o=custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion 2>&1 "${kube_flags[@]}")
  # Post-condition: should contain name column
  kube::test::if_has_string "${output_message}" 'redis-master'

  ### Delete multiple services at once
  # Pre-condition: redis-master and redis-slave services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:redis-master:redis-slave:'
  # Command
  kubectl delete services redis-master redis-slave "${kube_flags[@]}" # delete multiple services at once
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create an ExternalName service
  # Pre-condition: Only the default kubernetes service exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create service externalname beep-boop --external-name bar.com
  # Post-condition: beep-boop service is created
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'beep-boop:kubernetes:'

  ### Delete beep-boop service by id
  # Pre-condition: beep-boop service exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'beep-boop:kubernetes:'
  # Command
  kubectl delete service beep-boop "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'

  ### Create deployent and service
  # Pre-condition: no deployment exists
  kube::test::wait_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run testmetadata --image=nginx --replicas=2 --port=80 --expose --service-overrides='{ "metadata": { "annotations": { "zone-context": "home" } } } '
  # Check result
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'testmetadata:'
  kube::test::get_object_assert 'service testmetadata' "{{.metadata.annotations}}" "map\[zone-context:home\]"

  ### Expose deployment as a new service
  # Command
  kubectl expose deployment testmetadata  --port=1000 --target-port=80 --type=NodePort --name=exposemetadata --overrides='{ "metadata": { "annotations": { "zone-context": "work" } } } '
  # Check result
  kube::test::get_object_assert 'service exposemetadata' "{{.metadata.annotations}}" "map\[zone-context:work\]"

  # Clean-Up
  # Command
  kubectl delete service exposemetadata testmetadata "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'kubernetes:'
  fi
  kubectl delete deployment testmetadata "${kube_flags[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  fi

  set +o nounset
  set +o errexit
}

run_rc_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:replicationcontrollers)"

  ### Create and stop controller, make sure it doesn't leak pods
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kubectl delete rc frontend "${kube_flags[@]}"
  # Post-condition: no pods from frontend controller
  kube::test::get_object_assert 'pods -l "name=frontend"' "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create replication controller frontend from JSON
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Describe command should print detailed information
  kube::test::describe_object_assert rc 'frontend' "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:" "GET_HOSTS_FROM:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert rc 'frontend'
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert rc 'frontend' false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert rc 'frontend' true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert rc "Name:" "Name:" "Pod Template:" "Labels:" "Selector:" "Replicas:" "Pods Status:" "Volumes:" "GET_HOSTS_FROM:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert rc
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert rc false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert rc true

  ### Scale replication controller frontend with current-replicas and replicas
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Scale replication controller frontend with (wrong) current-replicas and replicas
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  ! kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: nothing changed
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'

  ### Scale replication controller frontend with replicas only
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Command
  kubectl scale  --replicas=3 replicationcontrollers frontend "${kube_flags[@]}"
  # Post-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'

  ### Scale replication controller from JSON with replicas only
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl scale  --replicas=2 -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '2'
  # Clean-up
  kubectl delete rc frontend "${kube_flags[@]}"

  ### Scale multiple replication controllers
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-master-controller.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-slave-controller.yaml "${kube_flags[@]}"
  # Command
  kubectl scale rc/redis-master rc/redis-slave --replicas=4 "${kube_flags[@]}"
  # Post-condition: 4 replicas each
  kube::test::get_object_assert 'rc redis-master' "{{$rc_replicas_field}}" '4'
  kube::test::get_object_assert 'rc redis-slave' "{{$rc_replicas_field}}" '4'
  # Clean-up
  kubectl delete rc redis-{master,slave} "${kube_flags[@]}"

  ### Scale a job
  kubectl create -f test/fixtures/doc-yaml/user-guide/job.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --replicas=2 job/pi
  # Post-condition: 2 replicas for pi
  kube::test::get_object_assert 'job pi' "{{$job_parallelism_field}}" '2'
  # Clean-up
  kubectl delete job/pi "${kube_flags[@]}"

  ### Scale a deployment
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  # Command
  kubectl scale --current-replicas=3 --replicas=1 deployment/nginx-deployment
  # Post-condition: 1 replica for nginx-deployment
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$deployment_replicas}}" '1'
  # Clean-up
  kubectl delete deployment/nginx-deployment "${kube_flags[@]}"

  ### Expose deployments by creating a service
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1deployment.yaml --port 80 2>&1 "${kube_flags[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${kube_flags[@]}"
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta2deployment.yaml --port 80 2>&1 "${kube_flags[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${kube_flags[@]}"
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta1deployment.yaml --port 80 2>&1 "${kube_flags[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${kube_flags[@]}"
  # Contains no selectors, should fail.
  output_message=$(! kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1deployment-no-selectors.yaml --port 80 2>&1 "${kube_flags[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'invalid deployment: no selectors'
  # Contains no selectors, should fail.
  output_message=$(! kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta2deployment-no-selectors.yaml --port 80 2>&1 "${kube_flags[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'invalid deployment: no selectors'

  ### Expose a deployment as a service
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${kube_flags[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$deployment_replicas}}" '3'
  # Command
  kubectl expose deployment/nginx-deployment
  # Post-condition: service exists and exposes deployment port (80)
  kube::test::get_object_assert 'service nginx-deployment' "{{$port_field}}" '80'
  # Clean-up
  kubectl delete deployment/nginx-deployment service/nginx-deployment "${kube_flags[@]}"

  ### Expose replication controller as service
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$rc_replicas_field}}" '3'
  # Command
  kubectl expose rc frontend --port=80 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend' "{{$port_name}} {{$port_field}}" '<no value> 80'
  # Command
  kubectl expose service frontend --port=443 --name=frontend-2 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-2' "{{$port_name}} {{$port_field}}" '<no value> 443'
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${kube_flags[@]}"
  kubectl expose pod valid-pod --port=444 --name=frontend-3 "${kube_flags[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-3' "{{$port_name}} {{$port_field}}" '<no value> 444'
  # Create a service using service/v1 generator
  kubectl expose rc frontend --port=80 --name=frontend-4 --generator=service/v1 "${kube_flags[@]}"
  # Post-condition: service exists and the port is named default.
  kube::test::get_object_assert 'service frontend-4' "{{$port_name}} {{$port_field}}" 'default 80'
  # Verify that expose service works without specifying a port.
  kubectl expose service frontend --name=frontend-5 "${kube_flags[@]}"
  # Post-condition: service exists with the same port as the original service.
  kube::test::get_object_assert 'service frontend-5' "{{$port_field}}" '80'
  # Cleanup services
  kubectl delete pod valid-pod "${kube_flags[@]}"
  kubectl delete service frontend{,-2,-3,-4,-5} "${kube_flags[@]}"

  ### Expose negative invalid resource test
  # Pre-condition: don't need
  # Command
  output_message=$(! kubectl expose nodes 127.0.0.1 2>&1 "${kube_flags[@]}")
  # Post-condition: the error message has "cannot expose" string
  kube::test::if_has_string "${output_message}" 'cannot expose'

  ### Try to generate a service with invalid name (exceeding maximum valid size)
  # Pre-condition: use --name flag
  output_message=$(! kubectl expose -f hack/testdata/pod-with-large-name.yaml --name=invalid-large-service-name-that-has-more-than-sixty-three-characters --port=8081 2>&1 "${kube_flags[@]}")
  # Post-condition: should fail due to invalid name
  kube::test::if_has_string "${output_message}" 'metadata.name: Invalid value'
  # Pre-condition: default run without --name flag; should succeed by truncating the inherited name
  output_message=$(kubectl expose -f hack/testdata/pod-with-large-name.yaml --port=8081 2>&1 "${kube_flags[@]}")
  # Post-condition: inherited name from pod has been truncated
  kube::test::if_has_string "${output_message}" 'kubernetes-serve-hostname-testing-sixty-three-characters-in-len exposed'
  # Clean-up
  kubectl delete svc kubernetes-serve-hostname-testing-sixty-three-characters-in-len "${kube_flags[@]}"

  ### Expose multiport object as a new service
  # Pre-condition: don't use --port flag
  output_message=$(kubectl expose -f test/fixtures/doc-yaml/admin/high-availability/etcd.yaml --selector=test=etcd 2>&1 "${kube_flags[@]}")
  # Post-condition: expose succeeded
  kube::test::if_has_string "${output_message}" 'etcd-server exposed'
  # Post-condition: generated service has both ports from the exposed pod
  kube::test::get_object_assert 'service etcd-server' "{{$port_name}} {{$port_field}}" 'port-1 2380'
  kube::test::get_object_assert 'service etcd-server' "{{$second_port_name}} {{$second_port_field}}" 'port-2 2379'
  # Clean-up
  kubectl delete svc etcd-server "${kube_flags[@]}"

  ### Delete replication controller with id
  # Pre-condition: frontend replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # Command
  kubectl delete rc frontend "${kube_flags[@]}"
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Create two replication controllers
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-slave-controller.yaml "${kube_flags[@]}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple controllers at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl delete rc frontend redis-slave "${kube_flags[@]}" # delete multiple controllers at once
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  ### Auto scale replication controller
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'frontend:'
  # autoscale 1~2 pods, CPU utilization 70%, rc specified by file
  kubectl autoscale -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}" --max=2 --cpu-percent=70
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 70'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale 2~3 pods, no CPU utilization specified, rc specified by name
  kubectl autoscale rc frontend "${kube_flags[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa frontend' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '2 3 80'
  kubectl delete hpa frontend "${kube_flags[@]}"
  # autoscale without specifying --max should fail
  ! kubectl autoscale rc frontend "${kube_flags[@]}"
  # Clean up
  kubectl delete rc frontend "${kube_flags[@]}"

  ## Set resource limits/request of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Set resources of a local file without talking to the server
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --local -o yaml "${kube_flags[@]}"
  ! kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --dry-run -o yaml "${kube_flags[@]}"
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer-resources.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx-deployment-resources:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field1}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's cpu limits
  kubectl set resources deployment nginx-deployment-resources --limits=cpu=100m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "100m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set a non-existing container should fail
  ! kubectl set resources deployment nginx-deployment-resources -c=redis --limits=cpu=100m
  # Set the limit of a specific container in deployment
  kubectl set resources deployment nginx-deployment-resources -c=nginx --limits=cpu=200m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set limits/requests of a deployment specified by a file
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Show dry-run works on running deployments
  kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --dry-run -o yaml "${kube_flags[@]}"
  ! kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --local -o yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Clean up
  kubectl delete deployment nginx-deployment-resources "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_namespace_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:namespaces)"
  ### Create a new namespace
  # Pre-condition: only the "default" namespace exists
  # The Pre-condition doesn't hold anymore after we create and switch namespaces before creating pods with same name in the test.
  # kube::test::get_object_assert namespaces "{{range.items}}{{$id_field}}:{{end}}" 'default:'
  # Command
  kubectl create namespace my-namespace
  # Post-condition: namespace 'my-namespace' is created.
  kube::test::get_object_assert 'namespaces/my-namespace' "{{$id_field}}" 'my-namespace'
  # Clean up
  kubectl delete namespace my-namespace --wait=false
  # make sure that wait properly waits for finalization
  kubectl wait --for=delete ns/my-namespace
  output_message=$(! kubectl get ns/my-namespace 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" ' not found'

  ######################
  # Pods in Namespaces #
  ######################

  if kube::test::if_supports_resource "${pods}" ; then
    ### Create a new namespace
    # Pre-condition: the other namespace does not exist
    kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $id_field \"other\" }}found{{end}}{{end}}:' ':'
    # Command
    kubectl create namespace other
    # Post-condition: namespace 'other' is created.
    kube::test::get_object_assert 'namespaces/other' "{{$id_field}}" 'other'

    ### Create POD valid-pod in specific namespace
    # Pre-condition: no POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create "${kube_flags[@]}" --namespace=other -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
    # Post-condition: valid-pod POD is created
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Post-condition: verify shorthand `-n other` has the same results as `--namespace=other`
    kube::test::get_object_assert 'pods -n other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Post-condition: a resource cannot be retrieved by name across all namespaces
    output_message=$(! kubectl get "${kube_flags[@]}" pod valid-pod --all-namespaces 2>&1)
    kube::test::if_has_string "${output_message}" "a resource cannot be retrieved by name across all namespaces"

    ### Delete POD valid-pod in specific namespace
    # Pre-condition: valid-pod POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'
    # Command
    kubectl delete "${kube_flags[@]}" pod --namespace=other valid-pod --grace-period=0 --force
    # Post-condition: valid-pod POD doesn't exist
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$id_field}}:{{end}}" ''
    # Clean up
    kubectl delete namespace other
  fi

  set +o nounset
  set +o errexit
}

run_nodes_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:nodes)"

  kube::test::get_object_assert nodes "{{range.items}}{{$id_field}}:{{end}}" '127.0.0.1:'

  kube::test::describe_object_assert nodes "127.0.0.1" "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  # Describe command should print events information by default
  kube::test::describe_object_events_assert nodes "127.0.0.1"
  # Describe command should not print events information when show-events=false
  kube::test::describe_object_events_assert nodes "127.0.0.1" false
  # Describe command should print events information when show-events=true
  kube::test::describe_object_events_assert nodes "127.0.0.1" true
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert nodes "Name:" "Labels:" "CreationTimestamp:" "Conditions:" "Addresses:" "Capacity:" "Pods:"
  # Describe command should print events information by default
  kube::test::describe_resource_events_assert nodes
  # Describe command should not print events information when show-events=false
  kube::test::describe_resource_events_assert nodes false
  # Describe command should print events information when show-events=true
  kube::test::describe_resource_events_assert nodes true

  ### kubectl patch update can mark node unschedulable
  # Pre-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'
  kubectl patch "${kube_flags[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":true}}'
  # Post-condition: node is unschedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" 'true'
  kubectl patch "${kube_flags[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":null}}'
  # Post-condition: node is schedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" '<no value>'

  # check webhook token authentication endpoint, kubectl doesn't actually display the returned object so this isn't super useful
  # but it proves that works
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/tokenreview-v1beta1.json --validate=false
  kubectl create -f test/fixtures/pkg/kubectl/cmd/create/tokenreview-v1.json --validate=false

  set +o nounset
  set +o errexit
}

run_pod_templates_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing pod templates"

  ### Create PODTEMPLATE
  # Pre-condition: no PODTEMPLATE
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/walkthrough/podtemplate.json "${kube_flags[@]}"
  # Post-condition: nginx PODTEMPLATE is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'

  ### Printing pod templates works
  kubectl get podtemplates "${kube_flags[@]}"
  [[ "$(kubectl get podtemplates -o yaml "${kube_flags[@]}" | grep nginx)" ]]

  ### Delete nginx pod template by name
  # Pre-condition: nginx pod template is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'
  # Command
  kubectl delete podtemplate nginx "${kube_flags[@]}"
  # Post-condition: No templates exist
  kube::test::get_object_assert podtemplate "{{range.items}}{{.metadata.name}}:{{end}}" ''

  set +o nounset
  set +o errexit
}
