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
  kube::test::get_object_assert 'configmap/test-configmap' "{{$ID_FIELD}}" 'test-configmap'
  kubectl delete configmap test-configmap "${KUBE_FLAGS[@]}"

  ### Create a new namespace
  # Pre-condition: the test-configmaps namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $ID_FIELD \"test-configmaps\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-configmaps
  # Post-condition: namespace 'test-configmaps' is created.
  kube::test::get_object_assert 'namespaces/test-configmaps' "{{$ID_FIELD}}" 'test-configmaps'

  ### Create a generic configmap in a specific namespace
  # Pre-condition: configmap test-configmap and test-binary-configmap does not exist
  kube::test::get_object_assert 'configmaps' '{{range.items}}{{ if eq $ID_FIELD \"test-configmap\" }}found{{end}}{{end}}:' ':'
  kube::test::get_object_assert 'configmaps' '{{range.items}}{{ if eq $ID_FIELD \"test-binary-configmap\" }}found{{end}}{{end}}:' ':'

  # Command
  kubectl create configmap test-configmap --from-literal=key1=value1 --namespace=test-configmaps
  kubectl create configmap test-binary-configmap --from-file <( head -c 256 /dev/urandom ) --namespace=test-configmaps
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-configmaps' "{{$ID_FIELD}}" 'test-configmap'
  kube::test::get_object_assert 'configmap/test-binary-configmap --namespace=test-configmaps' "{{$ID_FIELD}}" 'test-binary-configmap'
  [[ "$(kubectl get configmap/test-configmap --namespace=test-configmaps -o yaml "${KUBE_FLAGS[@]}" | grep 'key1: value1')" ]]
  [[ "$(kubectl get configmap/test-binary-configmap --namespace=test-configmaps -o yaml "${KUBE_FLAGS[@]}" | grep 'binaryData')" ]]
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
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create "${KUBE_FLAGS[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${KUBE_FLAGS[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  kube::test::get_object_assert 'pod valid-pod' "{{$ID_FIELD}}" 'valid-pod'
  kube::test::get_object_assert 'pod/valid-pod' "{{$ID_FIELD}}" 'valid-pod'
  kube::test::get_object_assert 'pods/valid-pod' "{{$ID_FIELD}}" 'valid-pod'
  # Repeat above test using jsonpath template
  kube::test::get_object_jsonpath_assert pods "{.items[*]$ID_FIELD}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod valid-pod' "{$ID_FIELD}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pod/valid-pod' "{$ID_FIELD}" 'valid-pod'
  kube::test::get_object_jsonpath_assert 'pods/valid-pod' "{$ID_FIELD}" 'valid-pod'
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
  output_pod=$(kubectl get pod valid-pod -o yaml "${KUBE_FLAGS[@]}")

  ### Delete POD valid-pod by id
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${KUBE_FLAGS[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Delete POD valid-pod by id with --now
  # Pre-condition: valid-pod POD exists
  kubectl create "${KUBE_FLAGS[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pod valid-pod "${KUBE_FLAGS[@]}" --now
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Delete POD valid-pod by id with --grace-period=0
  # Pre-condition: valid-pod POD exists
  kubectl create "${KUBE_FLAGS[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command succeeds without --force by waiting
  kubectl delete pod valid-pod "${KUBE_FLAGS[@]}" --grace-period=0
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create POD valid-pod from dumped YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  echo "${output_pod}" | ${SED} '/namespace:/d' | kubectl create -f - "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod from JSON
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create POD valid-pod from JSON
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ### Delete POD valid-pod with label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$ID_FIELD}}:{{end}}' 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod)' "${KUBE_FLAGS[@]}" --grace-period=0 --force
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$ID_FIELD}}:{{end}}' ''

  ### Create POD valid-pod from YAML
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  output_message=$(kubectl get pods --field-selector metadata.name=valid-pod "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" "valid-pod"
  # Command
  phase=$(kubectl get "${KUBE_FLAGS[@]}" pod valid-pod -o go-template='{{ .status.phase }}')
  output_message=$(kubectl get pods --field-selector status.phase="${phase}" "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" "valid-pod"

  ### Delete PODs with no parameter mustn't kill everything
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete pods "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ### Delete PODs with --all and a label selector is not permitted
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  ! kubectl delete --all pods -l'name in (valid-pod)' "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ### Delete all PODs
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete --all pods "${KUBE_FLAGS[@]}" --grace-period=0 --force # --all remove all the pods
  # Post-condition: no POD exists
  kube::test::get_object_assert "pods -l'name in (valid-pod)'" '{{range.items}}{{$ID_FIELD}}:{{end}}' ''

  # Detailed tests for describe pod output
    ### Create a new namespace
  # Pre-condition: the test-secrets namespace does not exist
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $ID_FIELD \"test-kubectl-describe-pod\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-kubectl-describe-pod
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-kubectl-describe-pod' "{{$ID_FIELD}}" 'test-kubectl-describe-pod'

  ### Create a generic secret
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-kubectl-describe-pod' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key-1=value1 --type=test-type --namespace=test-kubectl-describe-pod
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$ID_FIELD}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-kubectl-describe-pod' "{{$SECRET_TYPE}}" 'test-type'

  ### Create a generic configmap
  # Pre-condition: CONFIGMAP test-configmap does not exist
  #kube::test::get_object_assert 'configmap/test-configmap --namespace=test-kubectl-describe-pod' "{{$ID_FIELD}}" ''
  kube::test::get_object_assert 'configmaps --namespace=test-kubectl-describe-pod' '{{range.items}}{{ if eq $ID_FIELD \"test-configmap\" }}found{{end}}{{end}}:' ':'

  #kube::test::get_object_assert 'configmaps --namespace=test-kubectl-describe-pod' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create configmap test-configmap --from-literal=key-2=value2 --namespace=test-kubectl-describe-pod
  # Post-condition: configmap exists and has expected values
  kube::test::get_object_assert 'configmap/test-configmap --namespace=test-kubectl-describe-pod' "{{$ID_FIELD}}" 'test-configmap'

  ### Create a pod disruption budget with minAvailable
  # Command
  kubectl create pdb test-pdb-1 --selector=app=rails --min-available=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-1 --namespace=test-kubectl-describe-pod' "{{$PDB_MIN_AVAILABLE}}" '2'
  # Command
  kubectl create pdb test-pdb-2 --selector=app=rails --min-available=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-2 --namespace=test-kubectl-describe-pod' "{{$PDB_MIN_AVAILABLE}}" '50%'

  ### Create a pod disruption budget with maxUnavailable
  # Command
  kubectl create pdb test-pdb-3 --selector=app=rails --max-unavailable=2 --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-3 --namespace=test-kubectl-describe-pod' "{{$PDB_MAX_UNAVAILABLE}}" '2'
  # Command
  kubectl create pdb test-pdb-4 --selector=app=rails --max-unavailable=50% --namespace=test-kubectl-describe-pod
  # Post-condition: pdb exists and has expected values
  kube::test::get_object_assert 'pdb/test-pdb-4 --namespace=test-kubectl-describe-pod' "{{$PDB_MAX_UNAVAILABLE}}" '50%'

  ### Fail creating a pod disruption budget if both maxUnavailable and minAvailable specified
  ! kubectl create pdb test-pdb --selector=app=rails --min-available=2 --max-unavailable=3 --namespace=test-kubectl-describe-pod

  # Create a pod that consumes secret, configmap, and downward API keys as envs
  kube::test::get_object_assert 'pods --namespace=test-kubectl-describe-pod' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
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
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  kubectl create -f test/e2e/testing-manifests/kubectl/redis-master-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod and redis-master PODs are created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete multiple PODs at once
  # Pre-condition: valid-pod and redis-master PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete pods valid-pod redis-master "${KUBE_FLAGS[@]}" --grace-period=0 --force # delete multiple pods at once
  # Post-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create valid-pod POD
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ### Label the valid-pod POD
  # Pre-condition: valid-pod is not labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$LABELS_FIELD}}{{.}}:{{end}}" 'valid-pod:'
  # Command
  kubectl label pods valid-pod new-name=new-valid-pod "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod is labelled
  kube::test::get_object_assert 'pod valid-pod' "{{range$LABELS_FIELD}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'

  ### Label the valid-pod POD with empty label value
  # Pre-condition: valid-pod does not have label "emptylabel"
  kube::test::get_object_assert 'pod valid-pod' "{{range$LABELS_FIELD}}{{.}}:{{end}}" 'valid-pod:new-valid-pod:'
  # Command
  kubectl label pods valid-pod emptylabel="" "${KUBE_FLAGS[@]}"
  # Post-condition: valid pod contains "emptylabel" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.emptylabel}}" ''

  ### Annotate the valid-pod POD with empty annotation value
  # Pre-condition: valid-pod does not have annotation "emptyannotation"
  kube::test::get_object_assert 'pod valid-pod' "{{${ANNOTATIONS_FIELD}.emptyannotation}}" '<no value>'
  # Command
  kubectl annotate pods valid-pod emptyannotation="" "${KUBE_FLAGS[@]}"
  # Post-condition: valid pod contains "emptyannotation" with no value
  kube::test::get_object_assert 'pod valid-pod' "{{${ANNOTATIONS_FIELD}.emptyannotation}}" ''

  ### Record label change
  # Pre-condition: valid-pod does not have record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range.items}}{{$ANNOTATIONS_FIELD}}:{{end}}" ''
  # Command
  kubectl label pods valid-pod record-change=true --record=true "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod has record annotation
  kube::test::get_object_assert 'pod valid-pod' "{{range$ANNOTATIONS_FIELD}}{{.}}:{{end}}" ".*--record=true.*"

  ### Do not record label change
  # Command
  kubectl label pods valid-pod no-record-change=true --record=false "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod's record annotation still contains command with --record=true
  kube::test::get_object_assert 'pod valid-pod' "{{range$ANNOTATIONS_FIELD}}{{.}}:{{end}}" ".*--record=true.*"

  ### Record label change with specified flag and previous change already recorded
  ### we are no longer tricked by data from another user into revealing more information about our client
  # Command
  kubectl label pods valid-pod new-record-change=true --record=true "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod's record annotation contains new change
  kube::test::get_object_assert 'pod valid-pod' "{{range$ANNOTATIONS_FIELD}}{{.}}:{{end}}" ".*new-record-change=true.*"


  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -lnew-name=new-valid-pod --grace-period=0 --force "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create pod-with-precision POD
  # Pre-condition: no POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/pod-with-precision.json "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is running
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'pod-with-precision:'

  ## Patch preserves precision
  # Command
  kubectl patch "${KUBE_FLAGS[@]}" pod pod-with-precision -p='{"metadata":{"annotations":{"patchkey": "patchvalue"}}}'
  # Post-condition: pod-with-precision POD has patched annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${ANNOTATIONS_FIELD}.patchkey}}" 'patchvalue'
  # Command
  kubectl label pods pod-with-precision labelkey=labelvalue "${KUBE_FLAGS[@]}"
  # Post-condition: pod-with-precision POD has label
  kube::test::get_object_assert 'pod pod-with-precision' "{{${LABELS_FIELD}.labelkey}}" 'labelvalue'
  # Command
  kubectl annotate pods pod-with-precision annotatekey=annotatevalue "${KUBE_FLAGS[@]}"
  # Post-condition: pod-with-precision POD has annotation
  kube::test::get_object_assert 'pod pod-with-precision' "{{${ANNOTATIONS_FIELD}.annotatekey}}" 'annotatevalue'
  # Cleanup
  kubectl delete pod pod-with-precision "${KUBE_FLAGS[@]}"

  ### Annotate POD YAML file locally without effecting the live pod.
  kubectl create -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  # Command
  kubectl annotate -f hack/testdata/pod.yaml annotatekey=annotatevalue "${KUBE_FLAGS[@]}"

  # Pre-condition: annotationkey is annotationvalue
  kube::test::get_object_assert 'pod test-pod' "{{${ANNOTATIONS_FIELD}.annotatekey}}" 'annotatevalue'

  # Command
  output_message=$(kubectl annotate --local -f hack/testdata/pod.yaml annotatekey=localvalue -o yaml "${KUBE_FLAGS[@]}")
  echo $output_message

  # Post-condition: annotationkey is still annotationvalue in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod test-pod' "{{${ANNOTATIONS_FIELD}.annotatekey}}" 'annotatevalue'
  kube::test::if_has_string "${output_message}" "localvalue"

  # Cleanup
  kubectl delete -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"

  ### Create valid-pod POD
  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  ## kubectl create --edit can update the label filed of multiple resources. tmp-editor.sh is a fake editor
  TEMP=$(mktemp /tmp/tmp-editor-XXXXXXXX.sh)
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/mock/modified/g\" \$1" > ${TEMP}
  chmod +x ${TEMP}
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-json.json "${KUBE_FLAGS[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$ID_FIELD}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'modified:'
  # Clean up
  kubectl delete service/modified "${KUBE_FLAGS[@]}"
  kubectl delete rc/modified "${KUBE_FLAGS[@]}"

  # Pre-condition: no services and no rcs exist
  kube::test::get_object_assert service "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  EDITOR=${TEMP} kubectl create --edit -f hack/testdata/multi-resource-list.json "${KUBE_FLAGS[@]}"
  # Post-condition: service named modified and rc named modified are created
  kube::test::get_object_assert service "{{range.items}}{{$ID_FIELD}}:{{end}}" 'modified:'
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'modified:'
  # Clean up
  rm ${TEMP}
  kubectl delete service/modified "${KUBE_FLAGS[@]}"
  kubectl delete rc/modified "${KUBE_FLAGS[@]}"

  ## kubectl create --edit won't create anything if user makes no changes
  [ "$(EDITOR=cat kubectl create --edit -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -o json 2>&1 | grep 'Edit cancelled')" ]

  ## Create valid-pod POD
  # Pre-condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD is created
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'

  ## Patch can modify a local object
  kubectl patch --local -f pkg/kubectl/validation/testdata/v1/validPod.yaml --patch='{"spec": {"restartPolicy":"Never"}}' -o yaml | grep -q "Never"

  ## Patch fails with type restore error and exit code 1
  output_message=$(! kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod -p='{"metadata":{"labels":"invalid"}}' 2>&1)
  kube::test::if_has_string "${output_message}" 'cannot restore map from string'

  ## Patch exits with error message "patched (no change)" and exit code 0 when no-op occurs
  output_message=$(kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod -p='{"metadata":{"labels":{"name":"valid-pod"}}}' 2>&1)
  kube::test::if_has_string "${output_message}" 'patched (no change)'

  ## Patch pod can change image
  # Command
  kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod --record -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]}}'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'nginx:'
  # Post-condition: valid-pod has the record annotation
  kube::test::get_object_assert pods "{{range.items}}{{$ANNOTATIONS_FIELD}}:{{end}}" "${CHANGE_CAUSE_ANNOTATION}"
  # prove that patch can use different types
  kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx2"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'nginx2:'
  # prove that patch can use different types
  kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod --type="json" -p='[{"op": "replace", "path": "/spec/containers/0/image", "value":"nginx"}]'
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'nginx:'
  # prove that yaml input works too
  YAML_PATCH=$'spec:\n  containers:\n  - name: kubernetes-serve-hostname\n    image: changed-with-yaml\n'
  kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod -p="${YAML_PATCH}"
  # Post-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'changed-with-yaml:'
  ## Patch pod from JSON can change image
  # Command
  kubectl patch "${KUBE_FLAGS[@]}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "k8s.gcr.io/pause:3.1"}]}}'
  # Post-condition: valid-pod POD has expected image
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'k8s.gcr.io/pause:3.1:'

  ## If resourceVersion is specified in the patch, it will be treated as a precondition, i.e., if the resourceVersion is different from that is stored in the server, the Patch should be rejected
  ERROR_FILE="${KUBE_TEMP}/conflict-error"
  ## If the resourceVersion is the same as the one stored in the server, the patch will be applied.
  # Command
  # Needs to retry because other party may change the resource.
  for count in {0..3}; do
    resourceVersion=$(kubectl get "${KUBE_FLAGS[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
    kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
    if grep -q "the object has been modified" "${ERROR_FILE}"; then
      kube::log::status "retry $1, error: $(cat ${ERROR_FILE})"
      rm "${ERROR_FILE}"
      sleep $((2**count))
    else
      rm "${ERROR_FILE}"
      kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'nginx:'
      break
    fi
  done

  ## If the resourceVersion is the different from the one stored in the server, the patch will be rejected.
  resourceVersion=$(kubectl get "${KUBE_FLAGS[@]}" pod valid-pod -o go-template='{{ .metadata.resourceVersion }}')
  ((resourceVersion+=100))
  # Command
  kubectl patch "${KUBE_FLAGS[@]}" pod valid-pod -p='{"spec":{"containers":[{"name": "kubernetes-serve-hostname", "image": "nginx"}]},"metadata":{"resourceVersion":"'$resourceVersion'"}}' 2> "${ERROR_FILE}" || true
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
  kubectl get "${KUBE_FLAGS[@]}" pod valid-pod -o json | ${SED} 's/"kubernetes-serve-hostname"/"replaced-k8s-serve-hostname"/g' > /tmp/tmp-valid-pod.json
  kubectl replace "${KUBE_FLAGS[@]}" --force -f /tmp/tmp-valid-pod.json
  # Post-condition: spec.container.name = "replaced-k8s-serve-hostname"
  kube::test::get_object_assert 'pod valid-pod' "{{(index .spec.containers 0).name}}" 'replaced-k8s-serve-hostname'

  ## check replace --grace-period requires --force
  output_message=$(! kubectl replace "${KUBE_FLAGS[@]}" --grace-period=1 -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-grace-period must have \-\-force specified'

  ## check replace --timeout requires --force
  output_message=$(! kubectl replace "${KUBE_FLAGS[@]}" --timeout=1s -f /tmp/tmp-valid-pod.json 2>&1)
  kube::test::if_has_string "${output_message}" '\-\-timeout must have \-\-force specified'

  #cleaning
  rm /tmp/tmp-valid-pod.json

  ## replace of a cluster scoped resource can succeed
  # Pre-condition: a node exists
  kubectl create -f - "${KUBE_FLAGS[@]}" << __EOF__
{
  "kind": "Node",
  "apiVersion": "v1",
  "metadata": {
    "name": "node-v1-test"
  }
}
__EOF__
  kubectl replace -f - "${KUBE_FLAGS[@]}" << __EOF__
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
  kubectl delete node node-v1-test "${KUBE_FLAGS[@]}"

  ## kubectl edit can update the image field of a POD. tmp-editor.sh is a fake editor
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/nginx/k8s.gcr.io\/serve_hostname/g\" \$1" > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  # Pre-condition: valid-pod POD has image nginx
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'nginx:'
  [[ "$(EDITOR=/tmp/tmp-editor.sh kubectl edit "${KUBE_FLAGS[@]}" pods/valid-pod --output-patch=true | grep Patch:)" ]]
  # Post-condition: valid-pod POD has image k8s.gcr.io/serve_hostname
  kube::test::get_object_assert pods "{{range.items}}{{$IMAGE_FIELD}}:{{end}}" 'k8s.gcr.io/serve_hostname:'
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
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod'
  # Command
  output_message=$(kubectl label --local --overwrite -f hack/testdata/pod.yaml name=localonlyvalue -o yaml "${KUBE_FLAGS[@]}")
  echo $output_message
  # Post-condition: name is still valid-pod in the live pod, but command output is the new value
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod'
  kube::test::if_has_string "${output_message}" "localonlyvalue"

  ### Overwriting an existing label is not permitted
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod'
  # Command
  ! kubectl label pods valid-pod name=valid-pod-super-sayan "${KUBE_FLAGS[@]}"
  # Post-condition: name is still valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod'

  ### --overwrite must be used to overwrite existing label, can be applied to all resources
  # Pre-condition: name is valid-pod
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod'
  # Command
  kubectl label --overwrite pods --all name=valid-pod-super-sayan "${KUBE_FLAGS[@]}"
  # Post-condition: name is valid-pod-super-sayan
  kube::test::get_object_assert 'pod valid-pod' "{{${LABELS_FIELD}.name}}" 'valid-pod-super-sayan'

  ### Delete POD by label
  # Pre-condition: valid-pod POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
  # Command
  kubectl delete pods -l'name in (valid-pod-super-sayan)' --grace-period=0 --force "${KUBE_FLAGS[@]}"
  # Post-condition: valid-pod POD doesn't exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create two PODs from 1 yaml file
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'redis-master:valid-pod:'

  ### Delete two PODs from 1 yaml file
  # Pre-condition: redis-master and valid-pod PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" 'redis-master:valid-pod:'
  # Command
  kubectl delete -f test/fixtures/doc-yaml/user-guide/multi-pod.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: no PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ## kubectl apply should update configuration annotations only if apply is already called
  ## 1. kubectl create doesn't set the annotation
  # Pre-Condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${LABELS_FIELD}.name}}" 'test-pod-label'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 2. kubectl replace doesn't set the annotation
  kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | ${SED} 's/test-pod-label/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${LABELS_FIELD}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" doesn't have configuration annotation
  ! [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 3. kubectl apply does set the annotation
  # Command: apply the pod "test-pod"
  kubectl apply -f hack/testdata/pod-apply.yaml "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" is applied
  kube::test::get_object_assert 'pods test-pod' "{{${LABELS_FIELD}.name}}" 'test-pod-applied'
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration
  ## 4. kubectl replace updates an existing annotation
  kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | ${SED} 's/test-pod-applied/test-pod-replaced/g' > "${KUBE_TEMP}"/test-pod-replace.yaml
  # Command: replace the pod "test-pod"
  kubectl replace -f "${KUBE_TEMP}"/test-pod-replace.yaml "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" is replaced
  kube::test::get_object_assert 'pods test-pod' "{{${LABELS_FIELD}.name}}" 'test-pod-replaced'
  # Post-Condition: pod "test-pod" has configuration annotation, and it's updated (different from the annotation when it's applied)
  [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration > "${KUBE_TEMP}"/annotation-configuration-replaced
  ! [[ $(diff -q "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced > /dev/null) ]]
  # Clean up
  rm "${KUBE_TEMP}"/test-pod-replace.yaml "${KUBE_TEMP}"/annotation-configuration "${KUBE_TEMP}"/annotation-configuration-replaced
  kubectl delete pods test-pod "${KUBE_FLAGS[@]}"

  set +o nounset
  set +o errexit
}

# runs specific kubectl create tests
run_create_secret_tests() {
    set -o nounset
    set -o errexit

    ### Create generic secret with explicit namespace
    # Pre-condition: secret 'mysecret' does not exist
    output_message=$(! kubectl get secrets mysecret 2>&1 "${KUBE_FLAGS[@]}")
    kube::test::if_has_string "${output_message}" 'secrets "mysecret" not found'
    # Command
    output_message=$(kubectl create "${KUBE_FLAGS[@]}" secret generic mysecret --dry-run --from-literal=foo=bar -o jsonpath='{.metadata.namespace}' --namespace=user-specified)
    # Post-condition: mysecret still not created since --dry-run was used
    # Output from 'create' command should contain the specified --namespace value
    failure_message=$(! kubectl get secrets mysecret 2>&1 "${KUBE_FLAGS[@]}")
    kube::test::if_has_string "${failure_message}" 'secrets "mysecret" not found'
    kube::test::if_has_string "${output_message}" 'user-specified'
    # Command
    output_message=$(kubectl create "${KUBE_FLAGS[@]}" secret generic mysecret --dry-run --from-literal=foo=bar -o jsonpath='{.metadata.namespace}')
    # Post-condition: jsonpath for .metadata.namespace should be empty for object since --namespace was not explicitly specified
    kube::test::if_empty_string "${output_message}"

    kubectl create configmap tester-create-cm -o json --dry-run | kubectl create "${KUBE_FLAGS[@]}" --raw /api/v1/namespaces/default/configmaps -f -
    kubectl delete -ndefault "${KUBE_FLAGS[@]}" configmap tester-create-cm

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
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $ID_FIELD \"test-secrets\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-secrets
  # Post-condition: namespace 'test-secrets' is created.
  kube::test::get_object_assert 'namespaces/test-secrets' "{{$ID_FIELD}}" 'test-secrets'

  ### Create a generic secret in a specific namespace
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create secret generic test-secret --from-literal=key1=value1 --type=test-type --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$ID_FIELD}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$SECRET_TYPE}}" 'test-type'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${KUBE_FLAGS[@]}" | grep 'key1: dmFsdWUx')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a docker-registry secret in a specific namespace
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create secret docker-registry test-secret --docker-username=test-user --docker-password=test-password --docker-email='test-user@test.com' --namespace=test-secrets
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$ID_FIELD}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$SECRET_TYPE}}" 'kubernetes.io/dockerconfigjson'
  [[ "$(kubectl get secret/test-secret --namespace=test-secrets -o yaml "${KUBE_FLAGS[@]}" | grep '.dockerconfigjson: eyJhdXRocyI6eyJodHRwczovL2luZGV4LmRvY2tlci5pby92MS8iOnsidXNlcm5hbWUiOiJ0ZXN0LXVzZXIiLCJwYXNzd29yZCI6InRlc3QtcGFzc3dvcmQiLCJlbWFpbCI6InRlc3QtdXNlckB0ZXN0LmNvbSIsImF1dGgiOiJkR1Z6ZEMxMWMyVnlPblJsYzNRdGNHRnpjM2R2Y21RPSJ9fX0=')" ]]
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  ### Create a tls secret
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  fi
  # Pre-condition: no SECRET exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create secret tls test-secret --namespace=test-secrets --key=hack/testdata/tls.key --cert=hack/testdata/tls.crt
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$ID_FIELD}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$SECRET_TYPE}}" 'kubernetes.io/tls'
  # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  # Command with process substitution
  kubectl create secret tls test-secret --namespace=test-secrets --key <(cat hack/testdata/tls.key) --cert <(cat hack/testdata/tls.crt)
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$ID_FIELD}}" 'test-secret'
  kube::test::get_object_assert 'secret/test-secret --namespace=test-secrets' "{{$SECRET_TYPE}}" 'kubernetes.io/tls'
    # Clean-up
  kubectl delete secret test-secret --namespace=test-secrets

  # Create a secret using stringData
  kubectl create --namespace=test-secrets -f - "${KUBE_FLAGS[@]}" << __EOF__
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
    kube::test::wait_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  fi
  # Pre-condition: no secret exists
  kube::test::get_object_assert 'secrets --namespace=test-secrets' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
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
  kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $ID_FIELD \"test-service-accounts\" }}found{{end}}{{end}}:' ':'
  # Command
  kubectl create namespace test-service-accounts
  # Post-condition: namespace 'test-service-accounts' is created.
  kube::test::get_object_assert 'namespaces/test-service-accounts' "{{$ID_FIELD}}" 'test-service-accounts'

  ### Create a service account in a specific namespace
  # Command
  kubectl create serviceaccount test-service-account --namespace=test-service-accounts
  # Post-condition: secret exists and has expected values
  kube::test::get_object_assert 'serviceaccount/test-service-account --namespace=test-service-accounts' "{{$ID_FIELD}}" 'test-service-account'
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
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:'
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
  kube::test::get_object_assert 'services redis-master' "{{range$SERVICE_SELECTOR_FIELD}}{{.}}:{{end}}" "redis:master:backend:"

  # Set selector of a local file without talking to the server
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan --local -o yaml "${KUBE_FLAGS[@]}"
  ! kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan --dry-run -o yaml "${KUBE_FLAGS[@]}"
  # Set command to change the selector.
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml role=padawan
  # prove role=padawan
  kube::test::get_object_assert 'services redis-master' "{{range$SERVICE_SELECTOR_FIELD}}{{.}}:{{end}}" "padawan:"
  # Set command to reset the selector back to the original one.
  kubectl set selector -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml app=redis,role=master,tier=backend
  # prove role=master
  kube::test::get_object_assert 'services redis-master' "{{range$SERVICE_SELECTOR_FIELD}}{{.}}:{{end}}" "redis:master:backend:"
  # Show dry-run works on running selector
  kubectl set selector services redis-master role=padawan --dry-run -o yaml "${KUBE_FLAGS[@]}"
  ! kubectl set selector services redis-master role=padawan --local -o yaml "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert 'services redis-master' "{{range$SERVICE_SELECTOR_FIELD}}{{.}}:{{end}}" "redis:master:backend:"

  ### Dump current redis-master service
  output_service=$(kubectl get service redis-master -o json "${KUBE_FLAGS[@]}")

  ### Delete redis-master-service by id
  # Pre-condition: redis-master service exists
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl delete service redis-master "${KUBE_FLAGS[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'

  ### Create redis-master-service from dumped JSON
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  # Command
  echo "${output_service}" | kubectl create -f - "${KUBE_FLAGS[@]}"
  # Post-condition: redis-master service is created
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:'

  ### Create redis-master-v1-test service
  # Pre-condition: redis-master-service service exists
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:'
  # Command
  kubectl create -f - "${KUBE_FLAGS[@]}" << __EOF__
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
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'

  ### Identity
  kubectl get service "${KUBE_FLAGS[@]}" service-v1-test -o json | kubectl replace "${KUBE_FLAGS[@]}" -f -

  ### Delete services by id
  # Pre-condition: service-v1-test exists
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:service-.*-test:'
  # Command
  kubectl delete service redis-master "${KUBE_FLAGS[@]}"
  kubectl delete service "service-v1-test" "${KUBE_FLAGS[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'

  ### Create two services
  # Pre-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-master-service.yaml "${KUBE_FLAGS[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/redis-slave-service.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: redis-master and redis-slave services are created
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:redis-slave:'

  ### Custom columns can be specified
  # Pre-condition: generate output using custom columns
  output_message=$(kubectl get services -o=custom-columns=NAME:.metadata.name,RSRC:.metadata.resourceVersion 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: should contain name column
  kube::test::if_has_string "${output_message}" 'redis-master'

  ### Delete multiple services at once
  # Pre-condition: redis-master and redis-slave services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:redis-master:redis-slave:'
  # Command
  kubectl delete services redis-master redis-slave "${KUBE_FLAGS[@]}" # delete multiple services at once
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'

  ### Create an ExternalName service
  # Pre-condition: Only the default kubernetes service exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  # Command
  kubectl create service externalname beep-boop --external-name bar.com
  # Post-condition: beep-boop service is created
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'beep-boop:kubernetes:'

  ### Delete beep-boop service by id
  # Pre-condition: beep-boop service exists
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'beep-boop:kubernetes:'
  # Command
  kubectl delete service beep-boop "${KUBE_FLAGS[@]}"
  if [[ "${WAIT_FOR_DELETION:-}" == "true" ]]; then
    kube::test::wait_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'
  fi
  # Post-condition: Only the default kubernetes services exist
  kube::test::get_object_assert services "{{range.items}}{{$ID_FIELD}}:{{end}}" 'kubernetes:'

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
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  kubectl delete rc frontend "${KUBE_FLAGS[@]}"
  # Post-condition: no pods from frontend controller
  kube::test::get_object_assert 'pods -l "name=frontend"' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create replication controller frontend from JSON
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:'
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
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '3'
  # Command
  kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${KUBE_FLAGS[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '2'

  ### Scale replication controller frontend with (wrong) current-replicas and replicas
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '2'
  # Command
  ! kubectl scale --current-replicas=3 --replicas=2 replicationcontrollers frontend "${KUBE_FLAGS[@]}"
  # Post-condition: nothing changed
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '2'

  ### Scale replication controller frontend with replicas only
  # Pre-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '2'
  # Command
  kubectl scale  --replicas=3 replicationcontrollers frontend "${KUBE_FLAGS[@]}"
  # Post-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '3'

  ### Scale replication controller from JSON with replicas only
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '3'
  # Command
  kubectl scale  --replicas=2 -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: 2 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '2'
  # Clean-up
  kubectl delete rc frontend "${KUBE_FLAGS[@]}"

  ### Scale multiple replication controllers
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-master-controller.yaml "${KUBE_FLAGS[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-slave-controller.yaml "${KUBE_FLAGS[@]}"
  # Command
  kubectl scale rc/redis-master rc/redis-slave --replicas=4 "${KUBE_FLAGS[@]}"
  # Post-condition: 4 replicas each
  kube::test::get_object_assert 'rc redis-master' "{{$RC_REPLICAS_FIELD}}" '4'
  kube::test::get_object_assert 'rc redis-slave' "{{$RC_REPLICAS_FIELD}}" '4'
  # Clean-up
  kubectl delete rc redis-{master,slave} "${KUBE_FLAGS[@]}"

  ### Scale a job
  kubectl create -f test/fixtures/doc-yaml/user-guide/job.yaml "${KUBE_FLAGS[@]}"
  # Command
  kubectl scale --replicas=2 job/pi
  # Post-condition: 2 replicas for pi
  kube::test::get_object_assert 'job pi' "{{$JOB_PARALLELISM_FIELD}}" '2'
  # Clean-up
  kubectl delete job/pi "${KUBE_FLAGS[@]}"

  ### Scale a deployment
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${KUBE_FLAGS[@]}"
  # Command
  kubectl scale --current-replicas=3 --replicas=1 deployment/nginx-deployment
  # Post-condition: 1 replica for nginx-deployment
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$DEPLOYMENT_REPLICAS}}" '1'
  # Clean-up
  kubectl delete deployment/nginx-deployment "${KUBE_FLAGS[@]}"

  ### Expose deployments by creating a service
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1deployment.yaml --port 80 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${KUBE_FLAGS[@]}"
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta2deployment.yaml --port 80 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${KUBE_FLAGS[@]}"
  # Uses deployment selectors for created service
  output_message=$(kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta1deployment.yaml --port 80 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'service/expose-test-deployment exposed'
  # Clean-up
  kubectl delete service/expose-test-deployment "${KUBE_FLAGS[@]}"
  # Contains no selectors, should fail.
  output_message=$(! kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1deployment-no-selectors.yaml --port 80 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'invalid deployment: no selectors'
  # Contains no selectors, should fail.
  output_message=$(! kubectl expose -f test/fixtures/pkg/kubectl/cmd/expose/appsv1beta2deployment-no-selectors.yaml --port 80 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: service created for deployment.
  kube::test::if_has_string "${output_message}" 'invalid deployment: no selectors'

  ### Expose a deployment as a service
  kubectl create -f test/fixtures/doc-yaml/user-guide/deployment.yaml "${KUBE_FLAGS[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'deployment nginx-deployment' "{{$DEPLOYMENT_REPLICAS}}" '3'
  # Command
  kubectl expose deployment/nginx-deployment
  # Post-condition: service exists and exposes deployment port (80)
  kube::test::get_object_assert 'service nginx-deployment' "{{$PORT_FIELD}}" '80'
  # Clean-up
  kubectl delete deployment/nginx-deployment service/nginx-deployment "${KUBE_FLAGS[@]}"

  ### Expose replication controller as service
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  # Pre-condition: 3 replicas
  kube::test::get_object_assert 'rc frontend' "{{$RC_REPLICAS_FIELD}}" '3'
  # Command
  kubectl expose rc frontend --port=80 "${KUBE_FLAGS[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend' "{{$PORT_NAME}} {{$PORT_FIELD}}" '<no value> 80'
  # Command
  kubectl expose service frontend --port=443 --name=frontend-2 "${KUBE_FLAGS[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-2' "{{$PORT_NAME}} {{$PORT_FIELD}}" '<no value> 443'
  # Command
  kubectl create -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml "${KUBE_FLAGS[@]}"
  kubectl expose pod valid-pod --port=444 --name=frontend-3 "${KUBE_FLAGS[@]}"
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service frontend-3' "{{$PORT_NAME}} {{$PORT_FIELD}}" '<no value> 444'
  # Create a service using service/v1 generator
  kubectl expose rc frontend --port=80 --name=frontend-4 --generator=service/v1 "${KUBE_FLAGS[@]}"
  # Post-condition: service exists and the port is named default.
  kube::test::get_object_assert 'service frontend-4' "{{$PORT_NAME}} {{$PORT_FIELD}}" 'default 80'
  # Verify that expose service works without specifying a port.
  kubectl expose service frontend --name=frontend-5 "${KUBE_FLAGS[@]}"
  # Post-condition: service exists with the same port as the original service.
  kube::test::get_object_assert 'service frontend-5' "{{$PORT_FIELD}}" '80'
  # Cleanup services
  kubectl delete pod valid-pod "${KUBE_FLAGS[@]}"
  kubectl delete service frontend{,-2,-3,-4,-5} "${KUBE_FLAGS[@]}"

  ### Expose negative invalid resource test
  # Pre-condition: don't need
  # Command
  output_message=$(! kubectl expose nodes 127.0.0.1 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: the error message has "cannot expose" string
  kube::test::if_has_string "${output_message}" 'cannot expose'

  ### Try to generate a service with invalid name (exceeding maximum valid size)
  # Pre-condition: use --name flag
  output_message=$(! kubectl expose -f hack/testdata/pod-with-large-name.yaml --name=invalid-large-service-name-that-has-more-than-sixty-three-characters --port=8081 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: should fail due to invalid name
  kube::test::if_has_string "${output_message}" 'metadata.name: Invalid value'
  # Pre-condition: default run without --name flag; should succeed by truncating the inherited name
  output_message=$(kubectl expose -f hack/testdata/pod-with-large-name.yaml --port=8081 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: inherited name from pod has been truncated
  kube::test::if_has_string "${output_message}" 'kubernetes-serve-hostname-testing-sixty-three-characters-in-len exposed'
  # Clean-up
  kubectl delete svc kubernetes-serve-hostname-testing-sixty-three-characters-in-len "${KUBE_FLAGS[@]}"

  ### Expose multiport object as a new service
  # Pre-condition: don't use --port flag
  output_message=$(kubectl expose -f test/fixtures/doc-yaml/admin/high-availability/etcd.yaml --selector=test=etcd 2>&1 "${KUBE_FLAGS[@]}")
  # Post-condition: expose succeeded
  kube::test::if_has_string "${output_message}" 'etcd-server exposed'
  # Post-condition: generated service has both ports from the exposed pod
  kube::test::get_object_assert 'service etcd-server' "{{$PORT_NAME}} {{$PORT_FIELD}}" 'port-1 2380'
  kube::test::get_object_assert 'service etcd-server' "{{$SECOND_PORT_NAME}} {{$SECOND_PORT_FIELD}}" 'port-2 2379'
  # Clean-up
  kubectl delete svc etcd-server "${KUBE_FLAGS[@]}"

  ### Delete replication controller with id
  # Pre-condition: frontend replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:'
  # Command
  kubectl delete rc frontend "${KUBE_FLAGS[@]}"
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Create two replication controllers
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  kubectl create -f test/e2e/testing-manifests/guestbook/legacy/redis-slave-controller.yaml "${KUBE_FLAGS[@]}"
  # Post-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:redis-slave:'

  ### Delete multiple controllers at once
  # Pre-condition: frontend and redis-slave
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:redis-slave:'
  # Command
  kubectl delete rc frontend redis-slave "${KUBE_FLAGS[@]}" # delete multiple controllers at once
  # Post-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ### Auto scale replication controller
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" 'frontend:'
  # autoscale 1~2 pods, CPU utilization 70%, rc specified by file
  kubectl autoscale -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}" --max=2 --cpu-percent=70
  kube::test::get_object_assert 'hpa frontend' "{{$HPA_MIN_FIELD}} {{$HPA_MAX_FIELD}} {{$HPA_CPU_FIELD}}" '1 2 70'
  kubectl delete hpa frontend "${KUBE_FLAGS[@]}"
  # autoscale 2~3 pods, no CPU utilization specified, rc specified by name
  kubectl autoscale rc frontend "${KUBE_FLAGS[@]}" --min=2 --max=3
  kube::test::get_object_assert 'hpa frontend' "{{$HPA_MIN_FIELD}} {{$HPA_MAX_FIELD}} {{$HPA_CPU_FIELD}}" '2 3 80'
  kubectl delete hpa frontend "${KUBE_FLAGS[@]}"
  # autoscale without specifying --max should fail
  ! kubectl autoscale rc frontend "${KUBE_FLAGS[@]}"
  # Clean up
  kubectl delete rc frontend "${KUBE_FLAGS[@]}"

  ## Set resource limits/request of a deployment
  # Pre-condition: no deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Set resources of a local file without talking to the server
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --local -o yaml "${KUBE_FLAGS[@]}"
  ! kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m --dry-run -o yaml "${KUBE_FLAGS[@]}"
  # Create a deployment
  kubectl create -f hack/testdata/deployment-multicontainer-resources.yaml "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$ID_FIELD}}:{{end}}" 'nginx-deployment-resources:'
  kube::test::get_object_assert deployment "{{range.items}}{{$IMAGE_FIELD0}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  kube::test::get_object_assert deployment "{{range.items}}{{$IMAGE_FIELD1}}:{{end}}" "${IMAGE_PERL}:"
  # Set the deployment's cpu limits
  kubectl set resources deployment nginx-deployment-resources --limits=cpu=100m "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "100m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set a non-existing container should fail
  ! kubectl set resources deployment nginx-deployment-resources -c=redis --limits=cpu=100m
  # Set the limit of a specific container in deployment
  kubectl set resources deployment nginx-deployment-resources -c=nginx --limits=cpu=200m "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "100m:"
  # Set limits/requests of a deployment specified by a file
  kubectl set resources -f hack/testdata/deployment-multicontainer-resources.yaml -c=perl --limits=cpu=300m --requests=cpu=300m "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Show dry-run works on running deployments
  kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --dry-run -o yaml "${KUBE_FLAGS[@]}"
  ! kubectl set resources deployment nginx-deployment-resources -c=perl --limits=cpu=400m --requests=cpu=400m --local -o yaml "${KUBE_FLAGS[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 0).resources.limits.cpu}}:{{end}}" "200m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.limits.cpu}}:{{end}}" "300m:"
  kube::test::get_object_assert deployment "{{range.items}}{{(index .spec.template.spec.containers 1).resources.requests.cpu}}:{{end}}" "300m:"
  # Clean up
  kubectl delete deployment nginx-deployment-resources "${KUBE_FLAGS[@]}"

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
  # kube::test::get_object_assert namespaces "{{range.items}}{{$ID_FIELD}}:{{end}}" 'default:'
  # Command
  kubectl create namespace my-namespace
  # Post-condition: namespace 'my-namespace' is created.
  kube::test::get_object_assert 'namespaces/my-namespace' "{{$ID_FIELD}}" 'my-namespace'
  # Clean up
  kubectl delete namespace my-namespace --wait=false
  # make sure that wait properly waits for finalization
  kubectl wait --for=delete ns/my-namespace
  output_message=$(! kubectl get ns/my-namespace 2>&1 "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" ' not found'

  ######################
  # Pods in Namespaces #
  ######################

  if kube::test::if_supports_resource "${PODS}" ; then
    ### Create a new namespace
    # Pre-condition: the other namespace does not exist
    kube::test::get_object_assert 'namespaces' '{{range.items}}{{ if eq $ID_FIELD \"other\" }}found{{end}}{{end}}:' ':'
    # Command
    kubectl create namespace other
    # Post-condition: namespace 'other' is created.
    kube::test::get_object_assert 'namespaces/other' "{{$ID_FIELD}}" 'other'

    ### Create POD valid-pod in specific namespace
    # Pre-condition: no POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
    # Command
    kubectl create "${KUBE_FLAGS[@]}" --namespace=other -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
    # Post-condition: valid-pod POD is created
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
    # Post-condition: verify shorthand `-n other` has the same results as `--namespace=other`
    kube::test::get_object_assert 'pods -n other' "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
    # Post-condition: a resource cannot be retrieved by name across all namespaces
    output_message=$(! kubectl get "${KUBE_FLAGS[@]}" pod valid-pod --all-namespaces 2>&1)
    kube::test::if_has_string "${output_message}" "a resource cannot be retrieved by name across all namespaces"

    ### Delete POD valid-pod in specific namespace
    # Pre-condition: valid-pod POD exists
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$ID_FIELD}}:{{end}}" 'valid-pod:'
    # Command
    kubectl delete "${KUBE_FLAGS[@]}" pod --namespace=other valid-pod --grace-period=0 --force
    # Post-condition: valid-pod POD doesn't exist
    kube::test::get_object_assert 'pods --namespace=other' "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
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

  kube::test::get_object_assert nodes "{{range.items}}{{$ID_FIELD}}:{{end}}" '127.0.0.1:'

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
  kubectl patch "${KUBE_FLAGS[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":true}}'
  # Post-condition: node is unschedulable
  kube::test::get_object_assert "nodes 127.0.0.1" "{{.spec.unschedulable}}" 'true'
  kubectl patch "${KUBE_FLAGS[@]}" nodes "127.0.0.1" -p='{"spec":{"unschedulable":null}}'
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
  kubectl create -f test/fixtures/doc-yaml/user-guide/walkthrough/podtemplate.json "${KUBE_FLAGS[@]}"
  # Post-condition: nginx PODTEMPLATE is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'

  ### Printing pod templates works
  kubectl get podtemplates "${KUBE_FLAGS[@]}"
  [[ "$(kubectl get podtemplates -o yaml "${KUBE_FLAGS[@]}" | grep nginx)" ]]

  ### Delete nginx pod template by name
  # Pre-condition: nginx pod template is available
  kube::test::get_object_assert podtemplates "{{range.items}}{{.metadata.name}}:{{end}}" 'nginx:'
  # Command
  kubectl delete podtemplate nginx "${KUBE_FLAGS[@]}"
  # Post-condition: No templates exist
  kube::test::get_object_assert podtemplate "{{range.items}}{{.metadata.name}}:{{end}}" ''

  set +o nounset
  set +o errexit
}
