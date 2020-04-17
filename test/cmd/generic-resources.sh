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

run_multi_resources_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:multiple resources)"

  FILES="hack/testdata/multi-resource-yaml
  hack/testdata/multi-resource-list
  hack/testdata/multi-resource-json
  hack/testdata/multi-resource-rclist
  hack/testdata/multi-resource-svclist"
  YAML=".yaml"
  JSON=".json"
  for file in $FILES; do
    if [ -f "${file}${YAML}" ]
    then
      file=${file}${YAML}
      replace_file="${file%.yaml}-modify.yaml"
    else
      file=${file}${JSON}
      replace_file="${file%.json}-modify.json"
    fi

    has_svc=true
    has_rc=true
    two_rcs=false
    two_svcs=false
    if [[ "${file}" == *rclist* ]]; then
      has_svc=false
      two_rcs=true
    fi
    if [[ "${file}" == *svclist* ]]; then
      has_rc=false
      two_svcs=true
    fi

    ### Create, get, describe, replace, label, annotate, and then delete service nginxsvc and replication controller my-nginx from 5 types of files:
    ### 1) YAML, separated by ---; 2) JSON, with a List type; 3) JSON, with JSON object concatenation
    ### 4) JSON, with a ReplicationControllerList type; 5) JSON, with a ServiceList type
    echo "Testing with file ${file} and replace with file ${replace_file}"
    # Pre-condition: no service (other than default kubernetes services) or replication controller exists
    kube::test::get_object_assert services "{{range.items}}{{${id_field:?}}}:{{end}}" ''
    kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
    # Command
    kubectl create -f "${file}" "${kube_flags[@]:?}"
    # Post-condition: mock service (and mock2) exists
    if [ "$has_svc" = true ]; then
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:mock2:'
      else
        kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
      fi
    fi
    # Post-condition: mock rc (and mock2) exists
    if [ "$has_rc" = true ]; then
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:mock2:'
      else
        kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
      fi
    fi
    # Command
    kubectl get -f "${file}" "${kube_flags[@]}"
    # Command: watching multiple resources should return "not supported" error
    WATCH_ERROR_FILE="${KUBE_TEMP}/kubectl-watch-error"
    kubectl get -f "${file}" "${kube_flags[@]}" "--watch" 2> "${WATCH_ERROR_FILE}" || true
    if ! grep -q "watch is only supported on individual resources and resource collections" "${WATCH_ERROR_FILE}"; then
      kube::log::error_exit "kubectl watch multiple resource returns unexpected error or non-error: $(cat "${WATCH_ERROR_FILE}")" "1"
    fi
    kubectl describe -f "${file}" "${kube_flags[@]}"
    # Command
    kubectl replace -f "${replace_file}" --force --cascade "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are replaced
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field:?}.status}}" 'replaced'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.status}}" 'replaced'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.status}}" 'replaced'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.status}}" 'replaced'
      fi
    fi
    # Command: kubectl edit multiple resources
    temp_editor="${KUBE_TEMP}/tmp-editor.sh"
    echo -e "#!/usr/bin/env bash\n${SED} -i \"s/status\:\ replaced/status\:\ edited/g\" \$@" > "${temp_editor}"
    chmod +x "${temp_editor}"
    EDITOR="${temp_editor}" kubectl edit "${kube_flags[@]}" -f "${file}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are edited
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field}.status}}" 'edited'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.status}}" 'edited'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.status}}" 'edited'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.status}}" 'edited'
      fi
    fi
    # cleaning
    rm "${temp_editor}"
    # Command
    # We need to set --overwrite, because otherwise, if the first attempt to run "kubectl label"
    # fails on some, but not all, of the resources, retries will fail because it tries to modify
    # existing labels.
    kubectl-with-retry label -f "${file}" labeled=true --overwrite "${kube_flags[@]}"
    # Post-condition: mock service and mock rc (and mock2) are labeled
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${labels_field}.labeled}}" 'true'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${labels_field}.labeled}}" 'true'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${labels_field}.labeled}}" 'true'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${labels_field}.labeled}}" 'true'
      fi
    fi
    # Command
    # Command
    # We need to set --overwrite, because otherwise, if the first attempt to run "kubectl annotate"
    # fails on some, but not all, of the resources, retries will fail because it tries to modify
    # existing annotations.
    kubectl-with-retry annotate -f "${file}" annotated=true --overwrite "${kube_flags[@]}"
    # Post-condition: mock service (and mock2) and mock rc (and mock2) are annotated
    if [ "$has_svc" = true ]; then
      kube::test::get_object_assert 'services mock' "{{${annotations_field:?}.annotated}}" 'true'
      if [ "$two_svcs" = true ]; then
        kube::test::get_object_assert 'services mock2' "{{${annotations_field}.annotated}}" 'true'
      fi
    fi
    if [ "$has_rc" = true ]; then
      kube::test::get_object_assert 'rc mock' "{{${annotations_field}.annotated}}" 'true'
      if [ "$two_rcs" = true ]; then
        kube::test::get_object_assert 'rc mock2' "{{${annotations_field}.annotated}}" 'true'
      fi
    fi
    # Cleanup resources created
    kubectl delete -f "${file}" "${kube_flags[@]}"
  done

  #############################
  # Multiple Resources via URL#
  #############################

  # Pre-condition: no service (other than default kubernetes services) or replication controller exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  # Command
  kubectl create -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: service(mock) and rc(mock) exist
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" 'mock:'
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'mock:'

  # Clean up
  kubectl delete -f https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/testdata/multi-resource-yaml.yaml "${kube_flags[@]}"

  # Post-condition: no service (other than default kubernetes services) or replication controller exists
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_recursive_resources_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing recursive resources"
  ### Create multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  output_message=$(! kubectl create -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are created, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'

  ## Edit multiple busybox PODs by updating the image field of multiple PODs recursively from a directory. tmp-editor.sh is a fake editor
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  # shellcheck disable=SC2016  # $1 here is not a Expressions
  echo -e '#!/usr/bin/env bash\nsed -i "s/image: busybox/image: prom\/busybox/g" $1' > /tmp/tmp-editor.sh
  chmod +x /tmp/tmp-editor.sh
  output_message=$(! EDITOR=/tmp/tmp-editor.sh kubectl edit -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are not edited, and since busybox2 is malformed, it should error
  # The reason why busybox0 & busybox1 PODs are not edited is because the editor tries to load all objects in
  # a list but since it contains invalid objects, it will never open.
  kube::test::get_object_assert pods "{{range.items}}{{${image_field:?}}}:{{end}}" 'busybox:busybox:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  # cleaning
  rm /tmp/tmp-editor.sh

  ## Replace multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl replace -f hack/testdata/recursive/pod-modify --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are replaced, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.status}}:{{end}}" 'replaced:replaced:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'

  ## Describe multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl describe -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are described, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "app=busybox0"
  kube::test::if_has_string "${output_message}" "app=busybox1"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Annotate multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl annotate -f hack/testdata/recursive/pod annotatekey='annotatevalue' --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are annotated, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${annotations_field}.annotatekey}}:{{end}}" 'annotatevalue:annotatevalue:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Apply multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl apply -f hack/testdata/recursive/pod-modify --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are updated, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.status}}:{{end}}" 'replaced:replaced:'
  kube::test::if_has_string "${output_message}" 'error validating data: kind not set'


  ### Convert deployment YAML file locally without affecting the live deployment.
  # Pre-condition: no deployments exist
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create a deployment (revision 1)
  kubectl create -f hack/testdata/deployment-revision1.yaml "${kube_flags[@]}"
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx:'
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Command
  output_message=$(kubectl convert --local -f hack/testdata/deployment-revision1.yaml --output-version=extensions/v1beta1 -o yaml "${kube_flags[@]}")
  # Post-condition: apiVersion is still apps/v1 in the live deployment, but command output is the new value
  kube::test::get_object_assert 'deployment nginx' "{{ .apiVersion }}" 'apps/v1'
  kube::test::if_has_string "${output_message}" "extensions/v1beta1"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]}"

  ## Convert multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: only busybox0 & busybox1 PODs exist
  kube::test::wait_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl convert -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are converted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Get multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl get -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]}" -o go-template="{{range.items}}{{$id_field}}:{{end}}")
  # Post-condition: busybox0 & busybox1 PODs are retrieved, but because busybox2 is malformed, it should not show up
  kube::test::if_has_string "${output_message}" "busybox0:busybox1:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Label multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl label -f hack/testdata/recursive/pod mylabel='myvalue' --recursive 2>&1 "${kube_flags[@]}")
  echo "${output_message}"
  # Post-condition: busybox0 & busybox1 PODs are labeled, but because busybox2 is malformed, it should not show up
  kube::test::get_object_assert pods "{{range.items}}{{${labels_field}.mylabel}}:{{end}}" 'myvalue:myvalue:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ## Patch multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl patch -f hack/testdata/recursive/pod -p='{"spec":{"containers":[{"name":"busybox","image":"prom/busybox"}]}}' --recursive 2>&1 "${kube_flags[@]}")
  echo "${output_message}"
  # Post-condition: busybox0 & busybox1 PODs are patched, but because busybox2 is malformed, it should not show up
  kube::test::get_object_assert pods "{{range.items}}{{$image_field}}:{{end}}" 'prom/busybox:prom/busybox:'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Delete multiple busybox PODs recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl delete -f hack/testdata/recursive/pod --recursive --grace-period=0 --force 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 PODs are deleted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Create replication controller recursively from directory of YAML files
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  ! kubectl create -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}" || exit 1
  # Post-condition: frontend replication controller is created
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'

  ### Autoscale multiple replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{${rc_replicas_field:?}}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl autoscale --min=1 --max=2 -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox replication controllers are autoscaled
  # with min. of 1 replica & max of 2 replicas, and since busybox2 is malformed, it should error
  kube::test::get_object_assert 'hpa busybox0' "{{${hpa_min_field:?}}} {{${hpa_max_field:?}}} {{${hpa_cpu_field:?}}}" '1 2 80'
  kube::test::get_object_assert 'hpa busybox1' "{{$hpa_min_field}} {{$hpa_max_field}} {{$hpa_cpu_field}}" '1 2 80'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kubectl delete hpa busybox0 "${kube_flags[@]}"
  kubectl delete hpa busybox1 "${kube_flags[@]}"

  ### Expose multiple replication controllers as service recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl expose -f hack/testdata/recursive/rc --recursive --port=80 2>&1 "${kube_flags[@]}")
  # Post-condition: service exists and the port is unnamed
  kube::test::get_object_assert 'service busybox0' "{{${port_name:?}}} {{${port_field:?}}}" '<no value> 80'
  kube::test::get_object_assert 'service busybox1' "{{$port_name}} {{$port_field}}" '<no value> 80'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Scale multiple replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 replication controllers exist & 1
  # replica each
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '1'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '1'
  # Command
  output_message=$(! kubectl scale --current-replicas=1 --replicas=2 -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 replication controllers are scaled to 2 replicas, and since busybox2 is malformed, it should error
  kube::test::get_object_assert 'rc busybox0' "{{$rc_replicas_field}}" '2'
  kube::test::get_object_assert 'rc busybox1' "{{$rc_replicas_field}}" '2'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Delete multiple busybox replication controllers recursively from directory of YAML files
  # Pre-condition: busybox0 & busybox1 PODs exist
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  output_message=$(! kubectl delete -f hack/testdata/recursive/rc --recursive --grace-period=0 --force 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 replication controllers are deleted, and since busybox2 is malformed, it should error
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  ### Rollout on multiple deployments recursively
  # Pre-condition: no deployments exist
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create deployments (revision 1) recursively from directory of YAML files
  ! kubectl create -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}" || exit 1
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" 'nginx0-deployment:nginx1-deployment:'
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
  ## Rollback the deployments to revision 1 recursively
  output_message=$(! kubectl rollout undo -f hack/testdata/recursive/deployment --recursive --to-revision=1 2>&1 "${kube_flags[@]}")
  # Post-condition: nginx0 & nginx1 should be a no-op, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{$image_field0}}:{{end}}" "${IMAGE_NGINX}:${IMAGE_NGINX}:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Pause the deployments recursively
  # shellcheck disable=SC2034  # PRESERVE_ERR_FILE is used in kubectl-with-retry
  PRESERVE_ERR_FILE=true
  kubectl-with-retry rollout pause -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}"
  output_message=$(cat "${ERROR_FILE}")
  # Post-condition: nginx0 & nginx1 should both have paused set to true, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{.spec.paused}}:{{end}}" "true:true:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Resume the deployments recursively
  kubectl-with-retry rollout resume -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}"
  output_message=$(cat "${ERROR_FILE}")
  # Post-condition: nginx0 & nginx1 should both have paused set to nothing, and since nginx2 is malformed, it should error
  kube::test::get_object_assert deployment "{{range.items}}{{.spec.paused}}:{{end}}" "<no value>:<no value>:"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Retrieve the rollout history of the deployments recursively
  output_message=$(! kubectl rollout history -f hack/testdata/recursive/deployment --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: nginx0 & nginx1 should both have a history, and since nginx2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "nginx0-deployment"
  kube::test::if_has_string "${output_message}" "nginx1-deployment"
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  # Clean up
  unset PRESERVE_ERR_FILE
  rm "${ERROR_FILE}"
  ! kubectl delete -f hack/testdata/recursive/deployment --recursive "${kube_flags[@]}" --grace-period=0 --force || exit 1
  sleep 1

  ### Rollout on multiple replication controllers recursively - these tests ensure that rollouts cannot be performed on resources that don't support it
  # Pre-condition: no replication controller exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  # Create replication controllers recursively from directory of YAML files
  ! kubectl create -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}" || exit 1
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" 'busybox0:busybox1:'
  # Command
  ## Attempt to rollback the replication controllers to revision 1 recursively
  output_message=$(! kubectl rollout undo -f hack/testdata/recursive/rc --recursive --to-revision=1 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" 'no rollbacker has been implemented for "ReplicationController"'
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  ## Attempt to pause the replication controllers recursively
  output_message=$(! kubectl rollout pause -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox0" pausing is not supported'
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox1" pausing is not supported'
  ## Attempt to resume the replication controllers recursively
  output_message=$(! kubectl rollout resume -f hack/testdata/recursive/rc --recursive 2>&1 "${kube_flags[@]}")
  # Post-condition: busybox0 & busybox1 should error as they are RC's, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox0" resuming is not supported'
  kube::test::if_has_string "${output_message}" 'replicationcontrollers "busybox1" resuming is not supported'
  # Clean up
  ! kubectl delete -f hack/testdata/recursive/rc --recursive "${kube_flags[@]}" --grace-period=0 --force || exit 1
  sleep 1

  set +o nounset
  set +o errexit
}

run_lists_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl(v1:lists)"

  ### Create a List with objects from multiple versions
  # Command
  kubectl create -f hack/testdata/list.yaml "${kube_flags[@]}"

  ### Delete the List with objects from multiple versions
  # Command
  kubectl delete service/list-service-test deployment/list-deployment-test

  set +o nounset
  set +o errexit
}
