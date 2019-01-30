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

# Runs tests for --save-config tests.
run_save_config_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl --save-config"
  ## Configuration annotations should be set when --save-config is enabled
  ## 1. kubectl create --save-config should generate configuration annotation
  # Pre-Condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml --save-config "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  ## 2. kubectl edit --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  ! [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: edit the pod "test-pod"
  temp_editor="${KUBE_TEMP}/tmp-editor.sh"
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/test-pod-label/test-pod-label-edited/g\" \$@" > "${temp_editor}"
  chmod +x "${temp_editor}"
  EDITOR=${temp_editor} kubectl edit pod test-pod --save-config "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  ## 3. kubectl replace --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  ! [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: replace the pod "test-pod"
  kubectl replace -f hack/testdata/pod.yaml --save-config "${KUBE_FLAGS[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  [[ "$(kubectl get pods test-pod -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${KUBE_FLAGS[@]}"
  ## 4. kubectl run --save-config should generate configuration annotation
  # Pre-Condition: no RC exists
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command: create the rc "nginx" with image nginx
  kubectl run nginx "--image=$IMAGE_NGINX" --save-config --generator=run/v1 "${KUBE_FLAGS[@]}"
  # Post-Condition: rc "nginx" has configuration annotation
  [[ "$(kubectl get rc nginx -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  ## 5. kubectl expose --save-config should generate configuration annotation
  # Pre-Condition: no service exists
  kube::test::get_object_assert svc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  # Command: expose the rc "nginx"
  kubectl expose rc nginx --save-config --port=80 --target-port=8000 "${KUBE_FLAGS[@]}"
  # Post-Condition: service "nginx" has configuration annotation
  [[ "$(kubectl get svc nginx -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Clean up
  kubectl delete rc,svc nginx
  ## 6. kubectl autoscale --save-config should generate configuration annotation
  # Pre-Condition: no RC exists, then create the rc "frontend", which shouldn't have configuration annotation
  kube::test::get_object_assert rc "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kubectl create -f hack/testdata/frontend-controller.yaml "${KUBE_FLAGS[@]}"
  ! [[ "$(kubectl get rc frontend -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Command: autoscale rc "frontend"
  kubectl autoscale -f hack/testdata/frontend-controller.yaml --save-config "${KUBE_FLAGS[@]}" --max=2
  # Post-Condition: hpa "frontend" has configuration annotation
  [[ "$(kubectl get hpa frontend -o yaml "${KUBE_FLAGS[@]}" | grep kubectl.kubernetes.io/last-applied-configuration)" ]]
  # Ensure we can interact with HPA objects in lists through autoscaling/v1 APIs
  output_message=$(kubectl get hpa -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  output_message=$(kubectl get hpa.autoscaling -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # tests kubectl group prefix matching
  output_message=$(kubectl get hpa.autoscal -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${KUBE_FLAGS[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the rc reaper.
  kubectl delete hpa frontend "${KUBE_FLAGS[@]}"
  kubectl delete rc  frontend "${KUBE_FLAGS[@]}"

  set +o nounset
  set +o errexit
}
