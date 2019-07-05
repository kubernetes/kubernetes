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
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command: create a pod "test-pod"
  kubectl create -f hack/testdata/pod.yaml --save-config "${kube_flags[@]:?}"
  # Post-Condition: pod "test-pod" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}")"
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 2. kubectl edit --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ! grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}")"
  # Command: edit the pod "test-pod"
  temp_editor="${KUBE_TEMP}/tmp-editor.sh"
  echo -e "#!/usr/bin/env bash\n${SED} -i \"s/test-pod-label/test-pod-label-edited/g\" \$@" > "${temp_editor}"
  chmod +x "${temp_editor}"
  EDITOR=${temp_editor} kubectl edit pod test-pod --save-config "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}")"
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 3. kubectl replace --save-config should generate configuration annotation
  # Pre-Condition: no POD exists, then create pod "test-pod", which shouldn't have configuration annotation
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ! grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}")"
  # Command: replace the pod "test-pod"
  kubectl replace -f hack/testdata/pod.yaml --save-config "${kube_flags[@]}"
  # Post-Condition: pod "test-pod" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]}")"
  # Clean up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]}"
  ## 4. kubectl run --save-config should generate configuration annotation
  # Pre-Condition: no RC exists
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: create the rc "nginx" with image nginx
  kubectl run nginx "--image=$IMAGE_NGINX" --save-config --generator=run/v1 "${kube_flags[@]}"
  # Post-Condition: rc "nginx" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get rc nginx -o yaml "${kube_flags[@]}")"
  ## 5. kubectl expose --save-config should generate configuration annotation
  # Pre-Condition: no service exists
  kube::test::get_object_assert svc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command: expose the rc "nginx"
  kubectl expose rc nginx --save-config --port=80 --target-port=8000 "${kube_flags[@]}"
  # Post-Condition: service "nginx" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get svc nginx -o yaml "${kube_flags[@]}")"
  # Clean up
  kubectl delete rc,svc nginx
  ## 6. kubectl autoscale --save-config should generate configuration annotation
  # Pre-Condition: no RC exists, then create the rc "frontend", which shouldn't have configuration annotation
  kube::test::get_object_assert rc "{{range.items}}{{$id_field}}:{{end}}" ''
  kubectl create -f hack/testdata/frontend-controller.yaml "${kube_flags[@]}"
  ! grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get rc frontend -o yaml "${kube_flags[@]}")"
  # Command: autoscale rc "frontend"
  kubectl autoscale -f hack/testdata/frontend-controller.yaml --save-config "${kube_flags[@]}" --max=2
  # Post-Condition: hpa "frontend" has configuration annotation
  grep -q "kubectl.kubernetes.io/last-applied-configuration" <<< "$(kubectl get hpa frontend -o yaml "${kube_flags[@]}")"
  # Ensure we can interact with HPA objects in lists through autoscaling/v1 APIs
  output_message=$(kubectl get hpa -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  output_message=$(kubectl get hpa.autoscaling -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # tests kubectl group prefix matching
  output_message=$(kubectl get hpa.autoscal -o=jsonpath='{.items[0].apiVersion}' 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'autoscaling/v1'
  # Clean up
  # Note that we should delete hpa first, otherwise it may fight with the rc reaper.
  kubectl delete hpa frontend "${kube_flags[@]}"
  kubectl delete rc  frontend "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}
