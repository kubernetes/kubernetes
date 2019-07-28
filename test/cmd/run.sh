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

run_kubectl_run_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl run"
  ## kubectl run should create deployments, jobs or cronjob
  # Pre-Condition: no Job exists
  kube::test::get_object_assert jobs "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl run pi --generator=job/v1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]:?}"
  # Post-Condition: Job "pi" is created
  kube::test::get_object_assert jobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'
  # Describe command (resource only) should print detailed information
  kube::test::describe_resource_assert pods "Name:" "Image:" "Node:" "Labels:" "Status:" "Controlled By"
  # Clean up
  kubectl delete jobs pi "${kube_flags[@]}"
  # Post-condition: no pods exist.
  kube::test::wait_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''

  # Pre-Condition: no Deployment exists
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run nginx-extensions "--image=$IMAGE_NGINX" "${kube_flags[@]}"
  # Post-Condition: Deployment "nginx" is created
  kube::test::get_object_assert deployment.apps "{{range.items}}{{$id_field}}:{{end}}" 'nginx-extensions:'
  # new generator was used
  output_message=$(kubectl get deployment.apps/nginx-extensions -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_string "${output_message}" '10'
  # Clean up
  kubectl delete deployment nginx-extensions "${kube_flags[@]}"
  # Command
  kubectl run nginx-apps "--image=$IMAGE_NGINX" --generator=deployment/apps.v1 "${kube_flags[@]}"
  # Post-Condition: Deployment "nginx" is created
  kube::test::get_object_assert deployment.apps "{{range.items}}{{$id_field}}:{{end}}" 'nginx-apps:'
  # and new generator was used, iow. new defaults are applied
  output_message=$(kubectl get deployment/nginx-apps -o jsonpath='{.spec.revisionHistoryLimit}')
  kube::test::if_has_string "${output_message}" '10'
  # Clean up
  kubectl delete deployment nginx-apps "${kube_flags[@]}"

  # Pre-Condition: no Job exists
  kube::test::get_object_assert cronjobs "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl run pi --schedule="*/5 * * * *" --generator=cronjob/v1beta1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(20)' "${kube_flags[@]}"
  # Post-Condition: CronJob "pi" is created
  kube::test::get_object_assert cronjobs "{{range.items}}{{$id_field}}:{{end}}" 'pi:'

  # Pre-condition: cronjob has perl image, not custom image
  output_message=$(kubectl get cronjob/pi -o jsonpath='{..image}')
  kube::test::if_has_not_string "${output_message}" "custom-image"
  kube::test::if_has_string     "${output_message}" "${IMAGE_PERL}"
  # Set cronjob image
  kubectl set image cronjob/pi '*=custom-image'
  # Post-condition: cronjob has custom image, not perl image
  output_message=$(kubectl get cronjob/pi -o jsonpath='{..image}')
  kube::test::if_has_string     "${output_message}" "custom-image"
  kube::test::if_has_not_string "${output_message}" "${IMAGE_PERL}"

  # Clean up
  kubectl delete cronjobs pi "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_cmd_with_img_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing cmd with image"

  # Test that a valid image reference value is provided as the value of --image in `kubectl run <name> --image`
  output_message=$(kubectl run test1 --image=validname)
  kube::test::if_has_string "${output_message}" 'deployment.apps/test1 created'
  kubectl delete deployments test1
  # test invalid image name
  output_message=$(! kubectl run test2 --image=InvalidImageName 2>&1)
  kube::test::if_has_string "${output_message}" 'error: Invalid image name "InvalidImageName": invalid reference format'

  set +o nounset
  set +o errexit
}
