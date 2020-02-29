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

  # Command with dry-run
  kubectl run --dry-run=client nginx-extensions "--image=${IMAGE_NGINX}" "${kube_flags[@]:?}"
  kubectl run --dry-run=server nginx-extensions "--image=${IMAGE_NGINX}" "${kube_flags[@]:?}"
  # Post-Condition: no Pod exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # Pre-Condition: no Pod exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl run nginx-extensions "--image=${IMAGE_NGINX}" "${kube_flags[@]:?}"
  # Post-Condition: Pod "nginx" is created
  kube::test::get_object_assert pod "{{range.items}}{{$id_field}}:{{end}}" 'nginx-extensions:'
  # Clean up
  kubectl delete pod nginx-extensions "${kube_flags[@]}"

  # Test that a valid image reference value is provided as the value of --image in `kubectl run <name> --image`
  output_message=$(kubectl run test1 --image=validname)
  kube::test::if_has_string "${output_message}" 'pod/test1 created'
  kubectl delete pods test1
  # test invalid image name
  output_message=$(! kubectl run test2 --image=InvalidImageName 2>&1)
  kube::test::if_has_string "${output_message}" 'error: Invalid image name "InvalidImageName": invalid reference format'

  set +o nounset
  set +o errexit
}
