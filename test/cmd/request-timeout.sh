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

run_kubectl_request_timeout_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl request timeout"
  ### Test global request timeout option
  # Pre-condition: no POD exists
  create_and_use_new_namespace
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  kubectl create "${kube_flags[@]:?}" -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml
  # Post-condition: valid-pod POD is created
  kubectl get "${kube_flags[@]}" pods -o json
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" 'valid-pod:'

  ## check --request-timeout on 'get pod'
  output_message=$(kubectl get pod valid-pod --request-timeout=1)
  kube::test::if_has_string "${output_message}" 'valid-pod'

  ## check --request-timeout on 'get pod' with --watch
  output_message=$(kubectl get pod valid-pod --request-timeout=1 --watch --v=5 2>&1)
  kube::test::if_has_string "${output_message}" 'Timeout'

  ## check --request-timeout value with no time unit
  output_message=$(kubectl get pod valid-pod --request-timeout=1 2>&1)
  kube::test::if_has_string "${output_message}" 'valid-pod'

  ## check --request-timeout value with invalid time unit
  output_message=$(! kubectl get pod valid-pod --request-timeout="1p" 2>&1)
  kube::test::if_has_string "${output_message}" 'Invalid timeout value'

  # cleanup
  kubectl delete pods valid-pod "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}
