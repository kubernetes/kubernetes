#!/usr/bin/env bash

# Copyright 2021 The Kubernetes Authors.
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

run_convert_tests() {
  set -o nounset
  set -o errexit

  ### Convert deployment YAML file locally without affecting the live deployment
  # Pre-condition: no deployments exist
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command
  # Create a deployment (revision 1)
  kubectl create -f hack/testdata/deployment-revision1.yaml "${kube_flags[@]:?}"
  kube::test::get_object_assert deployment "{{range.items}}{{${id_field:?}}}:{{end}}" 'nginx:'
  kube::test::get_object_assert deployment "{{range.items}}{{${image_field0:?}}}:{{end}}" "${IMAGE_DEPLOYMENT_R1}:"
  # Command
  output_message=$(kubectl convert --local -f hack/testdata/deployment-revision1.yaml --output-version=apps/v1beta1 -o yaml "${kube_flags[@]:?}")
  # Post-condition: apiVersion is still apps/v1 in the live deployment, but command output is the new value
  kube::test::get_object_assert 'deployment nginx' "{{ .apiVersion }}" 'apps/v1'
  kube::test::if_has_string "${output_message}" "apps/v1beta1"
  # Clean up
  kubectl delete deployment nginx "${kube_flags[@]:?}"

  ## Convert multiple busybox PODs recursively from directory of YAML files
  # Command
  output_message=$(! kubectl-convert -f hack/testdata/recursive/pod --recursive 2>&1 "${kube_flags[@]:?}")
  # Post-condition: busybox0 & busybox1 PODs are converted, and since busybox2 is malformed, it should error
  kube::test::if_has_string "${output_message}" "Object 'Kind' is missing"

  # check that convert command supports --template output
  output_message=$(kubectl-convert "${kube_flags[@]:?}" -f hack/testdata/deployment-revision1.yaml --output-version=apps/v1beta2 --template="{{ .metadata.name }}:")
  kube::test::if_has_string "${output_message}" 'nginx:'

  set +o nounset
  set +o errexit
}
