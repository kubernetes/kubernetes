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

run_certificates_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing certificates"

  # approve
  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]:?}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate approve foo "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Approved'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{${id_field:?}}}{{end}}" ''

  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate approve -f hack/testdata/csr.yml "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Approved'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  # deny
  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate deny foo "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Denied'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  kubectl create -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' ''
  kubectl certificate deny -f hack/testdata/csr.yml "${kube_flags[@]}"
  kubectl get csr "${kube_flags[@]}" -o json
  kube::test::get_object_assert 'csr/foo' '{{range.status.conditions}}{{.type}}{{end}}' 'Denied'
  kubectl delete -f hack/testdata/csr.yml "${kube_flags[@]}"
  kube::test::get_object_assert csr "{{range.items}}{{$id_field}}{{end}}" ''

  set +o nounset
  set +o errexit
}
