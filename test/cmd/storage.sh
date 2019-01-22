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

run_persistent_volumes_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing persistent volumes"

  ### Create and delete persistent volume examples
  # Pre-condition: no persistent volumes currently exist
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/local-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0001:'
  kubectl delete pv pv0001 "${kube_flags[@]}"
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/local-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0002:'
  kubectl delete pv pv0002 "${kube_flags[@]}"
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/volumes/gce.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" 'pv0003:'
  kubectl delete pv pv0003 "${kube_flags[@]}"
  # Post-condition: no PVs
  kube::test::get_object_assert pv "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_persistent_volume_claims_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing persistent volumes claims"

  ### Create and delete persistent volume claim examples
  # Pre-condition: no persistent volume claims currently exist
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-01.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-1:'
  kubectl delete pvc myclaim-1 "${kube_flags[@]}"

  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-02.yaml "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-2:'
  kubectl delete pvc myclaim-2 "${kube_flags[@]}"

  kubectl create -f test/fixtures/doc-yaml/user-guide/persistent-volumes/claims/claim-03.json "${kube_flags[@]}"
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" 'myclaim-3:'
  kubectl delete pvc myclaim-3 "${kube_flags[@]}"
  # Post-condition: no PVCs
  kube::test::get_object_assert pvc "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

run_storage_class_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing storage class"

  ### Create and delete storage class
  # Pre-condition: no storage classes currently exist
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -f - "${kube_flags[@]}" << __EOF__
{
  "kind": "StorageClass",
  "apiVersion": "storage.k8s.io/v1",
  "metadata": {
  "name": "storage-class-name"
  },
  "provisioner": "kubernetes.io/fake-provisioner-type",
  "parameters": {
  "zone":"us-east-1b",
  "type":"ssd"
  }
}
__EOF__
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" 'storage-class-name:'
  kube::test::get_object_assert sc "{{range.items}}{{$id_field}}:{{end}}" 'storage-class-name:'
  kubectl delete storageclass storage-class-name "${kube_flags[@]}"
  # Post-condition: no storage classes
  kube::test::get_object_assert storageclass "{{range.items}}{{$id_field}}:{{end}}" ''

  set +o nounset
  set +o errexit

}
