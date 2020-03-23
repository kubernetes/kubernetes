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

run_clusterroles_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing clusterroles"

  # make sure the server was properly bootstrapped with clusterroles and bindings
  kube::test::get_object_assert clusterroles/cluster-admin "{{.metadata.name}}" 'cluster-admin'
  kube::test::get_object_assert clusterrolebindings/cluster-admin "{{.metadata.name}}" 'cluster-admin'

  # Pre-condition: no ClusterRole pod-admin exists
  output_message=$(! kubectl get clusterrole pod-admin 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'clusterroles.rbac.authorization.k8s.io "pod-admin" not found'
  # Dry-run test `kubectl create clusterrole`
  kubectl create "${kube_flags[@]:?}" clusterrole pod-admin --dry-run=client --verb=* --resource=pods
  kubectl create "${kube_flags[@]:?}" clusterrole pod-admin --dry-run=server --verb=* --resource=pods
  output_message=$(! kubectl get clusterrole pod-admin 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'clusterroles.rbac.authorization.k8s.io "pod-admin" not found'
  # test `kubectl create clusterrole`
  kubectl create "${kube_flags[@]:?}" clusterrole pod-admin --verb=* --resource=pods
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" '\*:'
  output_message=$(kubectl delete clusterrole pod-admin -n test 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'warning: deleting cluster-scoped resources'
  kube::test::if_has_string "${output_message}" 'clusterrole.rbac.authorization.k8s.io "pod-admin" deleted'

  kubectl create "${kube_flags[@]}" clusterrole pod-admin --verb=* --resource=pods
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" '\*:'
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert clusterrole/pod-admin "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kubectl create "${kube_flags[@]}" clusterrole resource-reader --verb=get,list --resource=pods,deployments.apps
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:get:list:'
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:deployments:'
  kube::test::get_object_assert clusterrole/resource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':apps:'
  kubectl create "${kube_flags[@]}" clusterrole resourcename-reader --verb=get,list --resource=pods --resource-name=foo
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kube::test::get_object_assert clusterrole/resourcename-reader "{{range.rules}}{{range.resourceNames}}{{.}}:{{end}}{{end}}" 'foo:'
  kubectl create "${kube_flags[@]}" clusterrole url-reader --verb=get --non-resource-url=/logs/* --non-resource-url=/healthz/*
  kube::test::get_object_assert clusterrole/url-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:'
  kube::test::get_object_assert clusterrole/url-reader "{{range.rules}}{{range.nonResourceURLs}}{{.}}:{{end}}{{end}}" '/logs/\*:/healthz/\*:'
  kubectl create "${kube_flags[@]}" clusterrole aggregation-reader --aggregation-rule="foo1=foo2"
  kube::test::get_object_assert clusterrole/aggregation-reader "{{${id_field:?}}}" 'aggregation-reader'

  # Pre-condition: no ClusterRoleBinding super-admin exists
  output_message=$(! kubectl get clusterrolebinding super-admin 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'clusterrolebindings.rbac.authorization.k8s.io "super-admin" not found'
  # Dry-run test `kubectl create clusterrolebinding`
  kubectl create "${kube_flags[@]}" clusterrolebinding super-admin --dry-run=client --clusterrole=admin --user=super-admin
  kubectl create "${kube_flags[@]}" clusterrolebinding super-admin --dry-run=server --clusterrole=admin --user=super-admin
  output_message=$(! kubectl get clusterrolebinding super-admin 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'clusterrolebindings.rbac.authorization.k8s.io "super-admin" not found'
  # test `kubectl create clusterrolebinding`
  # test `kubectl set subject clusterrolebinding`
  kubectl create "${kube_flags[@]}" clusterrolebinding super-admin --clusterrole=admin --user=super-admin
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  kubectl set subject --dry-run=client "${kube_flags[@]}" clusterrolebinding super-admin --user=foo
  kubectl set subject --dry-run=server "${kube_flags[@]}" clusterrolebinding super-admin --user=foo
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-admin --user=foo
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:foo:'
  kubectl create "${kube_flags[@]}" clusterrolebinding multi-users --clusterrole=admin --user=user-1 --user=user-2
  kube::test::get_object_assert clusterrolebinding/multi-users "{{range.subjects}}{{.name}}:{{end}}" 'user-1:user-2:'

  kubectl create "${kube_flags[@]}" clusterrolebinding super-group --clusterrole=admin --group=the-group
  kube::test::get_object_assert clusterrolebinding/super-group "{{range.subjects}}{{.name}}:{{end}}" 'the-group:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-group --group=foo
  kube::test::get_object_assert clusterrolebinding/super-group "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:'
  kubectl create "${kube_flags[@]}" clusterrolebinding multi-groups --clusterrole=admin --group=group-1 --group=group-2
  kube::test::get_object_assert clusterrolebinding/multi-groups "{{range.subjects}}{{.name}}:{{end}}" 'group-1:group-2:'

  kubectl create "${kube_flags[@]}" clusterrolebinding super-sa --clusterrole=admin --serviceaccount=otherns:sa-name
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:'
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:'
  kubectl set subject "${kube_flags[@]}" clusterrolebinding super-sa --serviceaccount=otherfoo:foo
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:otherfoo:'
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:'

  # test `kubectl set subject clusterrolebinding --all`
  kubectl set subject "${kube_flags[@]}" clusterrolebinding --all --user=test-all-user
  kube::test::get_object_assert clusterrolebinding/super-admin "{{range.subjects}}{{.name}}:{{end}}" 'super-admin:foo:test-all-user:'
  kube::test::get_object_assert clusterrolebinding/super-group "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:test-all-user:'
  kube::test::get_object_assert clusterrolebinding/super-sa "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:test-all-user:'

  # test `kubectl create rolebinding`
  # test `kubectl set subject rolebinding`
  kubectl create "${kube_flags[@]}" rolebinding admin --dry-run=client --clusterrole=admin --user=default-admin
  kubectl create "${kube_flags[@]}" rolebinding admin --dry-run=server --clusterrole=admin --user=default-admin
  output_message=$(! kubectl get rolebinding/admin 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" ' not found'
  kubectl create "${kube_flags[@]}" rolebinding admin --clusterrole=admin --user=default-admin
  kube::test::get_object_assert rolebinding/admin "{{.roleRef.kind}}" 'ClusterRole'
  kube::test::get_object_assert rolebinding/admin "{{range.subjects}}{{.name}}:{{end}}" 'default-admin:'
  kubectl set subject "${kube_flags[@]}" rolebinding admin --user=foo
  kube::test::get_object_assert rolebinding/admin "{{range.subjects}}{{.name}}:{{end}}" 'default-admin:foo:'

  kubectl create "${kube_flags[@]}" rolebinding localrole --role=localrole --group=the-group
  kube::test::get_object_assert rolebinding/localrole "{{.roleRef.kind}}" 'Role'
  kube::test::get_object_assert rolebinding/localrole "{{range.subjects}}{{.name}}:{{end}}" 'the-group:'
  kubectl set subject "${kube_flags[@]}" rolebinding localrole --group=foo
  kube::test::get_object_assert rolebinding/localrole "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:'

  kubectl create "${kube_flags[@]}" rolebinding sarole --role=localrole --serviceaccount=otherns:sa-name
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:'
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:'
  kubectl set subject "${kube_flags[@]}" rolebinding sarole --serviceaccount=otherfoo:foo
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.namespace}}:{{end}}" 'otherns:otherfoo:'
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:'

  # test `kubectl set subject rolebinding --all`
  kubectl set subject "${kube_flags[@]}" rolebinding --all --user=test-all-user
  kube::test::get_object_assert rolebinding/admin "{{range.subjects}}{{.name}}:{{end}}" 'default-admin:foo:test-all-user:'
  kube::test::get_object_assert rolebinding/localrole "{{range.subjects}}{{.name}}:{{end}}" 'the-group:foo:test-all-user:'
  kube::test::get_object_assert rolebinding/sarole "{{range.subjects}}{{.name}}:{{end}}" 'sa-name:foo:test-all-user:'

  set +o nounset
  set +o errexit
}

run_role_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing role"

  # Dry-run create
  kubectl create "${kube_flags[@]}" role pod-admin --dry-run=client --verb=* --resource=pods
  kubectl create "${kube_flags[@]}" role pod-admin --dry-run=server --verb=* --resource=pods
  output_message=$(! kubectl get role/pod-admin 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" ' not found'
  # Create Role from command (only resource)
  kubectl create "${kube_flags[@]}" role pod-admin --verb=* --resource=pods
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" '\*:'
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert role/pod-admin "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-pod-admin --verb=* --resource=invalid-resource 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"invalid-resource\""
  # Create Role from command (resource + group)
  kubectl create "${kube_flags[@]}" role group-reader --verb=get,list --resource=deployments.apps
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'deployments:'
  kube::test::get_object_assert role/group-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" 'apps:'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-group --verb=get,list --resource=deployments.invalid-group 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"deployments\" in group \"invalid-group\""
  # Create Role from command (resource / subresource)
  kubectl create "${kube_flags[@]}" role subresource-reader --verb=get,list --resource=pods/status
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods/status:'
  kube::test::get_object_assert role/subresource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  # Create Role from command (resource + group / subresource)
  kubectl create "${kube_flags[@]}" role group-subresource-reader --verb=get,list --resource=replicasets.apps/scale
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'replicasets/scale:'
  kube::test::get_object_assert role/group-subresource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" 'apps:'
  output_message=$(! kubectl create "${kube_flags[@]}" role invalid-group --verb=get,list --resource=rs.invalid-group/scale 2>&1)
  kube::test::if_has_string "${output_message}" "the server doesn't have a resource type \"rs\" in group \"invalid-group\""
  # Create Role from command (resource + resourcename)
  kubectl create "${kube_flags[@]}" role resourcename-reader --verb=get,list --resource=pods --resource-name=foo
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods:'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':'
  kube::test::get_object_assert role/resourcename-reader "{{range.rules}}{{range.resourceNames}}{{.}}:{{end}}{{end}}" 'foo:'
  # Create Role from command (multi-resources)
  kubectl create "${kube_flags[@]}" role resource-reader --verb=get,list --resource=pods/status,deployments.apps
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.verbs}}{{.}}:{{end}}{{end}}" 'get:list:get:list:'
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.resources}}{{.}}:{{end}}{{end}}" 'pods/status:deployments:'
  kube::test::get_object_assert role/resource-reader "{{range.rules}}{{range.apiGroups}}{{.}}:{{end}}{{end}}" ':apps:'

  set +o nounset
  set +o errexit
}
