#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

run_kubectl_debug_pod_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl debug (pod tests)"

  ### Pod Troubleshooting by Copy

  # Pre-Condition: Pod "nginx" is created
  kubectl run target "--image=${IMAGE_NGINX:?}" "${kube_flags[@]:?}"
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target:'
  # Command: create a copy of target with a new debug container
  kubectl debug target -it --copy-to=target-copy --image=busybox --container=debug-container --attach=false "${kube_flags[@]:?}"
  # Post-Conditions
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target:target-copy:'
  kube::test::get_object_assert pod/target-copy '{{range.spec.containers}}{{.name}}:{{end}}' 'target:debug-container:'
  kube::test::get_object_assert pod/target-copy '{{range.spec.containers}}{{.image}}:{{end}}' "${IMAGE_NGINX:?}:busybox:"
  # Clean up
  kubectl delete pod target target-copy "${kube_flags[@]:?}"
 
  # Pre-Condition: Pod "nginx" is created
  kubectl run target "--image=${IMAGE_NGINX:?}" "${kube_flags[@]:?}"
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target:'
  # Command: create a copy of target with a new debug container replacing the previous pod
  kubectl debug target -it --copy-to=target-copy --image=busybox --container=debug-container --attach=false --replace "${kube_flags[@]:?}"
  # Post-Conditions
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target-copy:'
  kube::test::get_object_assert pod/target-copy '{{range.spec.containers}}{{.name}}:{{end}}' 'target:debug-container:'
  kube::test::get_object_assert pod/target-copy '{{range.spec.containers}}{{.image}}:{{end}}' "${IMAGE_NGINX:?}:busybox:"
  # Clean up
  kubectl delete pod target-copy "${kube_flags[@]:?}"

  # Pre-Condition: Pod "nginx" is created
  kubectl run target "--image=${IMAGE_NGINX:?}" "${kube_flags[@]:?}"
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target:'
  kube::test::get_object_assert pod/target '{{(index .spec.containers 0).name}}' 'target'
  # Command: copy the pod and replace the image of an existing container
  kubectl debug target --image=busybox --container=target --copy-to=target-copy "${kube_flags[@]:?}" -- sleep 1m
  # Post-Conditions
  kube::test::get_object_assert pod "{{range.items}}{{${id_field:?}}}:{{end}}" 'target:target-copy:'
  kube::test::get_object_assert pod/target-copy "{{(len .spec.containers)}}:{{${image_field:?}}}" '1:busybox'
  # Clean up
  kubectl delete pod target target-copy "${kube_flags[@]:?}"

  set +o nounset
  set +o errexit
}

run_kubectl_debug_node_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl debug (pod tests)"

  ### Node Troubleshooting by Privileged Container

  # Pre-Condition: Pod "nginx" is created
  kube::test::get_object_assert nodes "{{range.items}}{{${id_field:?}}}:{{end}}" '127.0.0.1:'
  # Command: create a new node debugger pod
  output_message=$(kubectl debug node/127.0.0.1 --image=busybox --attach=false "${kube_flags[@]:?}" -- true)
  # Post-Conditions
  kube::test::get_object_assert pod "{{(len .items)}}" '1'
  debugger=$(kubectl get pod -o go-template="{{(index .items 0)${id_field:?}}}")
  kube::test::if_has_string "${output_message:?}" "${debugger:?}"
  kube::test::get_object_assert "pod/${debugger:?}" "{{${image_field:?}}}" 'busybox'
  kube::test::get_object_assert "pod/${debugger:?}" '{{.spec.nodeName}}' '127.0.0.1'
  kube::test::get_object_assert "pod/${debugger:?}" '{{.spec.hostIPC}}' 'true'
  kube::test::get_object_assert "pod/${debugger:?}" '{{.spec.hostNetwork}}' 'true'
  kube::test::get_object_assert "pod/${debugger:?}" '{{.spec.hostPID}}' 'true'
  kube::test::get_object_assert "pod/${debugger:?}" '{{(index (index .spec.containers 0).volumeMounts 0).mountPath}}' '/host'
  kube::test::get_object_assert "pod/${debugger:?}" '{{(index .spec.volumes 0).hostPath.path}}' '/'
  # Clean up
  # pod.spec.nodeName is set by kubectl debug node which causes the delete to hang,
  # presumably waiting for a kubelet that's not present. Force the delete.
  kubectl delete --force pod "${debugger:?}" "${kube_flags[@]:?}"
 
  set +o nounset
  set +o errexit
}
