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

# Runs tests related to kubectl apply.
run_kubectl_apply_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl apply"
  ## kubectl apply should create the resource that doesn't exist yet
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command: apply a pod "test-pod" (doesn't exist) should create this pod
  kubectl apply -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field:?}.name}}" 'test-pod-label'
  # Post-Condition: pod "test-pod" has configuration annotation
  grep -q kubectl.kubernetes.io/last-applied-configuration <<< "$(kubectl get pods test-pod -o yaml "${kube_flags[@]:?}")"
  # Clean up
  kubectl delete pods test-pod "${kube_flags[@]:?}"


  ## kubectl apply should be able to clear defaulted fields.
  # Pre-Condition: no deployment exists
  kube::test::get_object_assert deployments "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command: apply a deployment "test-deployment-retainkeys" (doesn't exist) should create this deployment
  kubectl apply -f hack/testdata/retainKeys/deployment/deployment-before.yaml "${kube_flags[@]:?}"
  # Post-Condition: deployment "test-deployment-retainkeys" created
  kube::test::get_object_assert deployments "{{range.items}}{{${id_field:?}}}{{end}}" 'test-deployment-retainkeys'
  # Post-Condition: deployment "test-deployment-retainkeys" has defaulted fields
  grep -q RollingUpdate <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  grep -q maxSurge <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  grep -q maxUnavailable <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  grep -q emptyDir <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  # Command: apply a deployment "test-deployment-retainkeys" should clear
  # defaulted fields and successfully update the deployment
  [[ "$(kubectl apply -f hack/testdata/retainKeys/deployment/deployment-after.yaml "${kube_flags[@]:?}")" ]]
  # Post-Condition: deployment "test-deployment-retainkeys" has updated fields
  grep -q Recreate <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  ! grep -q RollingUpdate <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  grep -q hostPath <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  ! grep -q emptyDir <<< "$(kubectl get deployments test-deployment-retainkeys -o yaml "${kube_flags[@]:?}")"
  # Clean up
  kubectl delete deployments test-deployment-retainkeys "${kube_flags[@]:?}"


  ## kubectl apply -f with label selector should only apply matching objects
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply
  kubectl apply -l unique-label=bingbang -f hack/testdata/filter "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods selector-test-pod' "{{${labels_field:?}.name}}" 'selector-test-pod'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods selector-test-pod-dont-apply 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'pods "selector-test-pod-dont-apply" not found'
  # cleanup
  kubectl delete pods selector-test-pod

  ## kubectl apply --server-dry-run
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # apply dry-run
  kubectl apply --server-dry-run -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # No pod exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply non dry-run creates the pod
  kubectl apply -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # apply changes
  kubectl apply --server-dry-run -f hack/testdata/pod-apply.yaml "${kube_flags[@]:?}"
  # Post-Condition: label still has initial value
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field:?}.name}}" 'test-pod-label'

  # clean-up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]:?}"

  ## kubectl apply dry-run on CR
  # Create CRD
  kubectl "${kube_flags_with_token[@]:?}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "resources.mygroup.example.com"
  },
  "spec": {
    "group": "mygroup.example.com",
    "version": "v1alpha1",
    "scope": "Namespaced",
    "names": {
      "plural": "resources",
      "singular": "resource",
      "kind": "Kind",
      "listKind": "KindList"
    }
  }
}
__EOF__

  # Dry-run create the CR
  kubectl "${kube_flags[@]:?}" apply --server-dry-run -f hack/testdata/CRD/resource.yaml "${kube_flags[@]:?}"
  # Make sure that the CR doesn't exist
  ! kubectl "${kube_flags[@]:?}" get resource/myobj

  # clean-up
  kubectl "${kube_flags[@]:?}" delete customresourcedefinition resources.mygroup.example.com

  ## kubectl apply --prune
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # apply a
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods b 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'pods "b" not found'

  # apply b
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/b.yaml "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods b' "{{${id_field:?}}}" 'b'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods a 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'pods "a" not found'

  # cleanup
  kubectl delete pods b

  # same thing without prune for a sanity check
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # apply a
  kubectl apply -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods b 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'pods "b" not found'

  # apply b
  kubectl apply -l prune-group=true -f hack/testdata/prune/b.yaml "${kube_flags[@]:?}"
  # check both pods exist
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  kube::test::get_object_assert 'pods b' "{{${id_field:?}}}" 'b'
  # check wrong pod doesn't exist

  # cleanup
  kubectl delete pod/a pod/b

  ## kubectl apply --prune requires a --all flag to select everything
  output_message=$(! kubectl apply --prune -f hack/testdata/prune 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" \
    'all resources selected for prune without explicitly passing --all'
  # should apply everything
  kubectl apply --all --prune -f hack/testdata/prune
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  kube::test::get_object_assert 'pods b' "{{${id_field:?}}}" 'b'
  kubectl delete pod/a pod/b

  ## kubectl apply --prune should fallback to delete for non reapable types
  kubectl apply --all --prune -f hack/testdata/prune-reap/a.yml 2>&1 "${kube_flags[@]:?}"
  kube::test::get_object_assert 'pvc a-pvc' "{{${id_field:?}}}" 'a-pvc'
  kubectl apply --all --prune -f hack/testdata/prune-reap/b.yml 2>&1 "${kube_flags[@]:?}"
  kube::test::get_object_assert 'pvc b-pvc' "{{${id_field:?}}}" 'b-pvc'
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  kubectl delete pvc b-pvc 2>&1 "${kube_flags[@]:?}"

  ## kubectl apply --prune --prune-whitelist
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply pod a
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/a.yaml "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  # apply svc and don't prune pod a by overwriting whitelist
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/svc.yaml --prune-whitelist core/v1/Service 2>&1 "${kube_flags[@]:?}"
  kube::test::get_object_assert 'service prune-svc' "{{${id_field:?}}}" 'prune-svc'
  kube::test::get_object_assert 'pods a' "{{${id_field:?}}}" 'a'
  # apply svc and prune pod a with default whitelist
  kubectl apply --prune -l prune-group=true -f hack/testdata/prune/svc.yaml 2>&1 "${kube_flags[@]:?}"
  kube::test::get_object_assert 'service prune-svc' "{{${id_field:?}}}" 'prune-svc'
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # cleanup
  kubectl delete svc prune-svc 2>&1 "${kube_flags[@]:?}"


  ## kubectl apply -f some.yml --force
  # Pre-condition: no service exists
  kube::test::get_object_assert services "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply service a
  kubectl apply -f hack/testdata/service-revision1.yaml "${kube_flags[@]:?}"
  # check right service exists
  kube::test::get_object_assert 'services a' "{{${id_field:?}}}" 'a'
  # change immutable field and apply service a
  output_message=$(! kubectl apply -f hack/testdata/service-revision2.yaml 2>&1 "${kube_flags[@]:?}")
  kube::test::if_has_string "${output_message}" 'field is immutable'
  # apply --force to recreate resources for immutable fields
  kubectl apply -f hack/testdata/service-revision2.yaml --force "${kube_flags[@]:?}"
  # check immutable field exists
  kube::test::get_object_assert 'services a' "{{.spec.clusterIP}}" '10.0.0.12'
  # cleanup
  kubectl delete -f hack/testdata/service-revision2.yaml "${kube_flags[@]:?}"

  ## kubectl apply -k somedir
  kubectl apply -k hack/testdata/kustomize
  kube::test::get_object_assert 'configmap test-the-map' "{{${id_field}}}" 'test-the-map'
  kube::test::get_object_assert 'deployment test-the-deployment' "{{${id_field}}}" 'test-the-deployment'
  kube::test::get_object_assert 'service test-the-service' "{{${id_field}}}" 'test-the-service'
  # cleanup
  kubectl delete -k hack/testdata/kustomize

  ## kubectl apply --kustomize somedir
  kubectl apply --kustomize hack/testdata/kustomize
  kube::test::get_object_assert 'configmap test-the-map' "{{${id_field}}}" 'test-the-map'
  kube::test::get_object_assert 'deployment test-the-deployment' "{{${id_field}}}" 'test-the-deployment'
  kube::test::get_object_assert 'service test-the-service' "{{${id_field}}}" 'test-the-service'
  # cleanup
  kubectl delete --kustomize hack/testdata/kustomize

  set +o nounset
  set +o errexit
}

# Runs tests related to kubectl apply (server-side)
run_kubectl_apply_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl apply --experimental-server-side"
  ## kubectl apply should create the resource that doesn't exist yet
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # Command: apply a pod "test-pod" (doesn't exist) should create this pod
  kubectl apply --experimental-server-side -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # Post-Condition: pod "test-pod" is created
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field:?}.name}}" 'test-pod-label'
  # Clean up
  kubectl delete pods test-pod "${kube_flags[@]:?}"

  ## kubectl apply --server-dry-run
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  # apply dry-run
  kubectl apply --experimental-server-side --server-dry-run -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # No pod exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # apply non dry-run creates the pod
  kubectl apply --experimental-server-side -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # apply changes
  kubectl apply --experimental-server-side --server-dry-run -f hack/testdata/pod-apply.yaml "${kube_flags[@]:?}"
  # Post-Condition: label still has initial value
  kube::test::get_object_assert 'pods test-pod' "{{${labels_field:?}.name}}" 'test-pod-label'

  # clean-up
  kubectl delete -f hack/testdata/pod.yaml "${kube_flags[@]:?}"

  ## kubectl apply dry-run on CR
  # Create CRD
  kubectl "${kube_flags_with_token[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "resources.mygroup.example.com"
  },
  "spec": {
    "group": "mygroup.example.com",
    "version": "v1alpha1",
    "scope": "Namespaced",
    "names": {
      "plural": "resources",
      "singular": "resource",
      "kind": "Kind",
      "listKind": "KindList"
    }
  }
}
__EOF__

  # Dry-run create the CR
  kubectl "${kube_flags[@]:?}" apply --experimental-server-side --server-dry-run -f hack/testdata/CRD/resource.yaml "${kube_flags[@]:?}"
  # Make sure that the CR doesn't exist
  ! kubectl "${kube_flags[@]:?}" get resource/myobj

  # clean-up
  kubectl "${kube_flags[@]:?}" delete customresourcedefinition resources.mygroup.example.com

  set +o nounset
  set +o errexit
}
