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

# Runs tests related to kubectl create --filename(-f) --selector(-l).
run_kubectl_create_filter_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create filter"
  ## kubectl create -f with label selector should only create matching objects
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{$id_field}}:{{end}}" ''
  # create
  kubectl create -l unique-label=bingbang -f hack/testdata/filter "${kube_flags[@]}"
  # check right pod exists
  kube::test::get_object_assert 'pods selector-test-pod' "{{${labels_field}.name}}" 'selector-test-pod'
  # check wrong pod doesn't exist
  output_message=$(! kubectl get pods selector-test-pod-dont-apply 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'pods "selector-test-pod-dont-apply" not found'
  # cleanup
  kubectl delete pods selector-test-pod

  set +o nounset
  set +o errexit
}

run_kubectl_create_error_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create with error"

  # Passing no arguments to create is an error
  ! kubectl create

  ## kubectl create should not panic on empty string lists in a template
  ERROR_FILE="${KUBE_TEMP}/validation-error"
  kubectl create -f hack/testdata/invalid-rc-with-empty-args.yaml "${kube_flags[@]}" 2> "${ERROR_FILE}" || true
  # Post-condition: should get an error reporting the empty string
  if grep -q "unknown object type \"nil\" in ReplicationController" "${ERROR_FILE}"; then
    kube::log::status "\"kubectl create with empty string list returns error as expected: $(cat ${ERROR_FILE})"
  else
    kube::log::status "\"kubectl create with empty string list returns unexpected error or non-error: $(cat ${ERROR_FILE})"
    exit 1
  fi
  rm "${ERROR_FILE}"

  # Posting a pod to namespaces should fail.  Also tests --raw forcing the post location
  [ "$( kubectl convert -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml -o json | kubectl create "${kube_flags[@]}" --raw /api/v1/namespaces -f - --v=8 2>&1 | grep 'cannot be handled as a Namespace: converting (v1.Pod)')" ]

  [ "$( kubectl create "${kube_flags[@]}" --raw /api/v1/namespaces -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml --edit 2>&1 | grep 'raw and --edit are mutually exclusive')" ]

  set +o nounset
  set +o errexit
}

# Runs kubectl create job tests
run_create_job_tests() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace

    # Test kubectl create job
    kubectl create job test-job --image=k8s.gcr.io/nginx:test-cmd
    # Post-Condition: job nginx is created
    kube::test::get_object_assert 'job test-job' "{{$image_field0}}" 'k8s.gcr.io/nginx:test-cmd'
    # Clean up
    kubectl delete job test-job "${kube_flags[@]}"

    # Test kubectl create job with command
    kubectl create job test-job-pi "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)'
    kube::test::get_object_assert 'job test-job-pi' "{{$image_field0}}" $IMAGE_PERL
    # Clean up
    kubectl delete job test-job-pi

    # Test kubectl create job from cronjob
    # Pre-Condition: create a cronjob
    kubectl run test-pi --schedule="* */5 * * *" --generator=cronjob/v1beta1 "--image=$IMAGE_PERL" --restart=OnFailure -- perl -Mbignum=bpi -wle 'print bpi(10)'
    kubectl create job my-pi --from=cronjob/test-pi
    # Post-condition: container args contain expected command
    output_message=$(kubectl get job my-pi -o go-template='{{(index .spec.template.spec.containers 0).args}}' "${kube_flags[@]}")
    kube::test::if_has_string "${output_message}" "perl -Mbignum=bpi -wle print bpi(10)"

    # Clean up
    kubectl delete job my-pi
    kubectl delete cronjob test-pi

    set +o nounset
    set +o errexit
}

run_kubectl_create_kustomization_directory_tests() {
  set -o nounset
  set -o errexit

  ## kubectl create -k <dir> for kustomization directory
  # Pre-condition: no ConfigMap, Deployment, Service exist
  kube::test::get_object_assert configmaps "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert deployment "{{range.items}}{{$id_field}}:{{end}}" ''
  kube::test::get_object_assert services "{{range.items}}{{$id_field}}:{{end}}" ''
  # Command
  kubectl create -k hack/testdata/kustomize
  # Post-condition: test-the-map, test-the-deployment, test-the-service exist

  # Check that all items in the list are printed
  kube::test::get_object_assert 'configmap test-the-map' "{{${id_field}}}" 'test-the-map'
  kube::test::get_object_assert 'deployment test-the-deployment' "{{${id_field}}}" 'test-the-deployment'
  kube::test::get_object_assert 'service test-the-service' "{{${id_field}}}" 'test-the-service'

  # cleanup
  kubectl delete -k hack/testdata/kustomize

  set +o nounset
  set +o errexit
}