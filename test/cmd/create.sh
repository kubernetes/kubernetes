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

# Runs tests related to kubectl create --dry-run.
run_kubectl_create_dry_run_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create dry-run"

  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # dry-run create
  kubectl create --dry-run=client -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  kubectl create --dry-run=server -f hack/testdata/pod.yaml "${kube_flags[@]:?}"
  # check no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''

  set +o nounset
  set +o errexit
}

# Runs tests related to kubectl create --filename(-f) --selector(-l).
run_kubectl_create_filter_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl create filter"
  ## kubectl create -f with label selector should only create matching objects
  # Pre-Condition: no POD exists
  kube::test::get_object_assert pods "{{range.items}}{{${id_field:?}}}:{{end}}" ''
  # create
  kubectl create -l unique-label=bingbang -f hack/testdata/filter "${kube_flags[@]:?}"
  # check right pod exists
  kube::test::get_object_assert 'pods selector-test-pod' "{{${labels_field:?}.name}}" 'selector-test-pod'
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
  ! kubectl create || exit 1

  # Posting a pod to namespaces should fail.  Also tests --raw forcing the post location
  grep -q 'the object provided is unrecognized (must be of type Namespace)' <<< "$( kubectl create "${kube_flags[@]}" --raw /api/v1/namespaces -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml --v=8 2>&1 )"

  grep -q "raw and --edit are mutually exclusive" <<< "$( kubectl create "${kube_flags[@]}" --raw /api/v1/namespaces -f test/fixtures/doc-yaml/admin/limitrange/valid-pod.yaml --edit 2>&1 )"

  set +o nounset
  set +o errexit
}

# Runs kubectl create job tests
run_create_job_tests() {
    set -o nounset
    set -o errexit

    create_and_use_new_namespace

    # Test kubectl create job
    kubectl create job test-job --image=registry.k8s.io/nginx:test-cmd
    # Post-Condition: job nginx is created
    kube::test::get_object_assert 'job test-job' "{{${image_field0:?}}}" 'registry.k8s.io/nginx:test-cmd'
    # Clean up
    kubectl delete job test-job "${kube_flags[@]}"

    # Test kubectl create job with command
    kubectl create job test-job-pi "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(20)'
    kube::test::get_object_assert 'job test-job-pi' "{{$image_field0}}" "$IMAGE_PERL"
    # Clean up
    kubectl delete job test-job-pi

    # Test kubectl create job from cronjob
    # Pre-Condition: create a cronjob
    kubectl create cronjob test-pi --schedule="* */5 * * *" "--image=$IMAGE_PERL" -- perl -Mbignum=bpi -wle 'print bpi(10)'
    kubectl create job my-pi --from=cronjob/test-pi
    # Post-condition: container args contain expected command
    output_message=$(kubectl get job my-pi -o go-template='{{(index .spec.template.spec.containers 0).command}}' "${kube_flags[@]}")
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
  # Pre-Condition: No configmaps with name=test-the-map, no Deployment, Service exist
  kube::test::get_object_assert 'configmaps --field-selector=metadata.name=test-the-map' "{{range.items}}{{${id_field:?}}}:{{end}}" ''
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

has_one_of_error_message() {
  local message=$1
  local match1=$2
  local match2=$3

  if (grep -q "${match1}" <<< "${message}") || (grep -q "${match2}" <<< "${message}"); then
    echo "Successful"
    echo "message:${message}"
    echo "has either:${match1}"
    echo "or:${match2}"
    return 0
  else
    echo "FAIL!"
    echo "message:${message}"
    echo "has neither:${match1}"
    echo "nor:${match2}"
    caller
    return 1
  fi
}

# Runs tests related to kubectl create --validate
run_kubectl_create_validate_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace

   ## test --validate no value expects default strict is used
   kube::log::status "Testing kubectl create --validate"
   # create and verify
   output_message=$(! kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate 2>&1)
   has_one_of_error_message "${output_message}" 'strict decoding error' 'error validating data'

  ## test --validate=true
  kube::log::status "Testing kubectl create --validate=true"
  # create and verify
  output_message=$(! kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=true 2>&1)
  has_one_of_error_message "${output_message}" 'strict decoding error' 'error validating data'

  ## test  --validate=false
  kube::log::status "Testing kubectl create --validate=false"
  # create and verify
  output_message=$(kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=false)
  kube::test::if_has_string "${output_message}" "deployment.apps/invalid-nginx-deployment created"
  # cleanup
  kubectl delete deployment invalid-nginx-deployment

  ## test --validate=strict
  kube::log::status "Testing kubectl create --validate=strict"
  # create and verify
  output_message=$(! kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=strict 2>&1)
  has_one_of_error_message "${output_message}" 'strict decoding error' 'error validating data'

  ## test --validate=warn
  kube::log::status "Testing kubectl create --validate=warn"
  # create and verify
  output_message=$(kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=warn)
  kube::test::if_has_string "${output_message}" "deployment.apps/invalid-nginx-deployment created"
  # cleanup
  kubectl delete deployment invalid-nginx-deployment

  ## test  --validate=ignore
  kube::log::status "Testing kubectl create --validate=ignore"
  # create and verify
  output_message=$(kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=ignore)
  kube::test::if_has_string "${output_message}" "deployment.apps/invalid-nginx-deployment created"
  # cleanup
  kubectl delete deployment invalid-nginx-deployment

  ## test default is strict validation
  kube::log::status "Testing kubectl create"
  # create and verify
  output_message=$(! kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml 2>&1)
  has_one_of_error_message "${output_message}" 'strict decoding error' 'error validating data'

  ## test invalid validate value
  kube::log::status "Testing kubectl create --validate=foo"
  # create and verify
  output_message=$(! kubectl create -f hack/testdata/invalid-deployment-unknown-and-duplicate-fields.yaml --validate=foo 2>&1)
  kube::test::if_has_string "${output_message}" 'invalid - validate option "foo"'

  set +o nounset
  set +o errexit
}
