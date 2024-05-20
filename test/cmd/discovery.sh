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

run_RESTMapper_evaluation_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing RESTMapper"

  RESTMAPPER_ERROR_FILE="${KUBE_TEMP}/restmapper-error"

  ### Non-existent resource type should give a recognizable error
  # Pre-condition: None
  # Command
  kubectl get "${kube_flags[@]:?}" unknownresourcetype 2>"${RESTMAPPER_ERROR_FILE}" || true
  if grep -q "the server doesn't have a resource type" "${RESTMAPPER_ERROR_FILE}"; then
    kube::log::status "\"kubectl get unknownresourcetype\" returns error as expected: $(cat "${RESTMAPPER_ERROR_FILE}")"
  else
    kube::log::status "\"kubectl get unknownresourcetype\" returns unexpected error or non-error: $(cat "${RESTMAPPER_ERROR_FILE}")"
    exit 1
  fi
  rm "${RESTMAPPER_ERROR_FILE}"
  # Post-condition: None

  set +o nounset
  set +o errexit
}

run_assert_short_name_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing assert short name"

  kube::log::status "Testing propagation of short names for resources"
  output_message=$(kubectl get --raw=/api/v1)

  ## test if a short name is exported during discovery
  kube::test::if_has_string "${output_message}" '{"name":"configmaps","singularName":"configmap","namespaced":true,"kind":"ConfigMap","verbs":\["create","delete","deletecollection","get","list","patch","update","watch"\],"shortNames":\["cm"\],"storageVersionHash":'

  # check that there is no pod with the name test-crd-example
  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

    kubectl create -f - << __EOF__
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: examples.test.com
spec:
  group: test.com
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                test:
                  type: string
  names:
    plural: examples
    singular: example
    shortNames:
      - pod
    kind: Example
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert customresourcedefinitions "{{range.items}}{{if eq ${id_field:?} \"examples.test.com\"}}{{$id_field}}:{{end}}{{end}}" 'examples.test.com:'

  kubectl create -f - << __EOF__
apiVersion: test.com/v1
kind: Example
metadata:
  name: test-crd-example
spec:
  test: test
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert examples "{{range.items}}{{${id_field:?}}}:{{end}}" 'test-crd-example:'

  output_message=$(kubectl get examples)
  kube::test::if_has_string "${output_message}" "test-crd-example"

  # test that get pod returns v1/pod instead crd
  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

  # invalidate cache and assure that still correct resource is shown
  kubectl api-resources

  # retest the above cases after invalidating cache
  output_message=$(kubectl get examples)
  kube::test::if_has_string "${output_message}" "test-crd-example"

  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

  # Cleanup
  kubectl delete examples/test-crd-example
  kubectl delete customresourcedefinition examples.test.com

  set +o nounset
  set +o errexit
}

run_assert_singular_name_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing assert singular name"

  # check that there is no pod with the name test-crd-example
  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

  kubectl create -f - << __EOF__
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: examples.test.com
spec:
  group: test.com
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                test:
                  type: string
  names:
    plural: examples
    singular: pod
    kind: Example
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert customresourcedefinitions "{{range.items}}{{if eq ${id_field:?} \"examples.test.com\"}}{{$id_field}}:{{end}}{{end}}" 'examples.test.com:'

  kubectl create -f - << __EOF__
apiVersion: test.com/v1
kind: Example
metadata:
  name: test-crd-example
spec:
  test: test
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert examples "{{range.items}}{{$id_field}}:{{end}}" 'test-crd-example:'

  output_message=$(kubectl get examples)
  kube::test::if_has_string "${output_message}" "test-crd-example"

  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

  # invalidate cache and assure that still correct resource is shown
  kubectl api-resources

  output_message=$(kubectl get examples)
  kube::test::if_has_string "${output_message}" "test-crd-example"

  output_message=$(kubectl get pod)
  kube::test::if_has_not_string "${output_message}" "test-crd-example"

  # Cleanup
  kubectl delete examples/test-crd-example
  kubectl delete customresourcedefinition examples.test.com

  set +o nounset
  set +o errexit
}

run_assert_categories_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing propagation of categories for resources"
  output_message=$(kubectl get --raw=/api/v1 | grep -o '"name":"pods"[^}]*}')
  kube::test::if_has_string "${output_message}" '"categories":\["all"\]'

  set +o nounset
  set +o errexit
}

run_resource_aliasing_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing resource aliasing"
  kubectl create -f test/e2e/testing-manifests/statefulset/cassandra/controller.yaml "${kube_flags[@]}"
  kubectl create -f test/e2e/testing-manifests/statefulset/cassandra/service.yaml "${kube_flags[@]}"

  object="all -l app=cassandra"
  request="{{range.items}}{{range .metadata.labels}}{{.}}:{{end}}{{end}}"

  # all 4 cassandra's might not be in the request immediately...
  # :? suffix is for possible service.kubernetes.io/headless
  # label with "" value
  kube::test::get_object_assert "$object" "$request" '(cassandra:){2}(cassandra:(cassandra::?)?)?'

  kubectl delete all -l app=cassandra "${kube_flags[@]}"

  set +o nounset
  set +o errexit
}

run_crd_deletion_recreation_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing resource creation, deletion, and re-creation"

  output_message=$(kubectl apply -f hack/testdata/CRD/example-crd-1-cluster-scoped.yaml)
  kube::test::if_has_string "${output_message}" 'created'
  output_message=$(kubectl apply -f hack/testdata/CRD/example-crd-1-cluster-scoped-resource.yaml)
  kube::test::if_has_string "${output_message}" 'created'
  output_message=$(kubectl delete -f hack/testdata/CRD/example-crd-1-cluster-scoped.yaml)
  kube::test::if_has_string "${output_message}" 'deleted'
  # Invalidate local cache because cluster scoped CRD in cache is stale.
  # Invalidation of cache may take up to 6 hours and we are manually
  # invalidate cache and expect that scope changed CRD should be created without problem.
  kubectl api-resources
  output_message=$(kubectl apply -f hack/testdata/CRD/example-crd-1-namespaced.yaml)
  kube::test::if_has_string "${output_message}" 'created'
  output_message=$(kubectl apply -f hack/testdata/CRD/example-crd-1-namespaced-resource.yaml)
  kube::test::if_has_string "${output_message}" 'created'

  # Cleanup
  kubectl delete -f hack/testdata/CRD/example-crd-1-namespaced-resource.yaml
  kubectl delete -f hack/testdata/CRD/example-crd-1-namespaced.yaml

  set +o nounset
  set +o errexit
}

run_kubectl_explain_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl(v1:explain)"
  kubectl explain pods
  # shortcuts work
  kubectl explain po
  kubectl explain po.status.message
  # cronjob work
  kubectl explain cronjob

  set +o nounset
  set +o errexit
}

run_swagger_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing swagger"

  # Verify schema
  file="${KUBE_TEMP}/schema.json"
  curl -kfsS -H 'Authorization: Bearer admin-token' "https://127.0.0.1:${SECURE_API_PORT}/openapi/v2" > "${file}"
  grep -q "list of returned" "${file}"
  grep -q "List of services" "${file}"
  grep -q "Watch for changes to the described resources" "${file}"

  set +o nounset
  set +o errexit
}

run_ambiguous_shortname_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing ambiguous short name"

  kubectl create -f - << __EOF__
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: foos.bar.com
spec:
  group: bar.com
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                test:
                  type: string
  names:
    plural: foos
    singular: foo
    shortNames:
      - exmp
    kind: Foo
    categories:
      - all
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert customresourcedefinitions "{{range.items}}{{if eq ${id_field:?} \"foos.bar.com\"}}{{$id_field}}:{{end}}{{end}}" 'foos.bar.com:'

  kubectl create -f - << __EOF__
apiVersion: bar.com/v1
kind: Foo
metadata:
  name: test-crd-foo
spec:
  test: test
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert foos "{{range.items}}{{$id_field}}:{{end}}" 'test-crd-foo:'

  output_message=$(kubectl get exmp)
  kube::test::if_has_string "${output_message}" "test-crd-foo"

  kubectl create -f - << __EOF__
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: examples.test.com
spec:
  group: test.com
  scope: Namespaced
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                test:
                  type: string
  names:
    plural: examples
    singular: example
    shortNames:
      - exmp
    kind: Example
__EOF__

  # Test that we can list this new custom resource
  kube::test::wait_object_assert customresourcedefinitions "{{range.items}}{{if eq ${id_field:?} \"examples.test.com\"}}{{$id_field}}:{{end}}{{end}}" 'examples.test.com:'

  output_message=$(kubectl  get examples 2>&1 "${kube_flags[@]}")
  kube::test::if_has_string "${output_message}" 'No resources found'

  output_message=$(kubectl get exmp 2>&1)
  kube::test::if_has_string "${output_message}" "test-crd-foo"
  kube::test::if_has_string "${output_message}" "short name \"exmp\" could also match lower priority resource examples.test.com"

  # Cleanup
  kubectl delete foos/test-crd-foo
  kubectl delete customresourcedefinition foos.bar.com
  kubectl delete customresourcedefinition examples.test.com

  set +o nounset
  set +o errexit
}

run_explain_crd_with_additional_properties_tests() {
  set -o nounset
  set -o errexit

  # create_and_use_new_namespace
  kube::log::status "Testing explain with custom CRD that uses additionalProperties as non boolean field"

  output_message=$(kubectl apply -f hack/testdata/CRD/example-crd-with-additionalfields.yaml)
  kube::test::if_has_string "${output_message}" 'created'

  kube::test::wait_object_assert customresourcedefinitions "{{range.items}}{{if eq ${id_field:?} \"mock-resources.test.com\"}}{{$id_field}}:{{end}}{{end}}" 'mock-resources.test.com:'

  # For some reason without this explain fails with message no resource found.
  sleep 1

  output_message=$(kubectl explain mock-resource)
  kube::test::if_has_string "${output_message}" 'FIELDS:'

  output_message=$(kubectl explain mock-resource --recursive)
  kube::test::if_has_string "${output_message}" 'FIELDS:'

  # Cleanup
  kubectl delete -f hack/testdata/CRD/example-crd-with-additionalfields.yaml

  set +o nounset
  set +o errexit
}
