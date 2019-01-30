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

run_crd_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl crd"
  kubectl "${KUBE_FLAGS_WITH_TOKEN[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "foos.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "scope": "Namespaced",
    "names": {
      "plural": "foos",
      "kind": "Foo"
    }
  }
}
__EOF__

  # Post-Condition: assertion object exist
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{if eq $ID_FIELD \\\"foos.company.com\\\"}}{{$ID_FIELD}}:{{end}}{{end}}" 'foos.company.com:'

  kubectl "${KUBE_FLAGS_WITH_TOKEN[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "bars.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "scope": "Namespaced",
    "names": {
      "plural": "bars",
      "kind": "Bar"
    }
  }
}
__EOF__

  # Post-Condition: assertion object exist
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{if eq $ID_FIELD \\\"foos.company.com\\\" \\\"bars.company.com\\\"}}{{$ID_FIELD}}:{{end}}{{end}}" 'bars.company.com:foos.company.com:'

  # This test ensures that the name printer is able to output a resource
  # in the proper "kind.group/resource_name" format, and that the
  # resource builder is able to resolve a GVK when a kind.group pair is given.
  kubectl "${KUBE_FLAGS_WITH_TOKEN[@]}" create -f - << __EOF__
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

  # Post-Condition: assertion crd with non-matching kind and resource exists
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{if eq $ID_FIELD \\\"foos.company.com\\\" \\\"bars.company.com\\\" \\\"resources.mygroup.example.com\\\"}}{{$ID_FIELD}}:{{end}}{{end}}" 'bars.company.com:foos.company.com:resources.mygroup.example.com:'

  # This test ensures that we can create complex validation without client-side validation complaining
  kubectl "${KUBE_FLAGS_WITH_TOKEN[@]}" create -f - << __EOF__
{
  "kind": "CustomResourceDefinition",
  "apiVersion": "apiextensions.k8s.io/v1beta1",
  "metadata": {
    "name": "validfoos.company.com"
  },
  "spec": {
    "group": "company.com",
    "version": "v1",
    "scope": "Namespaced",
    "names": {
      "plural": "validfoos",
      "kind": "ValidFoo"
    },
    "validation": {
      "openAPIV3Schema": {
        "properties": {
          "spec": {
            "type": "array",
            "items": {
              "type": "number"
            }
          }
        }
      }
    }
  }
}
__EOF__

  # Post-Condition: assertion crd with non-matching kind and resource exists
  kube::test::get_object_assert customresourcedefinitions "{{range.items}}{{if eq $ID_FIELD \\\"foos.company.com\\\" \\\"bars.company.com\\\" \\\"resources.mygroup.example.com\\\" \\\"validfoos.company.com\\\"}}{{$ID_FIELD}}:{{end}}{{end}}" 'bars.company.com:foos.company.com:resources.mygroup.example.com:validfoos.company.com:'

  run_non_native_resource_tests

  # teardown
  kubectl delete customresourcedefinitions/foos.company.com "${KUBE_FLAGS_WITH_TOKEN[@]}"
  kubectl delete customresourcedefinitions/bars.company.com "${KUBE_FLAGS_WITH_TOKEN[@]}"
  kubectl delete customresourcedefinitions/resources.mygroup.example.com "${KUBE_FLAGS_WITH_TOKEN[@]}"
  kubectl delete customresourcedefinitions/validfoos.company.com "${KUBE_FLAGS_WITH_TOKEN[@]}"

  set +o nounset
  set +o errexit
}

kube::util::non_native_resources() {
  local times
  local wait
  local failed
  times=30
  wait=10
  local i
  for i in $(seq 1 $times); do
    failed=""
    kubectl "${KUBE_FLAGS[@]}" get --raw '/apis/company.com/v1' || failed=true
    kubectl "${KUBE_FLAGS[@]}" get --raw '/apis/company.com/v1/foos' || failed=true
    kubectl "${KUBE_FLAGS[@]}" get --raw '/apis/company.com/v1/bars' || failed=true

    if [ -z "${failed}" ]; then
      return 0
    fi
    sleep ${wait}
  done

  kube::log::error "Timed out waiting for non-native-resources; tried ${times} waiting ${wait}s between each"
  return 1
}

run_non_native_resource_tests() {
  set -o nounset
  set -o errexit

  create_and_use_new_namespace
  kube::log::status "Testing kubectl non-native resources"
  kube::util::non_native_resources

  # Test that we can list this new CustomResource (foos)
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can list this new CustomResource (bars)
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can list this new CustomResource (resources)
  kube::test::get_object_assert resources "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can create a new resource of type Kind
  kubectl "${KUBE_FLAGS[@]}" create -f hack/testdata/CRD/resource.yaml "${KUBE_FLAGS[@]}"

  # Test that -o name returns kind.group/resourcename
  output_message=$(kubectl "${KUBE_FLAGS[@]}" get resource/myobj -o name)
  kube::test::if_has_string "${output_message}" 'kind.mygroup.example.com/myobj'

  output_message=$(kubectl "${KUBE_FLAGS[@]}" get resources/myobj -o name)
  kube::test::if_has_string "${output_message}" 'kind.mygroup.example.com/myobj'

  output_message=$(kubectl "${KUBE_FLAGS[@]}" get kind.mygroup.example.com/myobj -o name)
  kube::test::if_has_string "${output_message}" 'kind.mygroup.example.com/myobj'

  # Delete the resource with cascade.
  kubectl "${KUBE_FLAGS[@]}" delete resources myobj --cascade=true

  # Make sure it's gone
  kube::test::wait_object_assert resources "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can create a new resource of type Foo
  kubectl "${KUBE_FLAGS[@]}" create -f hack/testdata/CRD/foo.yaml "${KUBE_FLAGS[@]}"

  # Test that we can list this new custom resource
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'

  # Test alternate forms
  kube::test::get_object_assert foo                 "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'
  kube::test::get_object_assert foos.company.com    "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'
  kube::test::get_object_assert foos.v1.company.com "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'

  # Test all printers, with lists and individual items
  kube::log::status "Testing CustomResource printing"
  kubectl "${KUBE_FLAGS[@]}" get foos
  kubectl "${KUBE_FLAGS[@]}" get foos/test
  kubectl "${KUBE_FLAGS[@]}" get foos      -o name
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o name
  kubectl "${KUBE_FLAGS[@]}" get foos      -o wide
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o wide
  kubectl "${KUBE_FLAGS[@]}" get foos      -o json
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o json
  kubectl "${KUBE_FLAGS[@]}" get foos      -o yaml
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o yaml
  kubectl "${KUBE_FLAGS[@]}" get foos      -o "jsonpath={.items[*].someField}" --allow-missing-template-keys=false
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o "jsonpath={.someField}"          --allow-missing-template-keys=false
  kubectl "${KUBE_FLAGS[@]}" get foos      -o "go-template={{range .items}}{{.someField}}{{end}}" --allow-missing-template-keys=false
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o "go-template={{.someField}}"                        --allow-missing-template-keys=false
  output_message=$(kubectl "${KUBE_FLAGS[@]}" get foos/test -o name)
  kube::test::if_has_string "${output_message}" 'foo.company.com/test'

  # Test patching
  kube::log::status "Testing CustomResource patching"
  kubectl "${KUBE_FLAGS[@]}" patch foos/test -p '{"patched":"value1"}' --type=merge
  kube::test::get_object_assert foos/test "{{.patched}}" 'value1'
  kubectl "${KUBE_FLAGS[@]}" patch foos/test -p '{"patched":"value2"}' --type=merge --record
  kube::test::get_object_assert foos/test "{{.patched}}" 'value2'
  kubectl "${KUBE_FLAGS[@]}" patch foos/test -p '{"patched":null}' --type=merge --record
  kube::test::get_object_assert foos/test "{{.patched}}" '<no value>'
  # Get local version
  CRD_RESOURCE_FILE="${KUBE_TEMP}/crd-foos-test.json"
  kubectl "${KUBE_FLAGS[@]}" get foos/test -o json > "${CRD_RESOURCE_FILE}"
  # cannot apply strategic patch locally
  CRD_PATCH_ERROR_FILE="${KUBE_TEMP}/crd-foos-test-error"
  ! kubectl "${KUBE_FLAGS[@]}" patch --local -f "${CRD_RESOURCE_FILE}" -p '{"patched":"value3"}' 2> "${CRD_PATCH_ERROR_FILE}"
  if grep -q "try --type merge" "${CRD_PATCH_ERROR_FILE}"; then
    kube::log::status "\"kubectl patch --local\" returns error as expected for CustomResource: $(cat ${CRD_PATCH_ERROR_FILE})"
  else
    kube::log::status "\"kubectl patch --local\" returns unexpected error or non-error: $(cat ${CRD_PATCH_ERROR_FILE})"
    exit 1
  fi
  # can apply merge patch locally
  kubectl "${KUBE_FLAGS[@]}" patch --local -f "${CRD_RESOURCE_FILE}" -p '{"patched":"value3"}' --type=merge -o json
  # can apply merge patch remotely
  kubectl "${KUBE_FLAGS[@]}" patch --record -f "${CRD_RESOURCE_FILE}" -p '{"patched":"value3"}' --type=merge -o json
  kube::test::get_object_assert foos/test "{{.patched}}" 'value3'
  rm "${CRD_RESOURCE_FILE}"
  rm "${CRD_PATCH_ERROR_FILE}"

  # Test labeling
  kube::log::status "Testing CustomResource labeling"
  kubectl "${KUBE_FLAGS[@]}" label foos --all listlabel=true
  kubectl "${KUBE_FLAGS[@]}" label foo/test itemlabel=true

  # Test annotating
  kube::log::status "Testing CustomResource annotating"
  kubectl "${KUBE_FLAGS[@]}" annotate foos --all listannotation=true
  kubectl "${KUBE_FLAGS[@]}" annotate foo/test itemannotation=true

  # Test describing
  kube::log::status "Testing CustomResource describing"
  kubectl "${KUBE_FLAGS[@]}" describe foos
  kubectl "${KUBE_FLAGS[@]}" describe foos/test
  kubectl "${KUBE_FLAGS[@]}" describe foos | grep listlabel=true
  kubectl "${KUBE_FLAGS[@]}" describe foos | grep itemlabel=true

  # Delete the resource with cascade.
  kubectl "${KUBE_FLAGS[@]}" delete foos test --cascade=true

  # Make sure it's gone
  kube::test::wait_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can create a new resource of type Bar
  kubectl "${KUBE_FLAGS[@]}" create -f hack/testdata/CRD/bar.yaml "${KUBE_FLAGS[@]}"

  # Test that we can list this new custom resource
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'

  # Test that we can watch the resource.
  # Start watcher in background with process substitution,
  # so we can read from stdout asynchronously.
  kube::log::status "Testing CustomResource watching"
  exec 3< <(kubectl "${KUBE_FLAGS[@]}" get bars --request-timeout=1m --watch-only -o name & echo $! ; wait)
  local watch_pid
  read <&3 watch_pid

  # We can't be sure when the watch gets established,
  # so keep triggering events (in the background) until something comes through.
  local tries=0
  while [ ${tries} -lt 10 ]; do
    tries=$((tries+1))
    kubectl "${KUBE_FLAGS[@]}" patch bars/test -p "{\"patched\":\"${tries}\"}" --type=merge
    sleep 1
  done &
  local patch_pid=$!

  # Wait up to 30s for a complete line of output.
  local watch_output
  read <&3 -t 30 watch_output
  # Stop the watcher and the patch loop.
  kill -9 ${watch_pid}
  kill -9 ${patch_pid}
  kube::test::if_has_string "${watch_output}" 'bar.company.com/test'

  # Delete the resource without cascade.
  kubectl "${KUBE_FLAGS[@]}" delete bars test --cascade=false

  # Make sure it's gone
  kube::test::wait_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can create single item via apply
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/foo.yaml

  # Test that we have create a foo named test
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test '{{.someField}}' 'field1'

  # Test that apply an empty patch doesn't change fields
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/foo.yaml

  # Test that the field has the same value after re-apply
  kube::test::get_object_assert foos/test '{{.someField}}' 'field1'

  # Test that apply has updated the subfield
  kube::test::get_object_assert foos/test '{{.nestedField.someSubfield}}' 'subfield1'

  # Update a subfield and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/foo-updated-subfield.yaml

  # Test that apply has updated the subfield
  kube::test::get_object_assert foos/test '{{.nestedField.someSubfield}}' 'modifiedSubfield'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test '{{.nestedField.otherSubfield}}' 'subfield2'

  # Delete a subfield and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/foo-deleted-subfield.yaml

  # Test that apply has deleted the field
  kube::test::get_object_assert foos/test '{{.nestedField.otherSubfield}}' '<no value>'

  # Test that the field does not exist
  kube::test::get_object_assert foos/test '{{.nestedField.newSubfield}}' '<no value>'

  # Add a field and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/foo-added-subfield.yaml

  # Test that apply has added the field
  kube::test::get_object_assert foos/test '{{.nestedField.newSubfield}}' 'subfield3'

  # Delete the resource
  kubectl "${KUBE_FLAGS[@]}" delete -f hack/testdata/CRD/foo.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test that we can create list via apply
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/multi-crd-list.yaml

  # Test that we have create a foo and a bar from a list
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test-list:'
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test-list:'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Test that re-apply an list doesn't change anything
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/multi-crd-list.yaml

  # Test that the field has the same value after re-apply
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Test that the fields have the expected value
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'field1'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'field1'

  # Update fields and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/multi-crd-list-updated-field.yaml

  # Test that apply has updated the fields
  kube::test::get_object_assert foos/test-list '{{.someField}}' 'modifiedField'
  kube::test::get_object_assert bars/test-list '{{.someField}}' 'modifiedField'

  # Test that the field has the expected value
  kube::test::get_object_assert foos/test-list '{{.otherField}}' 'field2'
  kube::test::get_object_assert bars/test-list '{{.otherField}}' 'field2'

  # Delete fields and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/multi-crd-list-deleted-field.yaml

  # Test that apply has deleted the fields
  kube::test::get_object_assert foos/test-list '{{.otherField}}' '<no value>'
  kube::test::get_object_assert bars/test-list '{{.otherField}}' '<no value>'

  # Test that the fields does not exist
  kube::test::get_object_assert foos/test-list '{{.newField}}' '<no value>'
  kube::test::get_object_assert bars/test-list '{{.newField}}' '<no value>'

  # Add a field and then apply the change
  kubectl "${KUBE_FLAGS[@]}" apply -f hack/testdata/CRD/multi-crd-list-added-field.yaml

  # Test that apply has added the field
  kube::test::get_object_assert foos/test-list '{{.newField}}' 'field3'
  kube::test::get_object_assert bars/test-list '{{.newField}}' 'field3'

  # Delete the resource
  kubectl "${KUBE_FLAGS[@]}" delete -f hack/testdata/CRD/multi-crd-list.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  ## kubectl apply --prune
  # Test that no foo or bar exist
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # apply --prune on foo.yaml that has foo/test
  kubectl apply --prune -l pruneGroup=true -f hack/testdata/CRD/foo.yaml "${KUBE_FLAGS[@]}" --prune-whitelist=company.com/v1/Foo --prune-whitelist=company.com/v1/Bar
  # check right crds exist
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # apply --prune on bar.yaml that has bar/test
  kubectl apply --prune -l pruneGroup=true -f hack/testdata/CRD/bar.yaml "${KUBE_FLAGS[@]}" --prune-whitelist=company.com/v1/Foo --prune-whitelist=company.com/v1/Bar
  # check right crds exist
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" 'test:'

  # Delete the resource
  kubectl "${KUBE_FLAGS[@]}" delete -f hack/testdata/CRD/bar.yaml

  # Make sure it's gone
  kube::test::get_object_assert foos "{{range.items}}{{$ID_FIELD}}:{{end}}" ''
  kube::test::get_object_assert bars "{{range.items}}{{$ID_FIELD}}:{{end}}" ''

  # Test 'kubectl create' with namespace, and namespace cleanup.
  kubectl "${KUBE_FLAGS[@]}" create namespace non-native-resources
  kubectl "${KUBE_FLAGS[@]}" create -f hack/testdata/CRD/bar.yaml --namespace=non-native-resources
  kube::test::get_object_assert bars '{{len .items}}' '1' --namespace=non-native-resources
  kubectl "${KUBE_FLAGS[@]}" delete namespace non-native-resources
  # Make sure objects go away.
  kube::test::wait_object_assert bars '{{len .items}}' '0' --namespace=non-native-resources
  # Make sure namespace goes away.
  local tries=0
  while kubectl "${KUBE_FLAGS[@]}" get namespace non-native-resources && [ ${tries} -lt 10 ]; do
    tries=$((tries+1))
    sleep ${tries}
  done

  set +o nounset
  set +o errexit
}
