/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package transformers

import (
	"fmt"
	"sigs.k8s.io/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"testing"
)

type noopMutator struct {
	wasCalled     bool
	errorToReturn error
}

var errExpected = fmt.Errorf("oops")

const originalValue = "tomato"
const newValue = "notThe" + originalValue

func (m *noopMutator) mutate(in interface{}) (interface{}, error) {
	m.wasCalled = true
	return newValue, m.errorToReturn
}

func makeTestDeployment() ifc.Kunstructured {
	factory := kunstruct.NewKunstructuredFactoryImpl()
	return factory.FromMap(
		map[string]interface{}{
			"group":      "apps",
			"apiVersion": "v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": originalValue,
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"env": []interface{}{
						map[string]interface{}{
							"name":  "HELLO",
							"value": "hi there",
						},
						map[string]interface{}{
							"name":  "GOODBYE",
							"value": "adios!",
						},
					},
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"vegetable": originalValue,
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "tangerine",
								"image": originalValue,
							},
						},
					},
				},
			},
		})
}

func getFieldValue(t *testing.T, obj ifc.Kunstructured, fieldName string) string {
	v, err := obj.GetString(fieldName)
	if err != nil {
		t.Fatalf("unexpected field error: %v", err)
	}
	return v
}

func TestNoPath(t *testing.T) {
	obj := makeTestDeployment()
	m := &noopMutator{}
	err := MutateField(
		obj.Map(), []string{}, false, m.mutate)
	if m.wasCalled {
		t.Fatalf("mutator should not have been called.")
	}
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestHappyPath(t *testing.T) {
	obj := makeTestDeployment()
	v := getFieldValue(t, obj, "metadata.name")
	if v != originalValue {
		t.Fatalf("unexpected original value: %v", v)
	}
	v = getFieldValue(t, obj, "spec.template.metadata.labels.vegetable")
	if v != originalValue {
		t.Fatalf("unexpected original value: %v", v)
	}

	m := &noopMutator{}
	err := MutateField(
		obj.Map(), []string{"metadata", "name"}, false, m.mutate)
	if !m.wasCalled {
		t.Fatalf("mutator should have been called.")
	}
	if err != nil {
		t.Fatalf("unexpected mutate error: %v", err)
	}
	v = getFieldValue(t, obj, "metadata.name")
	if v != newValue {
		t.Fatalf("unexpected new value: %v", v)
	}

	m = &noopMutator{}
	err = MutateField(
		obj.Map(), []string{"spec", "template", "metadata", "labels", "vegetable"}, false, m.mutate)
	if !m.wasCalled {
		t.Fatalf("mutator should have been called.")
	}
	if err != nil {
		t.Fatalf("unexpected mutate error: %v", err)
	}
	v = getFieldValue(t, obj, "spec.template.metadata.labels.vegetable")
	if v != newValue {
		t.Fatalf("unexpected new value: %v", v)
	}
}

func TestWithError(t *testing.T) {
	obj := makeTestDeployment()
	m := noopMutator{errorToReturn: errExpected}
	err := MutateField(
		obj.Map(), []string{"metadata", "name"}, false, m.mutate)
	if !m.wasCalled {
		t.Fatalf("mutator was not called!")
	}
	if err != errExpected {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestWithNil(t *testing.T) {
	obj := makeTestDeployment()
	foo := obj.Map()["spec"]
	foo = foo.(map[string]interface{})["template"]
	foo = foo.(map[string]interface{})["metadata"]
	foo.(map[string]interface{})["labels"] = nil

	m := &noopMutator{}
	err := MutateField(
		obj.Map(), []string{"spec", "template", "metadata", "labels", "vegetable"}, false, m.mutate)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
}
