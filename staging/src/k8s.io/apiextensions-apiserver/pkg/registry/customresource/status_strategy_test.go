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

package customresource

import (
	"context"
	"reflect"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/yaml"
)

func TestPrepareForUpdate(t *testing.T) {
	strategy := statusStrategy{}
	tcs := []struct {
		old      *unstructured.Unstructured
		obj      *unstructured.Unstructured
		expected *unstructured.Unstructured
	}{
		{
			// changes to spec are ignored
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// changes to other places are also ignored
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"new": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// delete status
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec": "old",
				},
			},
		},
		{
			// update status
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"status": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "new",
				},
			},
		},
		{
			// update status and other parts
			old: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "old",
				},
			},
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "new",
					"new":    "new",
					"status": "new",
				},
			},
			expected: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"spec":   "old",
					"status": "new",
				},
			},
		},
	}
	for index, tc := range tcs {
		strategy.PrepareForUpdate(context.TODO(), tc.obj, tc.old)
		if !reflect.DeepEqual(tc.obj, tc.expected) {
			t.Errorf("test %d failed: expected: %v, got %v", index, tc.expected, tc.obj)
		}
	}
}

const listTypeResourceSchema = `
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: foos.test
spec:
  group: test
  names:
    kind: Foo
    listKind: FooList
    plural: foos
    singular: foo
  scope: Cluster
  versions:
  - name: v1
    schema:
      openAPIV3Schema:
        type: object
        properties:
          numArray:
            type: array
            x-kubernetes-list-type: set
            items:
              type: object
    served: true
    storage: true
  - name: v2
    schema:
      openAPIV3Schema:
        type: object
        properties:
          numArray2:
            type: array
`

func TestStatusStrategyValidateUpdate(t *testing.T) {
	crdV1 := &apiextensionsv1.CustomResourceDefinition{}
	err := yaml.Unmarshal([]byte(listTypeResourceSchema), &crdV1)
	if err != nil {
		t.Fatalf("unexpected decoding error: %v", err)
	}
	t.Logf("crd v1 details: %v", crdV1)
	crd := &apiextensions.CustomResourceDefinition{}
	if err = apiextensionsv1.Convert_v1_CustomResourceDefinition_To_apiextensions_CustomResourceDefinition(crdV1, crd, nil); err != nil {
		t.Fatalf("unexpected convert error: %v", err)
	}
	t.Logf("crd details: %v", crd)

	strategy := statusStrategy{}
	kind := schema.GroupVersionKind{
		Version: crd.Spec.Versions[0].Name,
		Kind:    crd.Spec.Names.Kind,
		Group:   crd.Spec.Group,
	}
	strategy.customResourceStrategy.validator.kind = kind
	ss, _ := structuralschema.NewStructural(crd.Spec.Versions[0].Schema.OpenAPIV3Schema)
	strategy.structuralSchema = ss

	ctx := context.TODO()

	tcs := []struct {
		name    string
		old     *unstructured.Unstructured
		obj     *unstructured.Unstructured
		isValid bool
	}{
		{
			name:    "bothValid",
			old:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "numArray": []interface{}{1, 2}}},
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "numArray": []interface{}{1, 3}, "metadata": map[string]interface{}{"resourceVersion": "1"}}},
			isValid: true,
		},
		{
			name:    "change to invalid",
			old:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "old", "numArray": []interface{}{1, 2}}},
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "new", "numArray": []interface{}{1, 1}, "metadata": map[string]interface{}{"resourceVersion": "1"}}},
			isValid: false,
		},
		{
			name:    "change to valid",
			old:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "new", "numArray": []interface{}{1, 1}}},
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "old", "numArray": []interface{}{1, 2}, "metadata": map[string]interface{}{"resourceVersion": "1"}}},
			isValid: true,
		},
		{
			name:    "keeps invalid",
			old:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "new", "numArray": []interface{}{1, 1}}},
			obj:     &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "test/v1", "kind": "Foo", "spec": "old", "numArray": []interface{}{1, 1}, "metadata": map[string]interface{}{"resourceVersion": "1"}}},
			isValid: true,
		},
	}

	for _, tc := range tcs {
		errs := strategy.ValidateUpdate(ctx, tc.obj, tc.old)
		if tc.isValid && len(errs) > 0 {
			t.Errorf("%v: unexpected error: %v", tc.name, errs)
		}
		if !tc.isValid && len(errs) == 0 {
			t.Errorf("%v: unexpected non-error", tc.name)
		}
	}
}
