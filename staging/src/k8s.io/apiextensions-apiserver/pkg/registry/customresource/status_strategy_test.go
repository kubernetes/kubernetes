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
func TestStatusStrategy_PrepareForUpdate_WithNilStructuralSchema(t *testing.T) {
	// This test covers the panic scenario from issue #133651
	// where CRDs created in v1beta1 (before 1.22) lack openAPIV3Schema fields
	// and cause a panic when accessing status subresource
	
	tests := []struct {
		name             string
		structuralSchema *structuralschema.Structural
		expectPanic      bool
		description      string
	}{
		{
			name:             "nil structural schema should not panic",
			structuralSchema: nil,
			expectPanic:      false,
			description:      "Legacy CRDs from v1beta1 may have nil structural schema",
		},
		{
			name: "valid structural schema should work normally",
			structuralSchema: &structuralschema.Structural{
				Generic: structuralschema.Generic{
					Type: "object",
				},
			},
			expectPanic: false,
			description: "Normal CRDs with proper schema should continue working",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a minimal status strategy with the test structural schema
			// We don't need to set up the full customResourceStrategy for this test
			crStrategy := customResourceStrategy{
				structuralSchema: tt.structuralSchema,
			}
			strategy := NewStatusStrategy(crStrategy)
			// Create test objects - a simple custom resource with status
			oldObj := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "TestResource",
					"metadata": map[string]interface{}{
						"name":      "test-resource",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"field": "old-value",
					},
					"status": map[string]interface{}{
						"phase": "Pending",
					},
				},
			}

			newObj := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "TestResource",
					"metadata": map[string]interface{}{
						"name":      "test-resource",
						"namespace": "default",
					},
					"spec": map[string]interface{}{
						"field": "new-value", // spec change should be ignored in status update
					},
					"status": map[string]interface{}{
						"phase": "Running", // status change should be preserved
					},
				},
			}

			// Test that PrepareForUpdate doesn't panic with nil structural schema
			defer func() {
				if r := recover(); r != nil {
					if tt.expectPanic {
						t.Logf("Expected panic occurred: %v", r)
					} else {
						t.Errorf("Unexpected panic with %s: %v", tt.description, r)
					}
				} else if tt.expectPanic {
					t.Errorf("Expected panic did not occur with %s", tt.description)
				}
			}()

			// This is the operation that was causing the panic
			strategy.PrepareForUpdate(context.TODO(), newObj, oldObj)

			// Verify that status updates work correctly even with nil schema
			if !tt.expectPanic {
				// The status field should be preserved from newObj
				newStatus, found, err := unstructured.NestedMap(newObj.Object, "status")
				if err != nil {
					t.Errorf("Error getting status from newObj: %v", err)
				}
				if !found {
					t.Error("Status field not found in newObj after PrepareForUpdate")
				}
				if phase, ok := newStatus["phase"].(string); !ok || phase != "Running" {
					t.Errorf("Expected status.phase to be 'Running', got %v", newStatus["phase"])
				}

				// The spec should be reset to old value (status updates shouldn't change spec)
				newSpec, found, err := unstructured.NestedMap(newObj.Object, "spec")
				if err != nil {
					t.Errorf("Error getting spec from newObj: %v", err)
				}
				if found {
					if field, ok := newSpec["field"].(string); ok && field != "old-value" {
						t.Errorf("Expected spec.field to be reset to 'old-value', got %v", field)
					}
				}
			}
		})
	}
}

func TestStatusStrategy_Validate_WithNilStructuralSchema(t *testing.T) {
	// Additional test to ensure validation also works with nil structural schema
	crStrategy := customResourceStrategy{
		structuralSchema: nil,
	}
	strategy := NewStatusStrategy(crStrategy)

	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "TestResource",
			"metadata": map[string]interface{}{
				"name":      "test-resource",
				"namespace": "default",
			},
			"status": map[string]interface{}{
				"phase": "Running",
			},
		},
	}

	// This should not panic even with nil structural schema
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Validate panicked with nil structural schema: %v", r)
		}
	}()

	errs := strategy.Validate(context.TODO(), obj)
	// Validation might return errors, but it should not panic
	if len(errs) > 0 {
		t.Logf("Validation returned errors (expected for nil schema): %v", errs)
	}
}
