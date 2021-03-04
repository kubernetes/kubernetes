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

package editor

import (
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
)

func TestHashOnLineBreak(t *testing.T) {
	tests := []struct {
		original string
		expected string
	}{
		{
			original: "",
			expected: "",
		},
		{
			original: "\n",
			expected: "\n",
		},
		{
			original: "a\na\na\n",
			expected: "a\n# a\n# a\n",
		},
		{
			original: "a\n\n\na\n\n",
			expected: "a\n# \n# \n# a\n# \n",
		},
	}
	for _, test := range tests {
		r := hashOnLineBreak(test.original)
		if r != test.expected {
			t.Errorf("expected: %s, saw: %s", test.expected, r)
		}
	}
}

func TestManagedFieldsExtractAndRestore(t *testing.T) {
	tests := map[string]struct {
		object        runtime.Object
		managedFields map[types.UID][]metav1.ManagedFieldsEntry
	}{
		"single object, empty managedFields": {
			object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("12345")}},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): nil,
			},
		},
		"multiple objects, empty managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): nil,
				types.UID("98765"): nil,
			},
		},
		"single object, all managedFields": {
			object: &corev1.Pod{ObjectMeta: metav1.ObjectMeta{
				UID: types.UID("12345"),
				ManagedFields: []metav1.ManagedFieldsEntry{
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
			}},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
			},
		},
		"multiple objects, all managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Apply",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Update",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
				types.UID("98765"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationUpdate,
					},
				},
			},
		},
		"multiple objects, some managedFields": {
			object: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{},
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "12345",
								"managedFields": []interface{}{
									map[string]interface{}{
										"manager":   "test",
										"operation": "Apply",
									},
								},
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Pod",
							"metadata": map[string]interface{}{
								"uid": "98765",
							},
							"spec":   map[string]interface{}{},
							"status": map[string]interface{}{},
						},
					},
				},
			},
			managedFields: map[types.UID][]metav1.ManagedFieldsEntry{
				types.UID("12345"): {
					{
						Manager:   "test",
						Operation: metav1.ManagedFieldsOperationApply,
					},
				},
				types.UID("98765"): nil,
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			// operate on a copy, so we can compare the original and the modified object
			objCopy := test.object.DeepCopyObject()
			var infos []*resource.Info
			o := NewEditOptions(NormalEditMode, genericclioptions.NewTestIOStreamsDiscard())
			err := o.extractManagedFields(objCopy)
			if err != nil {
				t.Errorf("unexpected extraction error %v", err)
			}
			if meta.IsListType(objCopy) {
				infos = []*resource.Info{}
				meta.EachListItem(objCopy, func(obj runtime.Object) error {
					metaObjs, _ := meta.Accessor(obj)
					if metaObjs.GetManagedFields() != nil {
						t.Errorf("unexpected managedFileds after extraction")
					}
					infos = append(infos, &resource.Info{Object: obj})
					return nil
				})
			} else {
				metaObjs, _ := meta.Accessor(objCopy)
				if metaObjs.GetManagedFields() != nil {
					t.Errorf("unexpected managedFileds after extraction")
				}
				infos = []*resource.Info{{Object: objCopy}}
			}

			err = o.restoreManagedFields(infos)
			if err != nil {
				t.Errorf("unexpected restore error %v", err)
			}
			if !reflect.DeepEqual(test.object, objCopy) {
				t.Errorf("mismatched object after extract and restore managedFields: %#v", objCopy)
			}
			if test.managedFields != nil && !reflect.DeepEqual(test.managedFields, o.managedFields) {
				t.Errorf("mismatched managedFields %#v vs %#v", test.managedFields, o.managedFields)
			}
		})
	}
}
