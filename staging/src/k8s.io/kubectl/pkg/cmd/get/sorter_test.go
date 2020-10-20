/*
Copyright 2014 The Kubernetes Authors.

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

package get

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubectl/pkg/scheme"
)

func toUnstructuredOrDie(data []byte) *unstructured.Unstructured {
	unstrBody := map[string]interface{}{}
	err := json.Unmarshal(data, &unstrBody)
	if err != nil {
		panic(err)
	}
	return &unstructured.Unstructured{Object: unstrBody}
}
func encodeOrDie(obj runtime.Object) []byte {
	data, err := runtime.Encode(scheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion), obj)
	if err != nil {
		panic(err.Error())
	}
	return data
}

func TestSortingPrinter(t *testing.T) {
	intPtr := func(val int32) *int32 { return &val }

	a := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "a",
		},
	}

	b := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "b",
		},
	}

	c := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "c",
		},
	}

	tests := []struct {
		obj         runtime.Object
		sort        runtime.Object
		field       string
		name        string
		expectedErr string
	}{
		{
			name: "empty",
			obj: &corev1.PodList{
				Items: []corev1.Pod{},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{},
			},
			field: "{.metadata.name}",
		},
		{
			name: "in-order-already",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "a",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "b",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "c",
						},
					},
				},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "a",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "b",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "c",
						},
					},
				},
			},
			field: "{.metadata.name}",
		},
		{
			name: "reverse-order",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "b",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "c",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "a",
						},
					},
				},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "a",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "b",
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							Name: "c",
						},
					},
				},
			},
			field: "{.metadata.name}",
		},
		{
			name: "random-order-timestamp",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(300, 0),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(100, 0),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(200, 0),
						},
					},
				},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(100, 0),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(200, 0),
						},
					},
					{
						ObjectMeta: metav1.ObjectMeta{
							CreationTimestamp: metav1.Unix(300, 0),
						},
					},
				},
			},
			field: "{.metadata.creationTimestamp}",
		},
		{
			name: "random-order-numbers",
			obj: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(5),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(1),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(9),
						},
					},
				},
			},
			sort: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(1),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(5),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: intPtr(9),
						},
					},
				},
			},
			field: "{.spec.replicas}",
		},
		{
			name: "v1.List in order",
			obj: &corev1.List{
				Items: []runtime.RawExtension{
					{Object: a, Raw: encodeOrDie(a)},
					{Object: b, Raw: encodeOrDie(b)},
					{Object: c, Raw: encodeOrDie(c)},
				},
			},
			sort: &corev1.List{
				Items: []runtime.RawExtension{
					{Object: a, Raw: encodeOrDie(a)},
					{Object: b, Raw: encodeOrDie(b)},
					{Object: c, Raw: encodeOrDie(c)},
				},
			},
			field: "{.metadata.name}",
		},
		{
			name: "v1.List in reverse",
			obj: &corev1.List{
				Items: []runtime.RawExtension{
					{Object: c, Raw: encodeOrDie(c)},
					{Object: b, Raw: encodeOrDie(b)},
					{Object: a, Raw: encodeOrDie(a)},
				},
			},
			sort: &corev1.List{
				Items: []runtime.RawExtension{
					{Object: a, Raw: encodeOrDie(a)},
					{Object: b, Raw: encodeOrDie(b)},
					{Object: c, Raw: encodeOrDie(c)},
				},
			},
			field: "{.metadata.name}",
		},
		{
			name: "some-missing-fields",
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"availableReplicas": 2,
							},
						},
					},
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status":     map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"availableReplicas": 1,
							},
						},
					},
				},
			},
			sort: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status":     map[string]interface{}{},
						},
					},
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"availableReplicas": 1,
							},
						},
					},
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"availableReplicas": 2,
							},
						},
					},
				},
			},
			field: "{.status.availableReplicas}",
		},
		{
			name: "all-missing-fields",
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"replicas": 0,
							},
						},
					},
					{
						Object: map[string]interface{}{
							"kind":       "ReplicationController",
							"apiVersion": "v1",
							"status": map[string]interface{}{
								"replicas": 0,
							},
						},
					},
				},
			},
			field:       "{.status.availableReplicas}",
			expectedErr: "couldn't find any field with path \"{.status.availableReplicas}\" in the list of objects",
		},
		{
			name: "model-invalid-fields",
			obj: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{
					{
						Status: corev1.ReplicationControllerStatus{},
					},
					{
						Status: corev1.ReplicationControllerStatus{},
					},
					{
						Status: corev1.ReplicationControllerStatus{},
					},
				},
			},
			field:       "{.invalid}",
			expectedErr: "couldn't find any field with path \"{.invalid}\" in the list of objects",
		},
		{
			name: "empty fields",
			obj: &corev1.EventList{
				Items: []corev1.Event{
					{
						ObjectMeta:    metav1.ObjectMeta{CreationTimestamp: metav1.Unix(300, 0)},
						LastTimestamp: metav1.Unix(300, 0),
					},
					{
						ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.Unix(200, 0)},
					},
				},
			},
			sort: &corev1.EventList{
				Items: []corev1.Event{
					{
						ObjectMeta: metav1.ObjectMeta{CreationTimestamp: metav1.Unix(200, 0)},
					},
					{
						ObjectMeta:    metav1.ObjectMeta{CreationTimestamp: metav1.Unix(300, 0)},
						LastTimestamp: metav1.Unix(300, 0),
					},
				},
			},
			field: "{.lastTimestamp}",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name+" table", func(t *testing.T) {
			table := &metav1beta1.Table{}
			meta.EachListItem(tt.obj, func(item runtime.Object) error {
				table.Rows = append(table.Rows, metav1beta1.TableRow{
					Object: runtime.RawExtension{Object: toUnstructuredOrDie(encodeOrDie(item))},
				})
				return nil
			})

			expectedTable := &metav1beta1.Table{}
			meta.EachListItem(tt.sort, func(item runtime.Object) error {
				expectedTable.Rows = append(expectedTable.Rows, metav1beta1.TableRow{
					Object: runtime.RawExtension{Object: toUnstructuredOrDie(encodeOrDie(item))},
				})
				return nil
			})

			sorter, err := NewTableSorter(table, tt.field)
			if err == nil {
				err = sorter.Sort()
			}
			if err != nil {
				if len(tt.expectedErr) > 0 {
					if strings.Contains(err.Error(), tt.expectedErr) {
						return
					}
					t.Fatalf("%s: expected error containing: %q, got: \"%v\"", tt.name, tt.expectedErr, err)
				}
				t.Fatalf("%s: unexpected error: %v", tt.name, err)
			}
			if len(tt.expectedErr) > 0 {
				t.Fatalf("%s: expected error containing: %q, got none", tt.name, tt.expectedErr)
			}
			if !reflect.DeepEqual(table, expectedTable) {
				t.Errorf("[%s]\nexpected/saw:\n%s", tt.name, diff.ObjectReflectDiff(expectedTable, table))
			}
		})
		t.Run(tt.name, func(t *testing.T) {
			sort := &SortingPrinter{SortField: tt.field, Decoder: scheme.Codecs.UniversalDecoder()}
			err := sort.sortObj(tt.obj)
			if err != nil {
				if len(tt.expectedErr) > 0 {
					if strings.Contains(err.Error(), tt.expectedErr) {
						return
					}
					t.Fatalf("%s: expected error containing: %q, got: \"%v\"", tt.name, tt.expectedErr, err)
				}
				t.Fatalf("%s: unexpected error: %v", tt.name, err)
			}
			if len(tt.expectedErr) > 0 {
				t.Fatalf("%s: expected error containing: %q, got none", tt.name, tt.expectedErr)
			}
			if !reflect.DeepEqual(tt.obj, tt.sort) {
				t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v", tt.name, tt.sort, tt.obj)
			}
		})
	}
}

func TestRuntimeSortLess(t *testing.T) {
	var testobj runtime.Object

	testobj = &corev1.PodList{
		Items: []corev1.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "b",
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "c",
				},
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "a",
				},
			},
		},
	}

	testobjs, err := meta.ExtractList(testobj)
	if err != nil {
		t.Fatalf("ExtractList testobj got unexpected error: %v", err)
	}

	testfield := "{.metadata.name}"
	testruntimeSort := NewRuntimeSort(testfield, testobjs)
	tests := []struct {
		name         string
		runtimeSort  *RuntimeSort
		i            int
		j            int
		expectResult bool
		expectErr    bool
	}{
		{
			name:         "test less true",
			runtimeSort:  testruntimeSort,
			i:            0,
			j:            1,
			expectResult: true,
		},
		{
			name:         "test less false",
			runtimeSort:  testruntimeSort,
			i:            1,
			j:            2,
			expectResult: false,
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.runtimeSort.Less(test.i, test.j)
			if result != test.expectResult {
				t.Errorf("case[%d]:%s Expected result: %v, Got result: %v", i, test.name, test.expectResult, result)
			}
		})
	}
}
