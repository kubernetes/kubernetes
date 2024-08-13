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

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/utils/ptr"
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
func createPodSpecResource(t *testing.T, memReq, memLimit, cpuReq, cpuLimit string) corev1.PodSpec {
	t.Helper()
	podSpec := corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{},
					Limits:   corev1.ResourceList{},
				},
			},
		},
	}

	req := podSpec.Containers[0].Resources.Requests
	if memReq != "" {
		memReq, err := resource.ParseQuantity(memReq)
		if err != nil {
			t.Errorf("memory request string is not a valid quantity")
		}
		req["memory"] = memReq
	}
	if cpuReq != "" {
		cpuReq, err := resource.ParseQuantity(cpuReq)
		if err != nil {
			t.Errorf("cpu request string is not a valid quantity")
		}
		req["cpu"] = cpuReq
	}
	limit := podSpec.Containers[0].Resources.Limits
	if memLimit != "" {
		memLimit, err := resource.ParseQuantity(memLimit)
		if err != nil {
			t.Errorf("memory limit string is not a valid quantity")
		}
		limit["memory"] = memLimit
	}
	if cpuLimit != "" {
		cpuLimit, err := resource.ParseQuantity(cpuLimit)
		if err != nil {
			t.Errorf("cpu limit string is not a valid quantity")
		}
		limit["cpu"] = cpuLimit
	}

	return podSpec
}
func createUnstructuredPodResource(t *testing.T, memReq, memLimit, cpuReq, cpuLimit string) unstructured.Unstructured {
	t.Helper()
	pod := &corev1.Pod{
		Spec: createPodSpecResource(t, memReq, memLimit, cpuReq, cpuLimit),
	}
	return *toUnstructuredOrDie(encodeOrDie(pod))
}

func TestSortingPrinter(t *testing.T) {
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
							Replicas: ptr.To[int32](5),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: ptr.To[int32](1),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: ptr.To[int32](9),
						},
					},
				},
			},
			sort: &corev1.ReplicationControllerList{
				Items: []corev1.ReplicationController{
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: ptr.To[int32](1),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: ptr.To[int32](5),
						},
					},
					{
						Spec: corev1.ReplicationControllerSpec{
							Replicas: ptr.To[int32](9),
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
		{
			name: "pod-resources-cpu-random-order-with-missing-fields",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						Spec: createPodSpecResource(t, "", "", "0.5", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "10", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "100m", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "", ""),
					},
				},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{
					{
						Spec: createPodSpecResource(t, "", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "100m", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "0.5", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "10", ""),
					},
				},
			},
			field: "{.spec.containers[].resources.requests.cpu}",
		},
		{
			name: "pod-resources-memory-random-order-with-missing-fields",
			obj: &corev1.PodList{
				Items: []corev1.Pod{
					{
						Spec: createPodSpecResource(t, "128Mi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "10Ei", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "8Ti", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "64Gi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "55Pi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "2Ki", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "", "", "", ""),
					},
				},
			},
			sort: &corev1.PodList{
				Items: []corev1.Pod{
					{
						Spec: createPodSpecResource(t, "", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "2Ki", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "128Mi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "64Gi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "8Ti", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "55Pi", "", "", ""),
					},
					{
						Spec: createPodSpecResource(t, "10Ei", "", "", ""),
					},
				},
			},
			field: "{.spec.containers[].resources.requests.memory}",
		},
		{
			name: "pod-unstructured-resources-cpu-random-order-with-missing-fields",
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					createUnstructuredPodResource(t, "", "", "0.5", ""),
					createUnstructuredPodResource(t, "", "", "10", ""),
					createUnstructuredPodResource(t, "", "", "100m", ""),
					createUnstructuredPodResource(t, "", "", "", ""),
				},
			},
			sort: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					createUnstructuredPodResource(t, "", "", "", ""),
					createUnstructuredPodResource(t, "", "", "100m", ""),
					createUnstructuredPodResource(t, "", "", "0.5", ""),
					createUnstructuredPodResource(t, "", "", "10", ""),
				},
			},
			field: "{.spec.containers[].resources.requests.cpu}",
		},
		{
			name: "pod-unstructured-resources-memory-random-order-with-missing-fields",
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					createUnstructuredPodResource(t, "128Mi", "", "", ""),
					createUnstructuredPodResource(t, "10Ei", "", "", ""),
					createUnstructuredPodResource(t, "8Ti", "", "", ""),
					createUnstructuredPodResource(t, "64Gi", "", "", ""),
					createUnstructuredPodResource(t, "55Pi", "", "", ""),
					createUnstructuredPodResource(t, "2Ki", "", "", ""),
					createUnstructuredPodResource(t, "", "", "", ""),
				},
			},
			sort: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
				},
				Items: []unstructured.Unstructured{
					createUnstructuredPodResource(t, "", "", "", ""),
					createUnstructuredPodResource(t, "2Ki", "", "", ""),
					createUnstructuredPodResource(t, "128Mi", "", "", ""),
					createUnstructuredPodResource(t, "64Gi", "", "", ""),
					createUnstructuredPodResource(t, "8Ti", "", "", ""),
					createUnstructuredPodResource(t, "55Pi", "", "", ""),
					createUnstructuredPodResource(t, "10Ei", "", "", ""),
				},
			},
			field: "{.spec.containers[].resources.requests.memory}",
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
				t.Errorf("[%s]\nexpected/saw:\n%s", tt.name, cmp.Diff(expectedTable, table))
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
				Spec: createPodSpecResource(t, "0.5", "", "1Gi", ""),
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "c",
				},
				Spec: createPodSpecResource(t, "2", "", "1Ti", ""),
			},
			{
				ObjectMeta: metav1.ObjectMeta{
					Name: "a",
				},
				Spec: createPodSpecResource(t, "10m", "", "1Ki", ""),
			},
		},
	}

	testobjs, err := meta.ExtractList(testobj)
	if err != nil {
		t.Fatalf("ExtractList testobj got unexpected error: %v", err)
	}

	testfieldName := "{.metadata.name}"
	testruntimeSortName := NewRuntimeSort(testfieldName, testobjs)

	testfieldCPU := "{.spec.containers[].resources.requests.cpu}"
	testruntimeSortCPU := NewRuntimeSort(testfieldCPU, testobjs)

	testfieldMemory := "{.spec.containers[].resources.requests.memory}"
	testruntimeSortMemory := NewRuntimeSort(testfieldMemory, testobjs)

	tests := []struct {
		name         string
		runtimeSort  *RuntimeSort
		i            int
		j            int
		expectResult bool
		expectErr    bool
	}{
		{
			name:         "test name b c less true",
			runtimeSort:  testruntimeSortName,
			i:            0,
			j:            1,
			expectResult: true,
		},
		{
			name:         "test name c a less false",
			runtimeSort:  testruntimeSortName,
			i:            1,
			j:            2,
			expectResult: false,
		},
		{
			name:         "test name b a less false",
			runtimeSort:  testruntimeSortName,
			i:            0,
			j:            2,
			expectResult: false,
		},
		{
			name:         "test cpu 0.5 2 less true",
			runtimeSort:  testruntimeSortCPU,
			i:            0,
			j:            1,
			expectResult: true,
		},
		{
			name:         "test cpu 2 10mi less false",
			runtimeSort:  testruntimeSortCPU,
			i:            1,
			j:            2,
			expectResult: false,
		},
		{
			name:         "test cpu 0.5 10mi less false",
			runtimeSort:  testruntimeSortCPU,
			i:            0,
			j:            2,
			expectResult: false,
		},
		{
			name:         "test memory 1Gi 1Ti less true",
			runtimeSort:  testruntimeSortMemory,
			i:            0,
			j:            1,
			expectResult: true,
		},
		{
			name:         "test memory 1Ti 1Ki less false",
			runtimeSort:  testruntimeSortMemory,
			i:            1,
			j:            2,
			expectResult: false,
		},
		{
			name:         "test memory 1Gi 1Ki less false",
			runtimeSort:  testruntimeSortMemory,
			i:            0,
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
