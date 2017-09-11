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

package kubectl

import (
	"reflect"
	"strings"
	"testing"

	api "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	internal "k8s.io/kubernetes/pkg/api"
)

func encodeOrDie(obj runtime.Object) []byte {
	data, err := runtime.Encode(internal.Codecs.LegacyCodec(api.SchemeGroupVersion), obj)
	if err != nil {
		panic(err.Error())
	}
	return data
}

func TestSortingPrinter(t *testing.T) {
	intPtr := func(val int32) *int32 { return &val }

	a := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "a",
		},
	}

	b := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "b",
		},
	}

	c := &api.Pod{
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
			name: "in-order-already",
			obj: &api.PodList{
				Items: []api.Pod{
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
			sort: &api.PodList{
				Items: []api.Pod{
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
			obj: &api.PodList{
				Items: []api.Pod{
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
			sort: &api.PodList{
				Items: []api.Pod{
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
			obj: &api.PodList{
				Items: []api.Pod{
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
			sort: &api.PodList{
				Items: []api.Pod{
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
			obj: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(5),
						},
					},
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(1),
						},
					},
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(9),
						},
					},
				},
			},
			sort: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(1),
						},
					},
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(5),
						},
					},
					{
						Spec: api.ReplicationControllerSpec{
							Replicas: intPtr(9),
						},
					},
				},
			},
			field: "{.spec.replicas}",
		},
		{
			name: "v1.List in order",
			obj: &api.List{
				Items: []runtime.RawExtension{
					{Raw: encodeOrDie(a)},
					{Raw: encodeOrDie(b)},
					{Raw: encodeOrDie(c)},
				},
			},
			sort: &api.List{
				Items: []runtime.RawExtension{
					{Raw: encodeOrDie(a)},
					{Raw: encodeOrDie(b)},
					{Raw: encodeOrDie(c)},
				},
			},
			field: "{.metadata.name}",
		},
		{
			name: "v1.List in reverse",
			obj: &api.List{
				Items: []runtime.RawExtension{
					{Raw: encodeOrDie(c)},
					{Raw: encodeOrDie(b)},
					{Raw: encodeOrDie(a)},
				},
			},
			sort: &api.List{
				Items: []runtime.RawExtension{
					{Raw: encodeOrDie(a)},
					{Raw: encodeOrDie(b)},
					{Raw: encodeOrDie(c)},
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
			obj: &api.ReplicationControllerList{
				Items: []api.ReplicationController{
					{
						Status: api.ReplicationControllerStatus{},
					},
					{
						Status: api.ReplicationControllerStatus{},
					},
					{
						Status: api.ReplicationControllerStatus{},
					},
				},
			},
			field:       "{.invalid}",
			expectedErr: "couldn't find any field with path \"{.invalid}\" in the list of objects",
		},
	}
	for _, test := range tests {
		sort := &SortingPrinter{SortField: test.field, Decoder: internal.Codecs.UniversalDecoder()}
		err := sort.sortObj(test.obj)
		if err != nil {
			if len(test.expectedErr) > 0 {
				if strings.Contains(err.Error(), test.expectedErr) {
					continue
				}
				t.Fatalf("%s: expected error containing: %q, got: \"%v\"", test.name, test.expectedErr, err)
			}
			t.Fatalf("%s: unexpected error: %v", test.name, err)
		}
		if len(test.expectedErr) > 0 {
			t.Fatalf("%s: expected error containing: %q, got none", test.name, test.expectedErr)
		}
		if !reflect.DeepEqual(test.obj, test.sort) {
			t.Errorf("[%s]\nexpected:\n%v\nsaw:\n%v", test.name, test.sort, test.obj)
		}
	}
}
