/*
Copyright 2017 The Kubernetes Authors.

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

package versioned

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
)

func initiateScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	example.AddToScheme(s)
	examplev1.AddToScheme(s)
	example2v1.AddToScheme(s)
	return s
}

func TestConvertToGVK(t *testing.T) {
	scheme := initiateScheme()
	c := Convertor{Scheme: scheme}
	table := map[string]struct {
		obj         runtime.Object
		gvk         schema.GroupVersionKind
		expectedObj runtime.Object
	}{
		"convert example#Pod to example/v1#Pod": {
			obj: &example.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example.PodSpec{
					RestartPolicy: example.RestartPolicy("never"),
				},
			},
			gvk: examplev1.SchemeGroupVersion.WithKind("Pod"),
			expectedObj: &examplev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: examplev1.PodSpec{
					RestartPolicy: examplev1.RestartPolicy("never"),
				},
			},
		},
		"convert example#replicaset to example2/v1#replicaset": {
			obj: &example.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name: "rs1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example.ReplicaSetSpec{
					Replicas: 1,
				},
			},
			gvk: example2v1.SchemeGroupVersion.WithKind("ReplicaSet"),
			expectedObj: &example2v1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name: "rs1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example2v1.ReplicaSetSpec{
					Replicas: func() *int32 { var i int32; i = 1; return &i }(),
				},
			},
		},
		"no conversion for Unstructured object whose gvk matches the desired gvk": {
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "mygroup.k8s.io/v1",
					"kind":       "Flunder",
					"data": map[string]interface{}{
						"Key": "Value",
					},
				},
			},
			gvk: schema.GroupVersionKind{Group: "mygroup.k8s.io", Version: "v1", Kind: "Flunder"},
			expectedObj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "mygroup.k8s.io/v1",
					"kind":       "Flunder",
					"data": map[string]interface{}{
						"Key": "Value",
					},
				},
			},
		},
	}

	for name, test := range table {
		t.Run(name, func(t *testing.T) {
			actual, err := c.ConvertToGVK(test.obj, test.gvk)
			if err != nil {
				t.Error(err)
			}
			if !reflect.DeepEqual(actual, test.expectedObj) {
				t.Errorf("\nexpected:\n%#v\ngot:\n %#v\n", test.expectedObj, actual)
			}
		})
	}
}

func TestConvert(t *testing.T) {
	scheme := initiateScheme()
	c := Convertor{Scheme: scheme}
	sampleCRD := unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.k8s.io/v1",
			"kind":       "Flunder",
			"data": map[string]interface{}{
				"Key": "Value",
			},
		},
	}

	table := map[string]struct {
		in          runtime.Object
		out         runtime.Object
		expectedObj runtime.Object
	}{
		"convert example/v1#Pod to example#Pod": {
			in: &examplev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: examplev1.PodSpec{
					RestartPolicy: examplev1.RestartPolicy("never"),
				},
			},
			out: &example.Pod{},
			expectedObj: &example.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example.PodSpec{
					RestartPolicy: example.RestartPolicy("never"),
				},
			},
		},
		"convert example2/v1#replicaset to example#replicaset": {
			in: &example2v1.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name: "rs1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example2v1.ReplicaSetSpec{
					Replicas: func() *int32 { var i int32; i = 1; return &i }(),
				},
			},
			out: &example.ReplicaSet{},
			expectedObj: &example.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name: "rs1",
					Labels: map[string]string{
						"key": "value",
					},
				},
				Spec: example.ReplicaSetSpec{
					Replicas: 1,
				},
			},
		},
		"no conversion if the object is the same": {
			in:          &sampleCRD,
			out:         &sampleCRD,
			expectedObj: &sampleCRD,
		},
	}
	for name, test := range table {
		t.Run(name, func(t *testing.T) {
			err := c.Convert(test.in, test.out)
			if err != nil {
				t.Error(err)
			}
			if !reflect.DeepEqual(test.out, test.expectedObj) {
				t.Errorf("\nexpected:\n%#v\ngot:\n %#v\n", test.expectedObj, test.out)
			}
		})
	}
}
