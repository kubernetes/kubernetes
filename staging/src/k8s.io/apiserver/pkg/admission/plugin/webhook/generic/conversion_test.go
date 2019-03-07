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

package generic

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
)

func initiateScheme(t *testing.T) *runtime.Scheme {
	s := runtime.NewScheme()
	require.NoError(t, example.AddToScheme(s))
	require.NoError(t, examplev1.AddToScheme(s))
	require.NoError(t, example2v1.AddToScheme(s))
	return s
}

func TestConvertToGVK(t *testing.T) {
	scheme := initiateScheme(t)
	c := convertor{Scheme: scheme}
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
				TypeMeta: metav1.TypeMeta{
					APIVersion: "example.apiserver.k8s.io/v1",
					Kind:       "Pod",
				},
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
				TypeMeta: metav1.TypeMeta{
					APIVersion: "example2.apiserver.k8s.io/v1",
					Kind:       "ReplicaSet",
				},
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

// TestRuntimeSchemeConvert verifies that scheme.Convert(x, x, nil) for an unstructured x is a no-op.
// This did not use to be like that and we had to wrap scheme.Convert before.
func TestRuntimeSchemeConvert(t *testing.T) {
	scheme := initiateScheme(t)
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"foo": "bar",
		},
	}
	clone := obj.DeepCopy()

	if err := scheme.Convert(obj, obj, nil); err != nil {
		t.Fatalf("unexpected convert error: %v", err)
	}
	if !reflect.DeepEqual(obj, clone) {
		t.Errorf("unexpected mutation of self-converted Unstructured: obj=%#v, clone=%#v", obj, clone)
	}
}
