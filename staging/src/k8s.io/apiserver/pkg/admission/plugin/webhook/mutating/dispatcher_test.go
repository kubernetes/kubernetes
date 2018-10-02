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

package mutating

import (
	"context"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
)

var sampleCRD = unstructured.Unstructured{
	Object: map[string]interface{}{
		"apiVersion": "mygroup.k8s.io/v1",
		"kind":       "Flunder",
		"data": map[string]interface{}{
			"Key": "Value",
		},
	},
}

func TestDispatch(t *testing.T) {
	scheme := runtime.NewScheme()
	require.NoError(t, example.AddToScheme(scheme))
	require.NoError(t, examplev1.AddToScheme(scheme))
	require.NoError(t, example2v1.AddToScheme(scheme))

	tests := []struct {
		name        string
		in          runtime.Object
		out         runtime.Object
		expectedObj runtime.Object
	}{
		{
			name: "convert example/v1#Pod to example#Pod",
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
		{
			name: "convert example2/v1#replicaset to example#replicaset",
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
		{
			name:        "no conversion if the object is the same",
			in:          &sampleCRD,
			out:         &sampleCRD,
			expectedObj: &sampleCRD,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := &mutatingDispatcher{
				plugin: &Plugin{
					scheme: scheme,
				},
			}
			attr := generic.VersionedAttributes{
				Attributes:         admission.NewAttributesRecord(test.out, nil, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", admission.Operation(""), false, nil),
				VersionedOldObject: nil,
				VersionedObject:    test.in,
			}
			if err := a.Dispatch(context.TODO(), &attr, nil); err != nil {
				t.Fatalf("%s: unexpected error: %v", test.name, err)
			}
			if !reflect.DeepEqual(attr.Attributes.GetObject(), test.expectedObj) {
				t.Errorf("\nexpected:\n%#v\ngot:\n %#v\n", test.expectedObj, test.out)
			}
		})
	}
}
