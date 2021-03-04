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
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/admission"
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
	o := admission.NewObjectInterfacesFromScheme(scheme)
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
					Replicas: func() *int32 { var i int32 = 1; return &i }(),
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
			actual, err := ConvertToGVK(test.obj, test.gvk, o)
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

func TestConvertVersionedAttributes(t *testing.T) {
	scheme := initiateScheme(t)
	o := admission.NewObjectInterfacesFromScheme(scheme)

	gvk := func(g, v, k string) schema.GroupVersionKind {
		return schema.GroupVersionKind{g, v, k}
	}
	attrs := func(obj, oldObj runtime.Object) admission.Attributes {
		return admission.NewAttributesRecord(obj, oldObj, schema.GroupVersionKind{}, "", "", schema.GroupVersionResource{}, "", "", nil, false, nil)
	}
	u := func(data string) *unstructured.Unstructured {
		t.Helper()
		m := map[string]interface{}{}
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			t.Fatal(err)
		}
		return &unstructured.Unstructured{Object: m}
	}
	testcases := []struct {
		Name          string
		Attrs         *VersionedAttributes
		GVK           schema.GroupVersionKind
		ExpectedAttrs *VersionedAttributes
	}{
		{
			Name: "noop",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				VersionedObject:    &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpodversioned"}},
				VersionedKind:      examplev1.SchemeGroupVersion.WithKind("Pod"),
				Dirty:              true,
			},
			GVK: examplev1.SchemeGroupVersion.WithKind("Pod"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				VersionedObject:    &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpodversioned"}},
				VersionedKind:      examplev1.SchemeGroupVersion.WithKind("Pod"),
				Dirty:              true,
			},
		},
		{
			Name: "clean, typed",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				VersionedObject:    &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpodversioned"}},
				VersionedKind:      gvk("g", "v", "k"),
			},
			GVK: examplev1.SchemeGroupVersion.WithKind("Pod"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				// name gets overwritten from converted attributes, type gets set explicitly
				VersionedObject:    &examplev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "example.apiserver.k8s.io/v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
				VersionedOldObject: &examplev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "example.apiserver.k8s.io/v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				VersionedKind:      examplev1.SchemeGroupVersion.WithKind("Pod"),
			},
		},
		{
			Name: "clean, unstructured",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobj"}}`),
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				),
				VersionedObject:    u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobjversioned"}}`),
				VersionedOldObject: u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobjversioned"}}`),
				VersionedKind:      gvk("g", "v", "k"), // claim a different current version to trigger conversion
			},
			GVK: gvk("mygroup.k8s.io", "v1", "Flunder"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobj"}}`),
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				),
				VersionedObject:    u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobj"}}`),
				VersionedOldObject: u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				VersionedKind:      gvk("mygroup.k8s.io", "v1", "Flunder"),
			},
		},
		{
			Name: "dirty, typed",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				VersionedObject:    &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpodversioned"}},
				VersionedKind:      gvk("g", "v", "k"), // claim a different current version to trigger conversion
				Dirty:              true,
			},
			GVK: examplev1.SchemeGroupVersion.WithKind("Pod"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				),
				// new name gets preserved from versioned object, type gets set explicitly
				VersionedObject: &examplev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "example.apiserver.k8s.io/v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				// old name gets overwritten from converted attributes, type gets set explicitly
				VersionedOldObject: &examplev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "example.apiserver.k8s.io/v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "oldpod"}},
				VersionedKind:      examplev1.SchemeGroupVersion.WithKind("Pod"),
				Dirty:              false,
			},
		},
		{
			Name: "dirty, unstructured",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobj"}}`),
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				),
				VersionedObject:    u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobjversioned"}}`),
				VersionedOldObject: u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobjversioned"}}`),
				VersionedKind:      gvk("g", "v", "k"), // claim a different current version to trigger conversion
				Dirty:              true,
			},
			GVK: gvk("mygroup.k8s.io", "v1", "Flunder"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobjversioned"}}`),
					u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				),
				// new name gets preserved from versioned object, type gets set explicitly
				VersionedObject: u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"newobjversioned"}}`),
				// old name gets overwritten from converted attributes, type gets set explicitly
				VersionedOldObject: u(`{"apiVersion": "mygroup.k8s.io/v1","kind": "Flunder","metadata":{"name":"oldobj"}}`),
				VersionedKind:      gvk("mygroup.k8s.io", "v1", "Flunder"),
				Dirty:              false,
			},
		},
		{
			Name: "nil old object",
			Attrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpod"}},
					nil,
				),
				VersionedObject:    &examplev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: nil,
				VersionedKind:      gvk("g", "v", "k"), // claim a different current version to trigger conversion
				Dirty:              true,
			},
			GVK: examplev1.SchemeGroupVersion.WithKind("Pod"),
			ExpectedAttrs: &VersionedAttributes{
				Attributes: attrs(
					&example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
					nil,
				),
				// new name gets preserved from versioned object, type gets set explicitly
				VersionedObject:    &examplev1.Pod{TypeMeta: metav1.TypeMeta{APIVersion: "example.apiserver.k8s.io/v1", Kind: "Pod"}, ObjectMeta: metav1.ObjectMeta{Name: "newpodversioned"}},
				VersionedOldObject: nil,
				VersionedKind:      examplev1.SchemeGroupVersion.WithKind("Pod"),
				Dirty:              false,
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			err := ConvertVersionedAttributes(tc.Attrs, tc.GVK, o)
			if err != nil {
				t.Fatal(err)
			}
			if e, a := tc.ExpectedAttrs.Attributes.GetObject(), tc.Attrs.Attributes.GetObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
			if e, a := tc.ExpectedAttrs.Attributes.GetOldObject(), tc.Attrs.Attributes.GetOldObject(); !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
			if e, a := tc.ExpectedAttrs.VersionedKind, tc.Attrs.VersionedKind; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
			if e, a := tc.ExpectedAttrs.VersionedObject, tc.Attrs.VersionedObject; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
			if e, a := tc.ExpectedAttrs.VersionedOldObject, tc.Attrs.VersionedOldObject; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
			if e, a := tc.ExpectedAttrs.Dirty, tc.Attrs.Dirty; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff:\n%s", diff.ObjectReflectDiff(e, a))
			}
		})
	}
}
