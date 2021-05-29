/*
Copyright 2015 The Kubernetes Authors.

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

package test

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
)

func TestDecodeUnstructured(t *testing.T) {
	groupVersionString := "v1"
	rawJson := fmt.Sprintf(`{"kind":"Pod","apiVersion":"%s","metadata":{"name":"test"}}`, groupVersionString)
	pl := &List{
		Items: []runtime.Object{
			&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			&runtime.Unknown{
				TypeMeta:    runtime.TypeMeta{Kind: "Pod", APIVersion: groupVersionString},
				Raw:         []byte(rawJson),
				ContentType: runtime.ContentTypeJSON,
			},
			&runtime.Unknown{
				TypeMeta:    runtime.TypeMeta{Kind: "", APIVersion: groupVersionString},
				Raw:         []byte(rawJson),
				ContentType: runtime.ContentTypeJSON,
			},
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"kind":       "Foo",
					"apiVersion": "Bar",
					"test":       "value",
				},
			},
		},
	}
	if errs := runtime.DecodeList(pl.Items, unstructured.UnstructuredJSONScheme); len(errs) == 1 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*unstructured.Unstructured); !ok || pod.Object["kind"] != "Pod" || pod.Object["metadata"].(map[string]interface{})["name"] != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
	if pod, ok := pl.Items[2].(*unstructured.Unstructured); !ok || pod.Object["kind"] != "Pod" || pod.Object["metadata"].(map[string]interface{})["name"] != "test" {
		t.Errorf("object not converted: %#v", pl.Items[2])
	}
}

func TestDecode(t *testing.T) {
	tcs := []struct {
		json []byte
		want runtime.Object
	}{
		{
			json: []byte(`{"apiVersion": "test", "kind": "test_kind"}`),
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_kind"},
			},
		},
		{
			json: []byte(`{"apiVersion": "test", "kind": "test_list", "items": []}`),
			want: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_list"},
				Items:  []unstructured.Unstructured{},
			},
		},
		{
			json: []byte(`{"items": [{"metadata": {"name": "object1", "deletionGracePeriodSeconds": 10}, "apiVersion": "test", "kind": "test_kind"}, {"metadata": {"name": "object2"}, "apiVersion": "test", "kind": "test_kind"}], "apiVersion": "test", "kind": "test_list"}`),
			want: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_list"},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"metadata":   map[string]interface{}{"name": "object1", "deletionGracePeriodSeconds": int64(10)},
							"apiVersion": "test",
							"kind":       "test_kind",
						},
					},
					{
						Object: map[string]interface{}{
							"metadata":   map[string]interface{}{"name": "object2"},
							"apiVersion": "test",
							"kind":       "test_kind",
						},
					},
				},
			},
		},
	}

	for _, tc := range tcs {
		got, _, err := unstructured.UnstructuredJSONScheme.Decode(tc.json, nil, nil)
		if err != nil {
			t.Errorf("Unexpected error for %q: %v", string(tc.json), err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("Decode(%q) want: %v\ngot: %v", string(tc.json), tc.want, got)
		}
	}
}

func TestUnstructuredGetters(t *testing.T) {
	trueVar := true
	ten := int64(10)
	unstruct := unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"name":                       "test_name",
				"namespace":                  "test_namespace",
				"generateName":               "test_generateName",
				"uid":                        "test_uid",
				"resourceVersion":            "test_resourceVersion",
				"generation":                 ten,
				"deletionGracePeriodSeconds": ten,
				"selfLink":                   "test_selfLink",
				"creationTimestamp":          "2009-11-10T23:00:00Z",
				"deletionTimestamp":          "2010-11-10T23:00:00Z",
				"labels": map[string]interface{}{
					"test_label": "test_value",
				},
				"annotations": map[string]interface{}{
					"test_annotation": "test_value",
				},
				"ownerReferences": []interface{}{
					map[string]interface{}{
						"kind":       "Pod",
						"name":       "poda",
						"apiVersion": "v1",
						"uid":        "1",
					},
					map[string]interface{}{
						"kind":       "Pod",
						"name":       "podb",
						"apiVersion": "v1",
						"uid":        "2",
						// though these fields are of type *bool, but when
						// decoded from JSON, they are unmarshalled as bool.
						"controller":         true,
						"blockOwnerDeletion": true,
					},
				},
				"finalizers": []interface{}{
					"finalizer.1",
					"finalizer.2",
				},
				"clusterName": "cluster123",
			},
		},
	}

	if got, want := unstruct.GetAPIVersion(), "test_version"; got != want {
		t.Errorf("GetAPIVersions() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetKind(), "test_kind"; got != want {
		t.Errorf("GetKind() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetNamespace(), "test_namespace"; got != want {
		t.Errorf("GetNamespace() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetName(), "test_name"; got != want {
		t.Errorf("GetName() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetGenerateName(), "test_generateName"; got != want {
		t.Errorf("GetGenerateName() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetUID(), types.UID("test_uid"); got != want {
		t.Errorf("GetUID() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetResourceVersion(), "test_resourceVersion"; got != want {
		t.Errorf("GetResourceVersion() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetSelfLink(), "test_selfLink"; got != want {
		t.Errorf("GetSelfLink() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetCreationTimestamp(), metav1.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC); !got.Equal(&want) {
		t.Errorf("GetCreationTimestamp() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetDeletionTimestamp(), metav1.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC); got == nil || !got.Equal(&want) {
		t.Errorf("GetDeletionTimestamp() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetLabels(), map[string]string{"test_label": "test_value"}; !reflect.DeepEqual(got, want) {
		t.Errorf("GetLabels() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetAnnotations(), map[string]string{"test_annotation": "test_value"}; !reflect.DeepEqual(got, want) {
		t.Errorf("GetAnnotations() = %s, want %s", got, want)
	}
	refs := unstruct.GetOwnerReferences()
	expectedOwnerReferences := []metav1.OwnerReference{
		{
			Kind:       "Pod",
			Name:       "poda",
			APIVersion: "v1",
			UID:        "1",
		},
		{
			Kind:               "Pod",
			Name:               "podb",
			APIVersion:         "v1",
			UID:                "2",
			Controller:         &trueVar,
			BlockOwnerDeletion: &trueVar,
		},
	}
	if got, want := refs, expectedOwnerReferences; !reflect.DeepEqual(got, want) {
		t.Errorf("GetOwnerReferences()=%v, want %v", got, want)
	}
	if got, want := unstruct.GetFinalizers(), []string{"finalizer.1", "finalizer.2"}; !reflect.DeepEqual(got, want) {
		t.Errorf("GetFinalizers()=%v, want %v", got, want)
	}
	if got, want := unstruct.GetClusterName(), "cluster123"; got != want {
		t.Errorf("GetClusterName()=%v, want %v", got, want)
	}
	if got, want := unstruct.GetDeletionGracePeriodSeconds(), &ten; !reflect.DeepEqual(got, want) {
		t.Errorf("GetDeletionGracePeriodSeconds()=%v, want %v", got, want)
	}
	if got, want := unstruct.GetGeneration(), ten; !reflect.DeepEqual(got, want) {
		t.Errorf("GetGeneration()=%v, want %v", got, want)
	}
}

func TestUnstructuredSetters(t *testing.T) {
	unstruct := unstructured.Unstructured{}
	trueVar := true
	ten := int64(10)

	want := unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"name":                       "test_name",
				"namespace":                  "test_namespace",
				"generateName":               "test_generateName",
				"uid":                        "test_uid",
				"resourceVersion":            "test_resourceVersion",
				"selfLink":                   "test_selfLink",
				"creationTimestamp":          "2009-11-10T23:00:00Z",
				"deletionTimestamp":          "2010-11-10T23:00:00Z",
				"deletionGracePeriodSeconds": ten,
				"generation":                 ten,
				"labels": map[string]interface{}{
					"test_label": "test_value",
				},
				"annotations": map[string]interface{}{
					"test_annotation": "test_value",
				},
				"ownerReferences": []interface{}{
					map[string]interface{}{
						"kind":       "Pod",
						"name":       "poda",
						"apiVersion": "v1",
						"uid":        "1",
					},
					map[string]interface{}{
						"kind":               "Pod",
						"name":               "podb",
						"apiVersion":         "v1",
						"uid":                "2",
						"controller":         true,
						"blockOwnerDeletion": true,
					},
				},
				"finalizers": []interface{}{
					"finalizer.1",
					"finalizer.2",
				},
				"clusterName": "cluster123",
			},
		},
	}

	unstruct.SetAPIVersion("test_version")
	unstruct.SetKind("test_kind")
	unstruct.SetNamespace("test_namespace")
	unstruct.SetName("test_name")
	unstruct.SetGenerateName("test_generateName")
	unstruct.SetUID(types.UID("test_uid"))
	unstruct.SetResourceVersion("test_resourceVersion")
	unstruct.SetSelfLink("test_selfLink")
	unstruct.SetCreationTimestamp(metav1.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC))
	date := metav1.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC)
	unstruct.SetDeletionTimestamp(&date)
	unstruct.SetLabels(map[string]string{"test_label": "test_value"})
	unstruct.SetAnnotations(map[string]string{"test_annotation": "test_value"})
	newOwnerReferences := []metav1.OwnerReference{
		{
			Kind:       "Pod",
			Name:       "poda",
			APIVersion: "v1",
			UID:        "1",
		},
		{
			Kind:               "Pod",
			Name:               "podb",
			APIVersion:         "v1",
			UID:                "2",
			Controller:         &trueVar,
			BlockOwnerDeletion: &trueVar,
		},
	}
	unstruct.SetOwnerReferences(newOwnerReferences)
	unstruct.SetFinalizers([]string{"finalizer.1", "finalizer.2"})
	unstruct.SetClusterName("cluster123")
	unstruct.SetDeletionGracePeriodSeconds(&ten)
	unstruct.SetGeneration(ten)

	if !reflect.DeepEqual(unstruct, want) {
		t.Errorf("Wanted: \n%s\n Got:\n%s", want, unstruct)
	}
}

func TestOwnerReferences(t *testing.T) {
	t.Parallel()
	trueVar := true
	falseVar := false
	refs := []metav1.OwnerReference{
		{
			APIVersion: "v2",
			Kind:       "K2",
			Name:       "n2",
			UID:        types.UID("abc1"),
		},
		{
			APIVersion:         "v1",
			Kind:               "K1",
			Name:               "n1",
			UID:                types.UID("abc2"),
			Controller:         &trueVar,
			BlockOwnerDeletion: &falseVar,
		},
		{
			APIVersion:         "v3",
			Kind:               "K3",
			Name:               "n3",
			UID:                types.UID("abc3"),
			Controller:         &falseVar,
			BlockOwnerDeletion: &trueVar,
		},
	}
	for i, ref := range refs {
		ref := ref
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			t.Parallel()
			u1 := unstructured.Unstructured{
				Object: make(map[string]interface{}),
			}
			refsX := []metav1.OwnerReference{ref}
			u1.SetOwnerReferences(refsX)

			have := u1.GetOwnerReferences()
			if !reflect.DeepEqual(have, refsX) {
				t.Errorf("Object references are not the same: %#v != %#v", have, refsX)
			}
		})
	}
}

func TestUnstructuredListGetters(t *testing.T) {
	unstruct := unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"resourceVersion": "test_resourceVersion",
				"selfLink":        "test_selfLink",
			},
		},
	}

	if got, want := unstruct.GetAPIVersion(), "test_version"; got != want {
		t.Errorf("GetAPIVersions() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetKind(), "test_kind"; got != want {
		t.Errorf("GetKind() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetResourceVersion(), "test_resourceVersion"; got != want {
		t.Errorf("GetResourceVersion() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetSelfLink(), "test_selfLink"; got != want {
		t.Errorf("GetSelfLink() = %s, want %s", got, want)
	}
}

func TestUnstructuredListSetters(t *testing.T) {
	unstruct := unstructured.UnstructuredList{}

	want := unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"resourceVersion": "test_resourceVersion",
				"selfLink":        "test_selfLink",
			},
		},
	}

	unstruct.SetAPIVersion("test_version")
	unstruct.SetKind("test_kind")
	unstruct.SetResourceVersion("test_resourceVersion")
	unstruct.SetSelfLink("test_selfLink")

	if !reflect.DeepEqual(unstruct, want) {
		t.Errorf("Wanted: \n%s\n Got:\n%s", unstruct, want)
	}
}

func TestDecodeNumbers(t *testing.T) {

	// Start with a valid pod
	originalJSON := []byte(`{
		"kind":"Carp",
		"apiVersion":"v1",
		"metadata":{"name":"pod","namespace":"foo"},
		"spec":{
			"containers":[{"name":"container","image":"container"}],
			"activeDeadlineSeconds":1000030003
		}
	}`)

	pod := &testapigroup.Carp{}

	_, codecs := TestScheme()
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Group: "", Version: runtime.APIVersionInternal})

	err := runtime.DecodeInto(codec, originalJSON, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Round-trip with unstructured codec
	unstructuredObj, err := runtime.Decode(unstructured.UnstructuredJSONScheme, originalJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	roundtripJSON, err := runtime.Encode(unstructured.UnstructuredJSONScheme, unstructuredObj)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Make sure we serialize back out in int form
	if !strings.Contains(string(roundtripJSON), `"activeDeadlineSeconds":1000030003`) {
		t.Errorf("Expected %s, got %s", `"activeDeadlineSeconds":1000030003`, string(roundtripJSON))
	}

	// Decode with structured codec again
	obj2, err := runtime.Decode(codec, roundtripJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// ensure pod is still valid
	pod2, ok := obj2.(*testapigroup.Carp)
	if !ok {
		t.Fatalf("expected an *api.Pod, got %#v", obj2)
	}

	// ensure round-trip preserved large integers
	if !reflect.DeepEqual(pod, pod2) {
		t.Fatalf("Expected\n\t%#v, got \n\t%#v", pod, pod2)
	}
}

// TestAccessorMethods does opaque roundtrip testing against an Unstructured
// instance's Object methods to ensure that what is "Set" matches what you
// subsequently "Get" without any assertions against internal state.
func TestAccessorMethods(t *testing.T) {
	int64p := func(i int) *int64 {
		v := int64(i)
		return &v
	}
	tests := []struct {
		accessor string
		val      interface{}
		nilVal   reflect.Value
	}{
		{accessor: "Namespace", val: "foo"},
		{accessor: "Name", val: "bar"},
		{accessor: "GenerateName", val: "baz"},
		{accessor: "UID", val: types.UID("uid")},
		{accessor: "ResourceVersion", val: "1"},
		{accessor: "Generation", val: int64(5)},
		{accessor: "SelfLink", val: "/foo"},
		// TODO: Handle timestamps, which are being marshalled as UTC and
		// unmarshalled as Local.
		// https://github.com/kubernetes/kubernetes/issues/21402
		// {accessor: "CreationTimestamp", val: someTime},
		// {accessor: "DeletionTimestamp", val: someTimeP},
		{accessor: "DeletionTimestamp", nilVal: reflect.ValueOf((*metav1.Time)(nil))},
		{accessor: "DeletionGracePeriodSeconds", val: int64p(10)},
		{accessor: "DeletionGracePeriodSeconds", val: int64p(0)},
		{accessor: "DeletionGracePeriodSeconds", nilVal: reflect.ValueOf((*int64)(nil))},
		{accessor: "Labels", val: map[string]string{"foo": "bar"}},
		{accessor: "Annotations", val: map[string]string{"foo": "bar"}},
		{accessor: "Finalizers", val: []string{"foo"}},
		{accessor: "OwnerReferences", val: []metav1.OwnerReference{{Name: "foo"}}},
		{accessor: "ClusterName", val: "foo"},
	}
	for i, test := range tests {
		t.Logf("evaluating test %d (%s)", i, test.accessor)

		u := &unstructured.Unstructured{}
		setter := reflect.ValueOf(u).MethodByName("Set" + test.accessor)
		getter := reflect.ValueOf(u).MethodByName("Get" + test.accessor)

		args := []reflect.Value{}
		if test.val != nil {
			args = append(args, reflect.ValueOf(test.val))
		} else {
			args = append(args, test.nilVal)
		}
		setter.Call(args)

		ret := getter.Call([]reflect.Value{})
		actual := ret[0].Interface()

		var expected interface{}
		if test.val != nil {
			expected = test.val
		} else {
			expected = test.nilVal.Interface()
		}

		if e, a := expected, actual; !reflect.DeepEqual(e, a) {
			t.Fatalf("%s: expected %v (%T), got %v (%T)", test.accessor, e, e, a, a)
		}
	}
}
