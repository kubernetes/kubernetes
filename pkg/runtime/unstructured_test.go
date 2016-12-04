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

package runtime_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta/metatypes"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
)

func TestDecodeUnstructured(t *testing.T) {
	groupVersionString := registered.GroupOrDie(api.GroupName).GroupVersion.String()
	rawJson := fmt.Sprintf(`{"kind":"Pod","apiVersion":"%s","metadata":{"name":"test"}}`, groupVersionString)
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
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
			&runtime.Unstructured{
				Object: map[string]interface{}{
					"kind":       "Foo",
					"apiVersion": "Bar",
					"test":       "value",
				},
			},
		},
	}
	if errs := runtime.DecodeList(pl.Items, runtime.UnstructuredJSONScheme); len(errs) == 1 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*runtime.Unstructured); !ok || pod.Object["kind"] != "Pod" || pod.Object["metadata"].(map[string]interface{})["name"] != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
	if pod, ok := pl.Items[2].(*runtime.Unstructured); !ok || pod.Object["kind"] != "Pod" || pod.Object["metadata"].(map[string]interface{})["name"] != "test" {
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
			want: &runtime.Unstructured{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_kind"},
			},
		},
		{
			json: []byte(`{"apiVersion": "test", "kind": "test_list", "items": []}`),
			want: &runtime.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_list"},
			},
		},
		{
			json: []byte(`{"items": [{"metadata": {"name": "object1"}, "apiVersion": "test", "kind": "test_kind"}, {"metadata": {"name": "object2"}, "apiVersion": "test", "kind": "test_kind"}], "apiVersion": "test", "kind": "test_list"}`),
			want: &runtime.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_list"},
				Items: []*runtime.Unstructured{
					{
						Object: map[string]interface{}{
							"metadata":   map[string]interface{}{"name": "object1"},
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
		got, _, err := runtime.UnstructuredJSONScheme.Decode(tc.json, nil, nil)
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
	unstruct := runtime.Unstructured{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"name":              "test_name",
				"namespace":         "test_namespace",
				"generateName":      "test_generateName",
				"uid":               "test_uid",
				"resourceVersion":   "test_resourceVersion",
				"selfLink":          "test_selfLink",
				"creationTimestamp": "2009-11-10T23:00:00Z",
				"deletionTimestamp": "2010-11-10T23:00:00Z",
				"labels": map[string]interface{}{
					"test_label": "test_value",
				},
				"annotations": map[string]interface{}{
					"test_annotation": "test_value",
				},
				"ownerReferences": []map[string]interface{}{
					{
						"kind":       "Pod",
						"name":       "poda",
						"apiVersion": "v1",
						"uid":        "1",
					},
					{
						"kind":       "Pod",
						"name":       "podb",
						"apiVersion": "v1",
						"uid":        "2",
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

	if got, want := unstruct.GetCreationTimestamp(), metav1.Date(2009, time.November, 10, 23, 0, 0, 0, time.UTC); !got.Equal(want) {
		t.Errorf("GetCreationTimestamp() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetDeletionTimestamp(), metav1.Date(2010, time.November, 10, 23, 0, 0, 0, time.UTC); got == nil || !got.Equal(want) {
		t.Errorf("GetDeletionTimestamp() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetLabels(), map[string]string{"test_label": "test_value"}; !reflect.DeepEqual(got, want) {
		t.Errorf("GetLabels() = %s, want %s", got, want)
	}

	if got, want := unstruct.GetAnnotations(), map[string]string{"test_annotation": "test_value"}; !reflect.DeepEqual(got, want) {
		t.Errorf("GetAnnotations() = %s, want %s", got, want)
	}
	refs := unstruct.GetOwnerReferences()
	expectedOwnerReferences := []metatypes.OwnerReference{
		{
			Kind:       "Pod",
			Name:       "poda",
			APIVersion: "v1",
			UID:        "1",
		},
		{
			Kind:       "Pod",
			Name:       "podb",
			APIVersion: "v1",
			UID:        "2",
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
}

func TestUnstructuredSetters(t *testing.T) {
	unstruct := runtime.Unstructured{}
	trueVar := true

	want := runtime.Unstructured{
		Object: map[string]interface{}{
			"kind":       "test_kind",
			"apiVersion": "test_version",
			"metadata": map[string]interface{}{
				"name":              "test_name",
				"namespace":         "test_namespace",
				"generateName":      "test_generateName",
				"uid":               "test_uid",
				"resourceVersion":   "test_resourceVersion",
				"selfLink":          "test_selfLink",
				"creationTimestamp": "2009-11-10T23:00:00Z",
				"deletionTimestamp": "2010-11-10T23:00:00Z",
				"labels": map[string]interface{}{
					"test_label": "test_value",
				},
				"annotations": map[string]interface{}{
					"test_annotation": "test_value",
				},
				"ownerReferences": []map[string]interface{}{
					{
						"kind":       "Pod",
						"name":       "poda",
						"apiVersion": "v1",
						"uid":        "1",
						"controller": (*bool)(nil),
					},
					{
						"kind":       "Pod",
						"name":       "podb",
						"apiVersion": "v1",
						"uid":        "2",
						"controller": &trueVar,
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
	newOwnerReferences := []metatypes.OwnerReference{
		{
			Kind:       "Pod",
			Name:       "poda",
			APIVersion: "v1",
			UID:        "1",
		},
		{
			Kind:       "Pod",
			Name:       "podb",
			APIVersion: "v1",
			UID:        "2",
			Controller: &trueVar,
		},
	}
	unstruct.SetOwnerReferences(newOwnerReferences)
	unstruct.SetFinalizers([]string{"finalizer.1", "finalizer.2"})
	unstruct.SetClusterName("cluster123")

	if !reflect.DeepEqual(unstruct, want) {
		t.Errorf("Wanted: \n%s\n Got:\n%s", want, unstruct)
	}
}

func TestUnstructuredListGetters(t *testing.T) {
	unstruct := runtime.UnstructuredList{
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
	unstruct := runtime.UnstructuredList{}

	want := runtime.UnstructuredList{
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
		"kind":"Pod",
		"apiVersion":"v1",
		"metadata":{"name":"pod","namespace":"foo"},
		"spec":{
			"containers":[{"name":"container","image":"container"}],
			"activeDeadlineSeconds":1000030003
		}
	}`)

	pod := &api.Pod{}

	// Decode with structured codec
	codec, err := testapi.GetCodecForObject(pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = runtime.DecodeInto(codec, originalJSON, pod)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// ensure pod is valid
	if errs := validation.ValidatePod(pod); len(errs) > 0 {
		t.Fatalf("pod should be valid: %v", errs)
	}

	// Round-trip with unstructured codec
	unstructuredObj, err := runtime.Decode(runtime.UnstructuredJSONScheme, originalJSON)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	roundtripJSON, err := runtime.Encode(runtime.UnstructuredJSONScheme, unstructuredObj)
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
	pod2, ok := obj2.(*api.Pod)
	if !ok {
		t.Fatalf("expected an *api.Pod, got %#v", obj2)
	}
	if errs := validation.ValidatePod(pod2); len(errs) > 0 {
		t.Fatalf("pod should be valid: %v", errs)
	}
	// ensure round-trip preserved large integers
	if !reflect.DeepEqual(pod, pod2) {
		t.Fatalf("Expected\n\t%#v, got \n\t%#v", pod, pod2)
	}
}
