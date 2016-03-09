/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestDecodeUnstructured(t *testing.T) {
	groupVersionString := testapi.Default.GroupVersion().String()
	rawJson := fmt.Sprintf(`{"kind":"Pod","apiVersion":"%s","metadata":{"name":"test"}}`, groupVersionString)
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
			&runtime.Unknown{TypeMeta: runtime.TypeMeta{Kind: "Pod", APIVersion: groupVersionString}, RawJSON: []byte(rawJson)},
			&runtime.Unknown{TypeMeta: runtime.TypeMeta{Kind: "", APIVersion: groupVersionString}, RawJSON: []byte(rawJson)},
			&runtime.Unstructured{TypeMeta: runtime.TypeMeta{Kind: "Foo", APIVersion: "Bar"}, Object: map[string]interface{}{"test": "value"}},
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
				TypeMeta: runtime.TypeMeta{
					APIVersion: "test",
					Kind:       "test_kind",
				},
				Object: map[string]interface{}{"apiVersion": "test", "kind": "test_kind"},
			},
		},
		{
			json: []byte(`{"apiVersion": "test", "kind": "test_list", "items": []}`),
			want: &runtime.UnstructuredList{
				TypeMeta: runtime.TypeMeta{
					APIVersion: "test",
					Kind:       "test_list",
				},
			},
		},
		{
			json: []byte(`{"items": [{"metadata": {"name": "object1"}, "apiVersion": "test", "kind": "test_kind"}, {"metadata": {"name": "object2"}, "apiVersion": "test", "kind": "test_kind"}], "apiVersion": "test", "kind": "test_list"}`),
			want: &runtime.UnstructuredList{
				TypeMeta: runtime.TypeMeta{
					APIVersion: "test",
					Kind:       "test_list",
				},
				Items: []*runtime.Unstructured{
					{
						TypeMeta: runtime.TypeMeta{
							APIVersion: "test",
							Kind:       "test_kind",
						},
						Name: "object1",
						Object: map[string]interface{}{
							"metadata":   map[string]interface{}{"name": "object1"},
							"apiVersion": "test",
							"kind":       "test_kind",
						},
					},
					{
						TypeMeta: runtime.TypeMeta{
							APIVersion: "test",
							Kind:       "test_kind",
						},
						Name: "object2",
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
