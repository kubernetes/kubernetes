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

package runtime_test

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestDecodeList(t *testing.T) {
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
			&runtime.Unknown{
				TypeMeta:    runtime.TypeMeta{Kind: "Pod", APIVersion: testapi.Default.GroupVersion().String()},
				Raw:         []byte(`{"kind":"Pod","apiVersion":"` + testapi.Default.GroupVersion().String() + `","metadata":{"name":"test"}}`),
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
	if errs := runtime.DecodeList(pl.Items, testapi.Default.Codec()); len(errs) != 0 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*api.Pod); !ok || pod.Name != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
}
