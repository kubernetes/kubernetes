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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestDecodeUnstructured(t *testing.T) {
	version := testapi.Version()
	rawJson := fmt.Sprintf(`{"kind":"Pod","apiVersion":"%s","metadata":{"name":"test"}}`, version)
	pl := &api.List{
		Items: []runtime.Object{
			&api.Pod{ObjectMeta: api.ObjectMeta{Name: "1"}},
			&runtime.Unknown{TypeMeta: runtime.TypeMeta{Kind: "Pod", APIVersion: version}, RawJSON: []byte(rawJson)},
			&runtime.Unknown{TypeMeta: runtime.TypeMeta{Kind: "", APIVersion: version}, RawJSON: []byte(rawJson)},
			&runtime.Unstructured{TypeMeta: runtime.TypeMeta{Kind: "Foo", APIVersion: "Bar"}, Object: map[string]interface{}{"test": "value"}},
		},
	}
	if errs := runtime.DecodeList(pl.Items, runtime.UnstructuredJSONScheme); len(errs) == 1 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*runtime.Unstructured); !ok || pod.Object["kind"] != "Pod" || pod.Object["metadata"].(map[string]interface{})["name"] != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
	if _, ok := pl.Items[2].(*runtime.Unknown); !ok {
		t.Errorf("object should not have been converted: %#v", pl.Items[2])
	}
}
