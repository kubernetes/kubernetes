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

package json

import (
	"k8s.io/kubernetes/pkg/runtime/schema"
	"reflect"
	"testing"
)

func TestSimpleMetaFactoryInterpret(t *testing.T) {
	factory := SimpleMetaFactory{}
	testCases := []struct {
		data        []byte
		expectedGVK *schema.GroupVersionKind
		errFn       func(err error)
	}{
		{
			data:        []byte(`{"apiVersion":"1","kind":"object"}`),
			expectedGVK: &schema.GroupVersionKind{Kind: "object", Group: "", Version: "1"},
			errFn: func(err error) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			},
		},
		{
			data:        []byte(`{"apiVersion":"group/1","kind":"object"}`),
			expectedGVK: &schema.GroupVersionKind{Kind: "object", Group: "group", Version: "1"},
			errFn: func(err error) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			},
		},
		{
			data:        []byte(`{}`),
			expectedGVK: &schema.GroupVersionKind{Kind: "", Group: "", Version: ""},
			errFn: func(err error) {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			},
		},
		{
			data:        []byte(`{`),
			expectedGVK: nil,
			errFn: func(err error) {
				if err == nil {
					t.Fatalf("unexpected non-error")
				}
			},
		},
	}
	for i, test := range testCases {
		gvk, err := factory.Interpret(test.data)
		test.errFn(err)
		if !reflect.DeepEqual(test.expectedGVK, gvk) {
			t.Errorf("%d: unexpected GVK: %v, actual GVK: %v", i, test.expectedGVK, gvk)
		}
	}
}
