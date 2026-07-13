//go:build goexperiment.jsonv2 || go1.27

/*
Copyright The Kubernetes Authors.

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

package json_test

import (
	"bytes"
	"strings"
	"testing"

	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

func TestDropFieldsOmitsManagedFields(t *testing.T) {
	full := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{})
	drop := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{DropFields: []string{"metadata.managedFields"}})

	list := &testapigroupv1.CarpList{Items: []testapigroupv1.Carp{*carpWithManagedFields(), *carpWithManagedFields()}}
	for _, tc := range []struct {
		name string
		obj  runtime.Object
	}{
		{"single", carpWithManagedFields()},
		{"list", list},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var fullBuf, dropBuf bytes.Buffer
			if err := full.Encode(tc.obj, &fullBuf); err != nil {
				t.Fatalf("full encode: %v", err)
			}
			if err := drop.Encode(tc.obj, &dropBuf); err != nil {
				t.Fatalf("drop encode: %v", err)
			}
			if !strings.Contains(fullBuf.String(), dropFieldsTestManager) {
				t.Fatalf("test fixture broken: full output lacks managedFields data")
			}
			if strings.Contains(dropBuf.String(), dropFieldsTestManager) {
				t.Errorf("managedFields data not dropped: %s", dropBuf.String())
			}
			if !strings.Contains(dropBuf.String(), `"name":"foo"`) {
				t.Errorf("rest of object not preserved: %s", dropBuf.String())
			}
		})
	}
}
