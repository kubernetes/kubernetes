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

package explain

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	tst "k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi/testing"
)

var resources = tst.NewFakeResources("test-swagger.json")

func TestReferenceTypename(t *testing.T) {
	schema := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "OneKind",
	})
	if schema == nil {
		t.Fatal("Couldn't find schema v1.OneKind")
	}

	tests := []struct {
		path     []string
		expected string
	}{
		{
			// Kind is "Object"
			path:     []string{},
			expected: "Object",
		},
		{
			// Reference is equal to pointed type "Object"
			path:     []string{"field1"},
			expected: "Object",
		},
		{
			// Reference is equal to pointed type "string"
			path:     []string{"field1", "primitive"},
			expected: "string",
		},
		{
			// Array of object of reference to string
			path:     []string{"field2"},
			expected: "[]map[string]string",
		},
		{
			// Array of integer
			path:     []string{"field1", "array"},
			expected: "[]integer",
		},
	}

	for _, test := range tests {
		s, err := LookupSchemaForField(schema, test.path)
		if err != nil {
			t.Fatalf("Invalid test.path %v: %v", test.path, err)
		}
		got := GetTypeName(s)
		if got != test.expected {
			t.Errorf("Got %q, expected %q", got, test.expected)
		}
	}

}
