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
)

func TestFindField(t *testing.T) {
	schema := resources.LookupResource(schema.GroupVersionKind{
		Group:   "",
		Version: "v1",
		Kind:    "OneKind",
	})
	if schema == nil {
		t.Fatal("Counldn't find schema v1.OneKind")
	}

	tests := []struct {
		path []string

		err          string
		expectedPath string
	}{
		{
			path:         []string{},
			expectedPath: "OneKind",
		},
		{
			path:         []string{"field1"},
			expectedPath: "OneKind.field1",
		},
		{
			path:         []string{"field1", "array"},
			expectedPath: "OtherKind.array",
		},
		{
			path: []string{"field1", "what?"},
			err:  `field "what?" does not exist`,
		},
		{
			path: []string{"field1", ""},
			err:  `field "" does not exist`,
		},
	}

	for _, test := range tests {
		path, err := LookupSchemaForField(schema, test.path)

		gotErr := ""
		if err != nil {
			gotErr = err.Error()
		}

		gotPath := ""
		if path != nil {
			gotPath = path.GetPath().String()
		}

		if gotErr != test.err || gotPath != test.expectedPath {
			t.Errorf("LookupSchemaForField(schema, %v) = (path: %q, err: %q), expected (path: %q, err: %q)",
				test.path, gotPath, gotErr, test.expectedPath, test.err)
		}
	}
}
