/*
Copyright 2024 The Kubernetes Authors.

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

// Package template implements expanding a string where each occurrence of {{...}}
// is a CEL expression. The parameters for the template are a single nested map
// which is available under the name "object" in the CEL expressions.
package template

import (
	"context"
	"encoding/json"
	"testing"
)

// NewTemplate parses the template string and all CEL expressions embedded in
// it. If there are parse errors, then the returned error contains one error
// for each expression that failed to parse. All errors from the
// underlying CEL libraries are wrapped.
//
// Parsing is more permissive when using [environment.NewExpression] as
// environment type. Use [environment.StoredExpression] when dealing with
// templates which were already persisted to storage earlier.
//
// Options might get added in the future. None are defined right now.
func TestTemplate(t *testing.T) {
	testcases := map[string]struct {
		template     string
		object       string
		options      []Option
		expectResult string
		expectError  string
	}{
		"nop": {
			template:     "hello world",
			object:       `{}`,
			expectResult: "hello world",
		},
		"string": {
			template:     "hello {{object.string}}",
			object:       `{"string": "world"}`,
			expectResult: "hello world",
		},
		"double-error": {
			template:    "hello {{object.double}} world",
			object:      `{"double": 42.0}`,
			expectError: "CEL result of type double for placeholder at #6 could not be converted to string: type conversion error from Double to 'string'",
		},
		"double-json": {
			template:     "hello {{object.double}} world",
			object:       `{"double": 42.0}`,
			options:      []Option{ToStringConversion(ToJSON)},
			expectResult: `hello 42 world`,
		},
		"object": {
			template:     "{{object.spec}}",
			object:       `{"spec": {"double": 42.0}}`,
			options:      []Option{ToStringConversion(ToJSON)},
			expectResult: `{"double":42}`,
		},
	}

	for name, testcase := range testcases {
		t.Run(name, func(t *testing.T) {
			actual, err := expand(t, testcase.template, testcase.object, testcase.options)
			if err != nil {
				if testcase.expectError == "" {
					t.Fatalf("unexpected error: %v", err)
				}
				if err.Error() != testcase.expectError {
					t.Fatalf("expected:\n%s\nto equal:\n%s\n", err.Error(), testcase.expectError)
				}
				return
			}

			if actual != testcase.expectResult {
				t.Fatalf("expected:\n%s\nto equal:\n%s\n", actual, testcase.expectResult)
			}
		})
	}
}

func expand(t *testing.T, template, objectStr string, options []Option) (string, error) {
	var object map[string]any
	if err := json.Unmarshal([]byte(objectStr), &object); err != nil {
		t.Fatalf("internal error, testcase object not valid JSON: %v", err)
	}
	tmpl, err := NewTemplate(template, options...)
	if err != nil {
		return "", err
	}
	return tmpl.Expand(context.Background(), object)
}
