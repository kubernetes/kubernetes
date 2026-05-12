/*
Copyright 2019 The Kubernetes Authors.

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

package yaml

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestInterpret(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected *schema.GroupVersionKind
		errFn    func(error) bool
	}{
		{
			name: "YAMLSuccessfullyInterpretVK",
			input: `apiVersion: v1
kind: Service`,
			expected: &schema.GroupVersionKind{Version: "v1", Kind: "Service"},
		},
		{
			name: "YAMLSuccessfullyInterpretGVK",
			input: `apiVersion: core/v2
kind: Deployment`,
			expected: &schema.GroupVersionKind{Group: "core", Version: "v2", Kind: "Deployment"},
		},
		{
			name:     "YAMLSuccessfullyInterpretV",
			input:    `apiVersion: v1`,
			expected: &schema.GroupVersionKind{Version: "v1"},
		},
		{
			name:     "YAMLSuccessfullyInterpretK",
			input:    `kind: Service`,
			expected: &schema.GroupVersionKind{Kind: "Service"},
		},
		{
			name:     "YAMLSuccessfullyInterpretEmptyString",
			input:    ``,
			expected: &schema.GroupVersionKind{},
		},
		{
			name:     "YAMLSuccessfullyInterpretEmptyDoc",
			input:    `---`,
			expected: &schema.GroupVersionKind{},
		},
		{
			name: "YAMLSuccessfullyInterpretMultiDoc",
			input: `---
apiVersion: v1
kind: Service
---
apiVersion: v2
kind: Deployment`,
			expected: &schema.GroupVersionKind{Version: "v1", Kind: "Service"},
		},
		{
			name:     "YAMLSuccessfullyInterpretOnlyG",
			input:    `apiVersion: core/`,
			expected: &schema.GroupVersionKind{Group: "core"},
		},
		{
			name:     "YAMLSuccessfullyWrongFormat",
			input:    `foo: bar`,
			expected: &schema.GroupVersionKind{},
		},
		{
			name:  "YAMLFailInterpretWrongSyntax",
			input: `foo`,
			errFn: func(err error) bool { return err != nil },
		},
		{
			name:     "JSONSuccessfullyInterpretVK",
			input:    `{"apiVersion": "v3", "kind": "DaemonSet"}`,
			expected: &schema.GroupVersionKind{Version: "v3", Kind: "DaemonSet"},
		},
		{
			name:     "JSONSuccessfullyInterpretGVK",
			input:    `{"apiVersion": "core/v2", "kind": "Deployment"}`,
			expected: &schema.GroupVersionKind{Group: "core", Version: "v2", Kind: "Deployment"},
		},
		{
			name:     "JSONSuccessfullyInterpretV",
			input:    `{"apiVersion": "v1"}`,
			expected: &schema.GroupVersionKind{Version: "v1"},
		},
		{
			name:     "JSONSuccessfullyInterpretK",
			input:    `{"kind": "Service"}`,
			expected: &schema.GroupVersionKind{Kind: "Service"},
		},
		{
			name:     "JSONSuccessfullyInterpretEmptyString",
			input:    ``,
			expected: &schema.GroupVersionKind{},
		},
		{
			name:     "JSONSuccessfullyInterpretEmptyObject",
			input:    `{}`,
			expected: &schema.GroupVersionKind{},
		},
		{
			name: "JSONSuccessfullyInterpretMultiDoc",
			input: `{"apiVersion": "v1", "kind": "Service"}, 
{"apiVersion": "v2", "kind": "Deployment"}`,
			expected: &schema.GroupVersionKind{Version: "v1", Kind: "Service"},
		},
		{
			name:     "JSONSuccessfullyWrongFormat",
			input:    `{"foo": "bar"}`,
			expected: &schema.GroupVersionKind{},
		},
		{
			name:  "JSONFailInterpretArray",
			input: `[]`,
			errFn: func(err error) bool { return err != nil },
		},
		{
			name:  "JSONFailInterpretWrongSyntax",
			input: `{"foo"`,
			errFn: func(err error) bool { return err != nil },
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			actual, err := DefaultMetaFactory.Interpret([]byte(test.input))
			switch {
			case test.errFn != nil:
				if !test.errFn(err) {
					t.Errorf("unexpected error: %v", err)
				}
			case err != nil:
				t.Errorf("unexpected error: %v", err)
			case !reflect.DeepEqual(test.expected, actual):
				t.Errorf("outcome mismatch -- expected: %#v, actual: %#v",
					test.expected, actual,
				)
			}
		})
	}
}
