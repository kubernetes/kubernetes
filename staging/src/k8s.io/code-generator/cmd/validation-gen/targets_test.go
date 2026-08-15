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

package main

import (
	"testing"

	"k8s.io/gengo/v2/types"
)

func TestParseDeepEqualFunc(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected types.Name
	}{{
		name:     "fully qualified with struct field",
		input:    "k8s.io/apimachinery/pkg/api/equality.Semantic.DeepEqual",
		expected: types.Name{Package: "k8s.io/apimachinery/pkg/api/equality", Name: "Semantic.DeepEqual"},
	}, {
		name:     "fully qualified package function",
		input:    "k8s.io/code-generator/cmd/validation-gen/testscheme.CustomEqual",
		expected: types.Name{Package: "k8s.io/code-generator/cmd/validation-gen/testscheme", Name: "CustomEqual"},
	}, {
		name:     "single level package function",
		input:    "testscheme.CustomEqual",
		expected: types.Name{Package: "testscheme", Name: "CustomEqual"},
	}, {
		name:     "single level package struct field",
		input:    "testscheme.Semantic.DeepEqual",
		expected: types.Name{Package: "testscheme", Name: "Semantic.DeepEqual"},
	}, {
		name:     "package-local function",
		input:    "CustomDeepEqual",
		expected: types.Name{Package: "", Name: "CustomDeepEqual"},
	}, {
		name:     "empty input",
		input:    "",
		expected: types.Name{Package: "", Name: ""},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := parseDeepEqualFunc(tc.input)
			if want, got := tc.expected, result; got != want {
				t.Errorf("expected %v, got %v", want, got)
			}
		})
	}
}
