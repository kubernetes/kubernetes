/*
Copyright 2018 The Kubernetes Authors.

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

package naming

import (
	"strings"
	"testing"
)

func TestGetNameFromCallsite(t *testing.T) {
	tests := []struct {
		name            string
		ignoredPackages []string
		expected        string
	}{
		{
			name:     "simple",
			expected: "k8s.io/apimachinery/pkg/util/naming/from_stack_test.go:",
		},
		{
			name:            "ignore-package",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming"},
			expected:        "testing/testing.go:",
		},
		{
			name:            "ignore-file",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming/from_stack_test.go"},
			expected:        "testing/testing.go:",
		},
		{
			name:            "ignore-multiple",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming/from_stack_test.go", "testing/testing.go"},
			expected:        "????",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := GetNameFromCallsite(tc.ignoredPackages...)
			if !strings.HasPrefix(actual, tc.expected) {
				t.Fatalf("expected string with prefix %q, got %q", tc.expected, actual)
			}
		})
	}
}
