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

import "testing"

func TestGetNameFromCallsite(t *testing.T) {
	// We do some tests which ignore our package, and therefore walk
	// into our caller, which is the testing framework.  That callsite
	// varies by go version.
	testingCallsites := []string{
		"testing/testing.go:777", // go 1.10
		"testing/testing.go:827", // go 1.11
	}

	tests := []struct {
		name            string
		ignoredPackages []string
		expected        []string
	}{
		{
			name:     "simple",
			expected: []string{"k8s.io/apimachinery/pkg/util/naming/from_stack_test.go:58"},
		},
		{
			name:            "ignore-package",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming"},
			expected:        testingCallsites,
		},
		{
			name:            "ignore-file",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming/from_stack_test.go"},
			expected:        testingCallsites,
		},
		{
			name:            "ignore-multiple",
			ignoredPackages: []string{"k8s.io/apimachinery/pkg/util/naming/from_stack_test.go", "testing/testing.go"},
			expected:        []string{"????"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			actual := GetNameFromCallsite(tc.ignoredPackages...)
			match := false
			for _, e := range tc.expected {
				if e == actual {
					match = true
					break
				}
			}

			if !match {
				t.Errorf("expected one of %v, got %q", tc.expected, actual)
			}
		})
	}
}
