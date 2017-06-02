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

package set

import "testing"

func TestSelectString(t *testing.T) {
	testCases := []struct {
		s        string
		spec     string
		expected bool
	}{
		{"abcd", "*", true},
		{"abcd", "dcba", false},
		{"abcd", "***d", true},
		{"abcd", "*bcd", true},
		{"abcd", "*bc", false},
		{"abcd", "*cd", true},
		{"abcd", "a*d", true},
		{"abcd", "a*cd", true},
		{"abcde", "a*c*e", false},
	}
	for _, item := range testCases {
		if actual := selectString(item.s, item.spec); actual != item.expected {
			t.Errorf("Expected: %v, got: %v for spec: %s, s: %s ", item.expected, actual, item.spec, item.s)
		}
	}
}
