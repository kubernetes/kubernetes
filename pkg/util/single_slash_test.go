/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package util

import (
	"testing"
)

func TestSingleJoiningSlash(t *testing.T) {
	tests := []struct {
		a, b, expected string
	}{
		{
			a:        "one",
			b:        "two",
			expected: "one/two",
		},
		{
			a:        "one/",
			b:        "two",
			expected: "one/two",
		},
		{
			a:        "one",
			b:        "/two",
			expected: "one/two",
		},
		{
			a:        "one/",
			b:        "/two",
			expected: "one/two",
		},
		{
			a:        "one",
			b:        "",
			expected: "one/",
		},
		{
			a:        "",
			b:        "two",
			expected: "/two",
		},
		{
			a:        "",
			b:        "",
			expected: "/",
		},
	}

	for _, tc := range tests {
		if a, e := SingleJoiningSlash(tc.a, tc.b), tc.expected; a != e {
			t.Errorf("Expected %s for inputs %s and %s. Got: %s", e, tc.a, tc.b, a)
		}
	}
}
