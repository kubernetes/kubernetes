/*
Copyright 2025 The Kubernetes Authors.

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

package resourceversion

import (
	"testing"
)

func TestCompareResourceVersion(t *testing.T) {
	testCases := []struct {
		name     string
		a        string
		b        string
		expected int
		err      bool
	}{
		{
			name:     "a less than b",
			a:        "100",
			b:        "200",
			expected: -1,
		},
		{
			name: "a is zero, invalid",
			a:    "0",
			b:    "1",
			err:  true,
		},
		{
			name: "both zero",
			a:    "0",
			b:    "0",
			err:  true,
		},
		{
			name:     "a greater than b",
			a:        "200",
			b:        "100",
			expected: 1,
		},
		{
			name: "b is 0, invalid",
			a:    "1",
			b:    "0",
			err:  true,
		},
		{
			name:     "a equal to b small",
			a:        "1",
			b:        "1",
			expected: 0,
		},
		{
			name:     "a equal to b",
			a:        "100",
			b:        "100",
			expected: 0,
		},
		{
			name:     "a shorter than b",
			a:        "99",
			b:        "100",
			expected: -1,
		},
		{
			name:     "a longer than b",
			a:        "100",
			b:        "99",
			expected: 1,
		},
		{
			name:     "a with leading zero",
			a:        "0100",
			b:        "100",
			expected: 0,
			err:      true,
		},
		{
			name:     "b with leading zero",
			a:        "100",
			b:        "0100",
			expected: 0,
			err:      true,
		},
		{
			name:     "a empty",
			a:        "",
			b:        "100",
			expected: 0,
			err:      true,
		},
		{
			name:     "b empty",
			a:        "100",
			b:        "",
			expected: 0,
			err:      true,
		},
		{
			name: "a non-digit",
			a:    "100a",
			b:    "100",
			err:  true,
		},
		{
			name: "b non-digit",
			a:    "100",
			b:    "100a",
			err:  true,
		},
		{
			name:     "large int a less than b",
			a:        "99999999999999999999999999999999999999999999999999",
			b:        "100000000000000000000000000000000000000000000000000",
			expected: -1,
		},
		{
			name:     "large int a greater than b",
			a:        "100000000000000000000000000000000000000000000000000",
			b:        "99999999999999999999999999999999999999999999999999",
			expected: 1,
		},
		{
			name:     "large int a equal to b",
			a:        "12345678901234567890123456789012345678901234567890",
			b:        "12345678901234567890123456789012345678901234567890",
			expected: 0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := CompareResourceVersion(tc.a, tc.b)
			if tc.err {
				if err == nil {
					t.Fatalf("expected error, but got none")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if actual != tc.expected {
				t.Errorf("expected %d, got %d", tc.expected, actual)
			}
		})
	}
}
