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

package sort

import (
	"testing"
)

func TestSortDiscoveryGroupsTopo(t *testing.T) {
	cases := []struct {
		name     string
		input    [][]string
		expected []string
	}{
		{
			name: "consensus ordering",
			input: [][]string{
				{"A", "B", "C", "D"},
				{"A", "B", "C", "D"},
				{"A", "X", "Z", "D"},
				{"Z", "Y"},
				{"Q"},
			},
			// "D" < "Q" < "Y" (lexical tiebreak)
			expected: []string{"A", "B", "C", "X", "Z", "D", "Q", "Y"},
		},
		{
			name:     "empty input",
			input:    [][]string{},
			expected: []string{},
		},
		{
			name:     "single peer",
			input:    [][]string{{"foo", "bar", "baz"}},
			expected: []string{"foo", "bar", "baz"},
		},
		{
			name:     "conflicting orderings",
			input:    [][]string{{"A", "B"}, {"B", "A"}},
			expected: []string{"A", "B"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := SortDiscoveryGroupsTopo(tc.input)
			if len(got) != len(tc.expected) {
				t.Errorf("length mismatch: got %v, expected %v", got, tc.expected)
				return
			}
			for i := range got {
				if got[i] != tc.expected[i] {
					t.Errorf("at %d: got %v, expected %v", i, got, tc.expected)
				}
			}
		})
	}
}
