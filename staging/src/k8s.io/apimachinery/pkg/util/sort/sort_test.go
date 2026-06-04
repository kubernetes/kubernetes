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
		name  string
		input [][]string
		want  []string
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
			want: []string{"A", "B", "C", "Q", "X", "Z", "D", "Y"},
		},
		{
			name:  "empty input",
			input: [][]string{},
			want:  []string{},
		},
		{
			name:  "single peer",
			input: [][]string{{"foo", "bar", "baz"}},
			want:  []string{"foo", "bar", "baz"},
		},
		{
			name:  "conflicting orderings",
			input: [][]string{{"A", "B"}, {"B", "A"}},
			want:  []string{"A", "B"},
		},
		{
			name:  "empty list merged with non-empty list",
			input: [][]string{{}, {"A", "B", "C"}},
			want:  []string{"A", "B", "C"},
		},
		{
			name:  "multiple empty lists merged",
			input: [][]string{{}, {}, {}},
			want:  []string{},
		},
		{
			name: "lexical tiebreak at beginning",
			input: [][]string{
				{"C", "D", "E"},
				{"B", "D", "E"},
				{"A", "D", "E"},
			},
			// A, B, C have no precedence constraints, so lexical order
			want: []string{"A", "B", "C", "D", "E"},
		},
		{
			name: "lexical tiebreak in middle",
			input: [][]string{
				{"A", "D", "E"},
				{"A", "C", "E"},
				{"A", "B", "E"},
			},
			// A comes first (consensus), then B, C, D (lexical), then E (consensus)
			want: []string{"A", "B", "C", "D", "E"},
		},
		{
			name: "conflicting orderings of 3 lists",
			input: [][]string{
				{"A", "B", "C"},
				{"B", "C", "A"},
				{"C", "A", "B"},
			},
			// Creates cycle: A->B, B->C, C->A
			// Fallback to lexicographic sort
			want: []string{"A", "B", "C"},
		},
		{
			name: "conflicting ordering with different list lengths",
			input: [][]string{
				{"A", "B", "C", "D"},
				{"B", "A"},
				{"C", "D"},
			},
			// A->B->C->D from first list, but B->A from second
			// Creates cycle between A and B
			// Fallback to lexicographic sort
			want: []string{"A", "B", "C", "D"},
		},
		{
			name: "conflicting partial lists",
			input: [][]string{
				{"A", "B"},
				{"C", "D"},
				{"B", "A"},
			},
			// A->B from first, B->A from third (cycle)
			// C->D is independent
			// Fallback to lexicographic sort
			want: []string{"A", "B", "C", "D"},
		},
		{
			name: "cycle",
			input: [][]string{
				{"A", "B"},
				{"B", "C"},
				{"C", "A"},
			},
			// Creates cycle: A->B->C->A
			// Fallback to lexicographic sort
			want: []string{"A", "B", "C"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := MergePreservingRelativeOrder(tc.input)
			if len(got) != len(tc.want) {
				t.Errorf("length mismatch:\n  got: %d\n  want: %d", len(got), len(tc.want))
				return
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("mismatch got: %v\n  want: %v", got, tc.want)
					return
				}
			}
		})
	}
}
