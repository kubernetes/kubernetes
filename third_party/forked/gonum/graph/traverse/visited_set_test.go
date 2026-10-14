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

package traverse

import (
	"math/rand"
	"testing"
)

func TestVisitedSet(t *testing.T) {
	testcases := []struct {
		name string
		ids  []int
	}{
		{name: "empty"},
		{name: "single", ids: []int{0}},
		// Stays in the linear list.
		{name: "below promotion threshold", ids: []int{7, 3, 64, 65, 1000000}},
		// Crosses over into the map.
		{name: "above promotion threshold", ids: seq(visitedSetMaxIDs * 4)},
		{name: "descending", ids: reverse(seq(visitedSetMaxIDs * 4))},
		{name: "sparse and interleaved", ids: []int{0, 1000000, 63, 64, 999999, 1, 4096, 2, 500000, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}},
		{name: "negative", ids: append(seq(visitedSetMaxIDs*2), -1, -64, -65)},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			var s visitedSet
			for i, id := range tc.ids {
				if s.has(id) {
					t.Fatalf("id %d reported visited before insert", id)
				}
				s.insert(id)
				for _, inserted := range tc.ids[:i+1] {
					if !s.has(inserted) {
						t.Fatalf("id %d reported unvisited after inserting %d", inserted, id)
					}
				}
			}
			if id := maxID(tc.ids) + 1; s.has(id) {
				t.Errorf("id %d reported visited but was never inserted", id)
			}

			s.clear()
			for _, id := range tc.ids {
				if s.has(id) {
					t.Fatalf("id %d still reported visited after clear", id)
				}
			}
		})
	}
}

// TestVisitedSetMatchesReference checks the set against a plain map for a
// random ID mix, since the switch to the map happens part way through.
func TestVisitedSetMatchesReference(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	var s visitedSet
	want := map[int]bool{}
	for i := 0; i < 10000; i++ {
		id := r.Intn(1 << 20)
		if got := s.has(id); got != want[id] {
			t.Fatalf("has(%d) = %t, want %t", id, got, want[id])
		}
		s.insert(id)
		want[id] = true
	}
}

func seq(n int) []int {
	ids := make([]int, n)
	for i := range ids {
		// Spread the IDs out rather than handing back a dense range.
		ids[i] = i * 997
	}
	return ids
}

func reverse(ids []int) []int {
	for i, j := 0, len(ids)-1; i < j; i, j = i+1, j-1 {
		ids[i], ids[j] = ids[j], ids[i]
	}
	return ids
}

func maxID(ids []int) int {
	maximum := 0
	for _, id := range ids {
		maximum = max(maximum, id)
	}
	return maximum
}
