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

package cm

import (
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
)

func TestFilterNUMATopology(t *testing.T) {
	topology := []cadvisorapi.Node{
		{Id: 0, Memory: 64 * 1024 * 1024 * 1024},
		{Id: 1, Memory: 64 * 1024 * 1024 * 1024},
		{Id: 2, Memory: 0},
		{Id: 3, Memory: 0},
		{Id: 4, Memory: 0},
	}

	tests := []struct {
		name        string
		excluded    []int
		expectedIDs []int
	}{
		{
			name:        "no exclusions keeps all nodes",
			excluded:    nil,
			expectedIDs: []int{0, 1, 2, 3, 4},
		},
		{
			name:        "exclude zero-memory nodes",
			excluded:    []int{2, 3, 4},
			expectedIDs: []int{0, 1},
		},
		{
			name:        "exclude subset of nodes",
			excluded:    []int{3},
			expectedIDs: []int{0, 1, 2, 4},
		},
		{
			name:        "exclude all nodes",
			excluded:    []int{0, 1, 2, 3, 4},
			expectedIDs: []int{},
		},
		{
			name:        "exclude non-existent node IDs is harmless",
			excluded:    []int{99, 100},
			expectedIDs: []int{0, 1, 2, 3, 4},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			filtered := filterNUMATopology(topology, tc.excluded)
			gotIDs := make([]int, len(filtered))
			for i, n := range filtered {
				gotIDs[i] = n.Id
			}
			if len(gotIDs) != len(tc.expectedIDs) {
				t.Fatalf("expected %v, got %v", tc.expectedIDs, gotIDs)
			}
			for i := range gotIDs {
				if gotIDs[i] != tc.expectedIDs[i] {
					t.Fatalf("expected %v, got %v", tc.expectedIDs, gotIDs)
				}
			}
		})
	}
}

func TestFilterNUMATopologyDeepCopy(t *testing.T) {
	topology := []cadvisorapi.Node{
		{
			Id:     0,
			Memory: 64 * 1024 * 1024 * 1024,
			Cores: []cadvisorapi.Core{
				{
					Id:      0,
					Threads: []int{0, 1},
					Caches: []cadvisorapi.Cache{
						{Id: 0, Size: 32 * 1024, Type: "data", Level: 1},
					},
					UncoreCaches: []cadvisorapi.Cache{
						{Id: 1, Size: 32 * 1024 * 1024, Type: "unified", Level: 3},
					},
				},
			},
			Caches: []cadvisorapi.Cache{
				{Id: 2, Size: 32 * 1024 * 1024, Type: "unified", Level: 3},
			},
			Distances: []uint64{10, 20},
		},
		{Id: 1, Memory: 0},
	}

	filtered := filterNUMATopology(topology, []int{1})
	if len(filtered) != 1 || filtered[0].Id != 0 {
		t.Fatalf("expected [0], got %v", filtered)
	}

	// Mutating the filtered result must not affect the original.
	filtered[0].Cores[0].Id = 99
	filtered[0].Cores[0].Threads[0] = 99
	filtered[0].Cores[0].Caches[0].Id = 99
	filtered[0].Cores[0].UncoreCaches[0].Id = 99
	filtered[0].Caches[0].Id = 99
	filtered[0].Distances[0] = 999

	if topology[0].Cores[0].Id != 0 {
		t.Error("filterNUMATopology did not deep-copy Cores: original was mutated")
	}
	if topology[0].Cores[0].Threads[0] != 0 {
		t.Error("filterNUMATopology did not deep-copy Core Threads: original was mutated")
	}
	if topology[0].Cores[0].Caches[0].Id != 0 {
		t.Error("filterNUMATopology did not deep-copy Core Caches: original was mutated")
	}
	if topology[0].Cores[0].UncoreCaches[0].Id != 1 {
		t.Error("filterNUMATopology did not deep-copy Core UncoreCaches: original was mutated")
	}
	if topology[0].Caches[0].Id != 2 {
		t.Error("filterNUMATopology did not deep-copy Node Caches: original was mutated")
	}
	if topology[0].Distances[0] != 10 {
		t.Error("filterNUMATopology did not deep-copy Distances: original was mutated")
	}
}
