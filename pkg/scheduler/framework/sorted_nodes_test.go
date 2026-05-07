/*
Copyright 2026 The Kubernetes Authors.

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

package framework

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	fwk "k8s.io/kube-scheduler/framework"
)

func TestSortedScoredNodes_Pop(t *testing.T) {
	tests := []struct {
		name     string
		input    []fwk.NodePluginScores
		expected []fwk.NodePluginScores
	}{
		{
			name:     "empty",
			input:    []fwk.NodePluginScores{},
			expected: []fwk.NodePluginScores{},
		},
		{
			name: "single node",
			input: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 5},
			},
			expected: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 5},
			},
		},
		{
			name: "descending by TotalScore",
			input: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node3", TotalScore: 3},
				{Name: "node2", TotalScore: 2},
			},
			expected: []fwk.NodePluginScores{
				{Name: "node3", TotalScore: 3},
				{Name: "node2", TotalScore: 2},
				{Name: "node1", TotalScore: 1},
			},
		},
		{
			name: "tie-break by Randomizer",
			input: []fwk.NodePluginScores{
				{Name: "nodeA", TotalScore: 5, Randomizer: 1},
				{Name: "nodeB", TotalScore: 5, Randomizer: 3},
				{Name: "nodeC", TotalScore: 5, Randomizer: 2},
			},
			expected: []fwk.NodePluginScores{
				{Name: "nodeB", TotalScore: 5, Randomizer: 3},
				{Name: "nodeC", TotalScore: 5, Randomizer: 2},
				{Name: "nodeA", TotalScore: 5, Randomizer: 1},
			},
		},
		{
			name: "mixed scores and tie-breaks",
			input: []fwk.NodePluginScores{
				{Name: "node3.1", TotalScore: 3, Randomizer: 1},
				{Name: "node2", TotalScore: 2},
				{Name: "node1", TotalScore: 1},
				{Name: "node3.2", TotalScore: 3, Randomizer: 2},
			},
			expected: []fwk.NodePluginScores{
				{Name: "node3.2", TotalScore: 3, Randomizer: 2},
				{Name: "node3.1", TotalScore: 3, Randomizer: 1},
				{Name: "node2", TotalScore: 2},
				{Name: "node1", TotalScore: 1},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := NewSortedScoredNodes(tc.input)
			var got []fwk.NodePluginScores
			for s.Len() > 0 {
				got = append(got, s.Pop())
			}
			if diff := cmp.Diff(tc.expected, got, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("pop order mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSortedScoredNodes_Len(t *testing.T) {
	nodes := []fwk.NodePluginScores{
		{Name: "node1", TotalScore: 10},
		{Name: "node2", TotalScore: 20},
		{Name: "node3", TotalScore: 30},
	}
	s := NewSortedScoredNodes(nodes)
	for want := len(nodes); want > 0; want-- {
		if got := s.Len(); got != want {
			t.Fatalf("Len() = %d, want %d", got, want)
		}
		s.Pop()
	}
	if got := s.Len(); got != 0 {
		t.Errorf("Len() after all pops = %d, want 0", got)
	}
}

func TestSortedScoredNodes_List(t *testing.T) {
	sortByName := cmpopts.SortSlices(func(a, b fwk.NodePluginScores) bool {
		return a.Name < b.Name
	})
	tests := []struct {
		name     string
		input    []fwk.NodePluginScores
		popCount int
		expected []fwk.NodePluginScores
	}{
		{
			name:     "empty",
			input:    []fwk.NodePluginScores{},
			expected: []fwk.NodePluginScores{},
		},
		{
			name: "all nodes before any pops",
			input: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
				{Name: "node3", TotalScore: 3},
			},
			expected: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
				{Name: "node3", TotalScore: 3},
			},
		},
		{
			name: "remaining nodes after a pop",
			input: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
				{Name: "node3", TotalScore: 3},
			},
			popCount: 1,
			expected: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
			},
		},
		{
			name: "empty after all pops",
			input: []fwk.NodePluginScores{
				{Name: "node1", TotalScore: 1},
				{Name: "node2", TotalScore: 2},
			},
			popCount: 2,
			expected: []fwk.NodePluginScores{},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := NewSortedScoredNodes(tc.input)
			for range tc.popCount {
				s.Pop()
			}
			got := s.List()
			if diff := cmp.Diff(tc.expected, got, sortByName, cmpopts.EquateEmpty()); diff != "" {
				t.Errorf("List() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
