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

package watchgroup

import (
	"fmt"
	"math"
	"testing"
)

func TestHashRing_EmptyRing(t *testing.T) {
	ring := NewHashRing(150)
	if got := ring.GetNode("any-key"); got != "" {
		t.Errorf("expected empty MemberID for empty ring, got %q", got)
	}
	if got := ring.NodeCount(); got != 0 {
		t.Errorf("expected 0 nodes, got %d", got)
	}
}

func TestHashRing_SingleNode(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")

	if ring.NodeCount() != 1 {
		t.Fatalf("expected 1 node, got %d", ring.NodeCount())
	}

	// All keys should map to the single node
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("resource-%d", i)
		if got := ring.GetNode(key); got != "node-1" {
			t.Errorf("key %q: expected node-1, got %q", key, got)
		}
	}
}

func TestHashRing_AddRemove(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	if ring.NodeCount() != 2 {
		t.Fatalf("expected 2 nodes, got %d", ring.NodeCount())
	}

	ring.RemoveNode("node-1")
	if ring.NodeCount() != 1 {
		t.Fatalf("expected 1 node after removal, got %d", ring.NodeCount())
	}

	// All keys should now map to node-2
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("resource-%d", i)
		if got := ring.GetNode(key); got != "node-2" {
			t.Errorf("key %q: expected node-2, got %q", key, got)
		}
	}
}

func TestHashRing_DuplicateAdd(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-1") // duplicate

	if ring.NodeCount() != 1 {
		t.Fatalf("expected 1 node after duplicate add, got %d", ring.NodeCount())
	}
}

func TestHashRing_RemoveNonexistent(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.RemoveNode("node-2") // doesn't exist

	if ring.NodeCount() != 1 {
		t.Fatalf("expected 1 node, got %d", ring.NodeCount())
	}
}

func TestHashRing_Distribution(t *testing.T) {
	tests := []struct {
		name      string
		nodes     []MemberID
		numKeys   int
		keyFunc   func(i int) string
		maxDevPct float64
	}{
		{
			name:      "short node names, simple keys",
			nodes:     []MemberID{"node-1", "node-2", "node-3"},
			numKeys:   10000,
			keyFunc:   func(i int) string { return fmt.Sprintf("default/resource-%d", i) },
			maxDevPct: 0.45,
		},
		{
			name:      "realistic k8s pod names and resource keys",
			nodes:     []MemberID{"sample-controller-manager-7685d875dd-kzm9f", "sample-controller-manager-7685d875dd-l5dnn", "sample-controller-manager-7685d875dd-s8nzf"},
			numKeys:   1000,
			keyFunc:   func(i int) string { return fmt.Sprintf("/piny940.com/customers/default/customer-%04d", i) },
			maxDevPct: 0.45,
		},
		{
			name:      "2 nodes, realistic keys",
			nodes:     []MemberID{"pod-abc123", "pod-def456"},
			numKeys:   10000,
			keyFunc:   func(i int) string { return fmt.Sprintf("/apps/v1/deployments/ns/deploy-%d", i) },
			maxDevPct: 0.45,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ring := NewHashRing(256)
			for _, n := range tt.nodes {
				ring.AddNode(n)
			}

			counts := make(map[MemberID]int)
			for i := 0; i < tt.numKeys; i++ {
				owner := ring.GetNode(tt.keyFunc(i))
				counts[owner]++
			}

			expected := float64(tt.numKeys) / float64(len(tt.nodes))
			for id, count := range counts {
				deviation := math.Abs(float64(count)-expected) / expected
				if deviation > tt.maxDevPct {
					t.Errorf("node %q has %d keys (expected ~%.0f, deviation %.1f%%)",
						id, count, expected, deviation*100)
				}
			}

			if len(counts) != len(tt.nodes) {
				t.Errorf("expected %d nodes with keys, got %d", len(tt.nodes), len(counts))
			}
		})
	}
}

func TestHashRing_MinimalReassignment(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	numKeys := 1000
	keys := make([]string, numKeys)
	originalOwner := make(map[string]MemberID)

	for i := 0; i < numKeys; i++ {
		keys[i] = fmt.Sprintf("default/resource-%d", i)
		originalOwner[keys[i]] = ring.GetNode(keys[i])
	}

	// Add a third node
	ring.AddNode("node-3")

	reassigned := 0
	for _, key := range keys {
		if ring.GetNode(key) != originalOwner[key] {
			reassigned++
		}
	}

	// Ideally ~1/3 of keys should be reassigned when going from 2 to 3 nodes.
	// Allow generous tolerance.
	maxReassigned := numKeys * 2 / 3
	if reassigned > maxReassigned {
		t.Errorf("too many reassignments: %d out of %d (max expected %d)",
			reassigned, numKeys, maxReassigned)
	}
	if reassigned == 0 {
		t.Error("expected some reassignments when adding a node, got 0")
	}
}

func TestHashRing_Clone(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	clone := ring.Clone()

	// Verify clone has same state
	if clone.NodeCount() != ring.NodeCount() {
		t.Fatalf("clone node count %d != original %d", clone.NodeCount(), ring.NodeCount())
	}

	// Modify clone and verify original is unchanged
	clone.AddNode("node-3")
	if ring.NodeCount() != 2 {
		t.Error("modifying clone affected original ring")
	}
	if clone.NodeCount() != 3 {
		t.Error("clone should have 3 nodes")
	}

	// Verify same key assignments
	for i := 0; i < 100; i++ {
		key := fmt.Sprintf("key-%d", i)
		ringOwner := ring.GetNode(key)
		// Clone now has 3 nodes so some will differ, just check the ring is functional
		cloneOwner := clone.GetNode(key)
		if cloneOwner == "" {
			t.Errorf("clone returned empty owner for key %q", key)
		}
		_ = ringOwner // suppress unused
	}
}

func TestHashRing_GetAllNodes(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-a")
	ring.AddNode("node-b")
	ring.AddNode("node-c")

	nodes := ring.GetAllNodes()
	if len(nodes) != 3 {
		t.Fatalf("expected 3 nodes, got %d", len(nodes))
	}

	nodeSet := make(map[MemberID]bool)
	for _, n := range nodes {
		nodeSet[n] = true
	}
	for _, expected := range []MemberID{"node-a", "node-b", "node-c"} {
		if !nodeSet[expected] {
			t.Errorf("missing node %q", expected)
		}
	}
}
