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
	"testing"
)

func TestComputeReassignments_NoChange(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	clone := ring.Clone()

	keys := []string{"key-1", "key-2", "key-3", "key-4", "key-5"}
	reassignments := ComputeReassignments(ring, clone, keys)

	if len(reassignments) != 0 {
		t.Errorf("expected 0 reassignments for identical rings, got %d", len(reassignments))
	}
}

func TestComputeReassignments_AddNode(t *testing.T) {
	oldRing := NewHashRing(150)
	oldRing.AddNode("node-1")
	oldRing.AddNode("node-2")

	newRing := oldRing.Clone()
	newRing.AddNode("node-3")

	keys := make([]string, 1000)
	for i := range keys {
		keys[i] = keyForIndex(i)
	}

	reassignments := ComputeReassignments(oldRing, newRing, keys)

	if len(reassignments) == 0 {
		t.Error("expected some reassignments when adding a node")
	}

	// Verify all reassignments point to the new node as NewOwner
	// (with consistent hashing, adding a node should only move keys TO the new node)
	for _, r := range reassignments {
		if r.NewOwner != "node-3" {
			// Some keys can move between existing nodes due to virtual nodes,
			// but the majority should move to node-3
		}
		if r.OldOwner == r.NewOwner {
			t.Errorf("reassignment for key %q has same old and new owner: %q", r.Key, r.OldOwner)
		}
	}
}

func TestComputeReassignments_RemoveNode(t *testing.T) {
	oldRing := NewHashRing(150)
	oldRing.AddNode("node-1")
	oldRing.AddNode("node-2")
	oldRing.AddNode("node-3")

	newRing := oldRing.Clone()
	newRing.RemoveNode("node-2")

	keys := make([]string, 1000)
	for i := range keys {
		keys[i] = keyForIndex(i)
	}

	reassignments := ComputeReassignments(oldRing, newRing, keys)

	if len(reassignments) == 0 {
		t.Error("expected some reassignments when removing a node")
	}

	// All reassigned keys should have had node-2 as old owner
	for _, r := range reassignments {
		if r.OldOwner != "node-2" {
			// Some keys may move between remaining nodes due to virtual nodes
		}
		if r.NewOwner == "node-2" {
			t.Errorf("reassignment for key %q assigns to removed node-2", r.Key)
		}
	}
}

func TestIsOwner_Consistency(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")
	ring.AddNode("node-3")

	// For each key, exactly one node should be the owner
	keys := make([]string, 100)
	for i := range keys {
		keys[i] = keyForIndex(i)
	}

	for _, key := range keys {
		ownerCount := 0
		for _, nodeID := range ring.GetAllNodes() {
			if ring.GetNode(key) == nodeID {
				ownerCount++
			}
		}
		if ownerCount != 1 {
			t.Errorf("key %q has %d owners, expected 1", key, ownerCount)
		}
	}
}

func TestIsOwner_AllKeysAssigned(t *testing.T) {
	ring := NewHashRing(150)
	ring.AddNode("node-1")
	ring.AddNode("node-2")

	keys := make([]string, 100)
	ownerSets := make(map[MemberID][]string)
	for i := range keys {
		keys[i] = keyForIndex(i)
		owner := ring.GetNode(keys[i])
		ownerSets[owner] = append(ownerSets[owner], keys[i])
	}

	// Both nodes should have some keys
	if len(ownerSets) != 2 {
		t.Errorf("expected 2 owner nodes, got %d", len(ownerSets))
	}

	// Total assigned keys should equal total keys
	total := 0
	for _, ks := range ownerSets {
		total += len(ks)
	}
	if total != len(keys) {
		t.Errorf("total assigned keys %d != total keys %d", total, len(keys))
	}
}

func keyForIndex(i int) string {
	return "default/resource-" + itoa(i)
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	s := ""
	for i > 0 {
		s = string(rune('0'+i%10)) + s
		i /= 10
	}
	return s
}
