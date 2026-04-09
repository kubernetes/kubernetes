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
	"hash/crc32"
	"sort"
)

// MemberID uniquely identifies a member within a watch group.
// Typically the Pod name of the controller.
type MemberID string

// HashRing implements consistent hashing for distributing resources among group members.
// Each member is represented by multiple virtual nodes (replicas) on the ring to ensure
// uniform distribution.
type HashRing struct {
	points    []uint64
	pointToID map[uint64]MemberID
	members   map[MemberID]struct{}
	replicas  int
}

// NewHashRing creates a new consistent hash ring with the given number of virtual nodes
// per member.
func NewHashRing(replicas int) *HashRing {
	return &HashRing{
		pointToID: make(map[uint64]MemberID),
		members:   make(map[MemberID]struct{}),
		replicas:  replicas,
	}
}

// AddNode adds a member to the hash ring with virtual nodes.
func (r *HashRing) AddNode(memberID MemberID) {
	if _, exists := r.members[memberID]; exists {
		return
	}
	r.members[memberID] = struct{}{}
	for i := 0; i < r.replicas; i++ {
		// Combine replica index and memberID to generate virtual node positions.
		h := hashKey(fmt.Sprintf("%d/%s", i, memberID))
		r.points = append(r.points, h)
		r.pointToID[h] = memberID
	}
	sort.Slice(r.points, func(i, j int) bool { return r.points[i] < r.points[j] })
}

// RemoveNode removes a member and all its virtual nodes from the ring.
func (r *HashRing) RemoveNode(memberID MemberID) {
	if _, exists := r.members[memberID]; !exists {
		return
	}
	delete(r.members, memberID)

	toRemove := make(map[uint64]bool)
	for i := 0; i < r.replicas; i++ {
		h := hashKey(fmt.Sprintf("%d/%s", i, memberID))
		toRemove[h] = true
		delete(r.pointToID, h)
	}

	newPoints := make([]uint64, 0, len(r.points)-len(toRemove))
	for _, p := range r.points {
		if !toRemove[p] {
			newPoints = append(newPoints, p)
		}
	}
	r.points = newPoints
}

// GetNode returns the member responsible for the given resource key.
// Returns empty MemberID if the ring is empty.
func (r *HashRing) GetNode(key string) MemberID {
	if len(r.points) == 0 {
		return ""
	}
	h := hashKey(key)
	idx := sort.Search(len(r.points), func(i int) bool { return r.points[i] >= h })
	if idx == len(r.points) {
		idx = 0 // wrap around
	}
	return r.pointToID[r.points[idx]]
}

// GetAllNodes returns all member IDs in the ring.
func (r *HashRing) GetAllNodes() []MemberID {
	result := make([]MemberID, 0, len(r.members))
	for id := range r.members {
		result = append(result, id)
	}
	return result
}

// NodeCount returns the number of members in the ring.
func (r *HashRing) NodeCount() int {
	return len(r.members)
}

// Clone creates a deep copy of the hash ring for rebalance computation.
func (r *HashRing) Clone() *HashRing {
	c := &HashRing{
		points:    make([]uint64, len(r.points)),
		pointToID: make(map[uint64]MemberID, len(r.pointToID)),
		members:   make(map[MemberID]struct{}, len(r.members)),
		replicas:  r.replicas,
	}
	copy(c.points, r.points)
	for k, v := range r.pointToID {
		c.pointToID[k] = v
	}
	for k, v := range r.members {
		c.members[k] = v
	}
	return c
}

// hashKey computes a deterministic hash for the given key using CRC32 (IEEE).
// CRC32 is the standard choice for consistent hash rings (used by GroupCache,
// libketama, etc.). It is deterministic across processes and machines, which
// is critical for multiple API server replicas to agree on resource ownership.
func hashKey(key string) uint64 {
	return uint64(crc32.ChecksumIEEE([]byte(key)))
}
