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

package store

import (
	"sort"
	"sync"

	"k8s.io/klog/v2"
)

// NewSnapshotter returns a Snapshotter that stores snapshots for serving read
// requests with exact resource versions (RV) and pagination.
//
// Snapshots are created by calling Clone on an OrderedLister, which is expected
// to be fast and efficient thanks to the copy-on-write semantics of B-trees.
//
// The implementation uses a sorted slice with binary search, leveraging the
// monotonic-append property: Add always appends to the end (increasing RV) and
// RemoveLess only removes from the front (increasing minimum RV). This gives:
//   - Add:            O(1) amortized (slice append)
//   - GetLessOrEqual: O(log n) (binary search)
//   - RemoveLess:     O(log n + k) where k = removed entries (binary search + head advance)
//   - Reset:          O(1)
//   - Len:            O(1)
func NewSnapshotter() Snapshotter {
	return &sliceSnapshotter{}
}

var _ Snapshotter = (*sliceSnapshotter)(nil)

type sliceSnapshotter struct {
	mux       sync.RWMutex
	snapshots []rvSnapshot
	head      int // index of first live element
}

func (s *sliceSnapshotter) Add(rv uint64, indexer OrderedLister) {
	s.mux.Lock()
	defer s.mux.Unlock()
	if live := len(s.snapshots) - s.head; live > 0 {
		last := s.snapshots[len(s.snapshots)-1]
		if rv <= last.resourceVersion {
			klog.Warningf("snapshotter: Add called with rv %d <= last rv %d; snapshots must be added in increasing order", rv, last.resourceVersion)
		}
	}
	s.snapshots = append(s.snapshots, rvSnapshot{resourceVersion: rv, snapshot: indexer.Clone()})
}

func (s *sliceSnapshotter) GetLessOrEqual(rv uint64) (OrderedLister, bool) {
	s.mux.RLock()
	defer s.mux.RUnlock()
	live := s.snapshots[s.head:]
	if len(live) == 0 {
		return nil, false
	}
	// Find the first entry with RV > rv.
	i := sort.Search(len(live), func(i int) bool {
		return live[i].resourceVersion > rv
	})
	// The entry at i-1 has the greatest RV <= rv.
	if i == 0 {
		return nil, false
	}
	return live[i-1].snapshot, true
}

func (s *sliceSnapshotter) RemoveLess(rv uint64) {
	s.mux.Lock()
	defer s.mux.Unlock()
	live := s.snapshots[s.head:]
	if len(live) == 0 {
		return
	}
	// Find the first entry with RV >= rv.
	i := sort.Search(len(live), func(i int) bool {
		return live[i].resourceVersion >= rv
	})
	// Nil out removed entries so GC can collect the snapshots.
	for j := 0; j < i; j++ {
		s.snapshots[s.head+j] = rvSnapshot{}
	}
	s.head += i
	s.compact()
}

// compact copies live entries to the front of the slice when the dead prefix
// (head) occupies at least half the capacity. This keeps amortized cost O(1)
// per removal.
func (s *sliceSnapshotter) compact() {
	if s.head > 0 && s.head >= len(s.snapshots)/2 {
		live := s.snapshots[s.head:]
		n := copy(s.snapshots, live)
		// Zero trailing slots for GC.
		for i := n; i < len(s.snapshots); i++ {
			s.snapshots[i] = rvSnapshot{}
		}
		s.snapshots = s.snapshots[:n]
		s.head = 0
	}
}

func (s *sliceSnapshotter) Reset() {
	s.mux.Lock()
	defer s.mux.Unlock()
	s.snapshots = nil
	s.head = 0
}

func (s *sliceSnapshotter) Len() int {
	s.mux.RLock()
	defer s.mux.RUnlock()
	return len(s.snapshots) - s.head
}
