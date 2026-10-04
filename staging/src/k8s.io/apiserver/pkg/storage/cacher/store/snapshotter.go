/*
Copyright 2022 The Kubernetes Authors.

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
	"k8s.io/utils/third_party/forked/golang/btree"
)

// newStoreSnapshotter returns a storeSnapshotter that stores snapshots for
// serving read requests with exact resource versions (RV) and pagination.
//
// Snapshots are created by calling Clone method on orderedLister, which is
// expected to be fast and efficient thanks to usage of B-trees.
// B-trees can create a lazy copy of the tree structure, minimizing overhead.
//
// Assuming the watch cache observes all events and snapshots cache after each of them,
// requests for a specific resource version can be served by retrieving
// the snapshot with the greatest RV less than or equal to the requested RV.
// To make snapshot retrivial efficient we need an ordered data structure, such as tree.
//
// The initial implementation uses a B-tree to achieve the following performance characteristics (n - number of snapshots stored):
//   - `Add`: Adds a new snapshot.
//     Complexity: O(log n).
//     Executed for each watch event observed by the cache.
//   - `GetLessOrEqual`: Retrieves the snapshot with the greatest RV less than or equal to the requested RV.
//     Complexity: O(log n).
//     Executed for each LIST request with match=Exact or continuation.
//   - `RemoveLess`: Cleans up snapshots outside the watch history window.
//     Complexity: O(k log n), k - number of snapshots to remove, usually only one if watch capacity was not reduced.
//     Executed per watch event observed when the cache is full.
//   - `Reset`: Cleans up all snapshots.
//     Complexity: O(1).
//     Executed when the watch cache is reinitialized.
//
// Further optimization is possible by leveraging the property that adds always
// increase the maximum RV and deletes only increase the minimum RV.
// For example, a binary search on a cyclic buffer of (RV, snapshot)
// should reduce number of allocations and improve removal complexity.
// However, this solution is more complex and is deferred for future implementation.
//
// TODO: Rewrite to use a cyclic buffer
func newSnapshotter() snapshotter {
	return snapshotter{
		snapshots: btree.New(btreeDegree, func(a, b rvSnapshot) bool {
			return a.resourceVersion < b.resourceVersion
		}),
	}
}

type snapshotter struct {
	snapshots *btree.BTree[rvSnapshot]
}

type rvSnapshot struct {
	resourceVersion uint64
	snapshot        Snapshot
}

func (s *snapshotter) Reset() {
	s.snapshots.Clear(false)
}

func (s *snapshotter) GetLessOrEqual(rv uint64) (Snapshot, bool) {
	var result *rvSnapshot
	s.snapshots.DescendLessOrEqual(rvSnapshot{resourceVersion: rv}, func(rvs rvSnapshot) bool {
		result = &rvs
		return false
	})
	if result == nil {
		return nil, false
	}
	return result.snapshot, true
}

func (s *snapshotter) Add(rv uint64, snapshot Snapshot) {
	s.snapshots.ReplaceOrInsert(rvSnapshot{resourceVersion: rv, snapshot: snapshot})
}

func (s *snapshotter) RemoveLess(rv uint64) {
	for s.snapshots.Len() > 0 {
		oldest, ok := s.snapshots.Min()
		if !ok {
			break
		}
		if rv <= oldest.resourceVersion {
			break
		}
		s.snapshots.DeleteMin()
	}
}

func (s *snapshotter) Len() int {
	return s.snapshots.Len()
}
