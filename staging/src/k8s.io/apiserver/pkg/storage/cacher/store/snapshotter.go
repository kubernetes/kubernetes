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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher/consistency"
	"k8s.io/klog/v2"
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
//   - `HasSnapshot`: Checks whether the requested RV is not older than the oldest stored snapshot.
//     Complexity: O(log n).
//     Executed before watch cache synchronization for each LIST request with match=Exact or continuation.
//   - `GetSnapshot`: Retrieves the snapshot with the greatest RV less than or equal to the requested RV.
//     Complexity: O(log n).
//     Executed after watch cache synchronization for each LIST request with match=Exact or continuation.
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
func newSnapshotter(enabled bool) snapshotter {
	return snapshotter{
		snapshots: btree.New(btreeDegree, func(a, b *btreeStore) bool {
			return a.resourceVersion < b.resourceVersion
		}),
		enabled: enabled,
	}
}

type snapshotter struct {
	snapshots       *btree.BTree[*btreeStore]
	resourceVersion uint64
	enabled         bool
}

func (s *snapshotter) Enabled() bool {
	return s.enabled
}

func (s *snapshotter) SetEnabled(enabled bool) {
	s.enabled = enabled
	s.reset()
}

func (s *snapshotter) reset() {
	s.snapshots.Clear(false)
	s.resourceVersion = 0
}

func (s *snapshotter) HasSnapshot(rv uint64) bool {
	if !s.enabled {
		return false
	}
	oldest, ok := s.snapshots.Min()
	if !ok {
		return false
	}
	return rv >= oldest.resourceVersion
}

func (s *snapshotter) GetSnapshot(rv uint64) (*btreeStore, error) {
	if !s.enabled || s.resourceVersion == 0 {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", rv))
	}
	if rv > s.resourceVersion {
		if consistency.PanicOnCacheInconsistency {
			panic(fmt.Sprintf("snapshotter (on %d) got future resourceVersion (%d) that it doesn't properly handle as it depends on caller ensuring consistency", s.resourceVersion, rv))
		}
		klog.ErrorS(nil, "Snapshotter got future resourceVersion that it doesn't properly handle as it depends on caller ensuring consistency", "requestResourceVersion", rv, "currentResourceVersion", s.resourceVersion)
		return nil, storage.NewTooLargeResourceVersionError(rv, s.resourceVersion, 0)
	}
	var result *btreeStore
	s.snapshots.DescendLessOrEqual(&btreeStore{resourceVersion: rv}, func(snap *btreeStore) bool {
		result = snap
		return false
	})
	if result == nil {
		return nil, errors.NewResourceExpired(fmt.Sprintf("too old resource version: %d", rv))
	}
	return result, nil
}

func (s *snapshotter) Replace(snapshot *btreeStore) {
	if !s.enabled {
		return
	}
	s.reset()
	s.Add(snapshot)
}

func (s *snapshotter) Add(snapshot *btreeStore) {
	if !s.enabled {
		return
	}
	if snapshot.resourceVersion <= s.resourceVersion {
		if consistency.PanicOnCacheInconsistency {
			panic(fmt.Sprintf("snapshot resourceVersion (%d) must be greater than current resourceVersion (%d)", snapshot.resourceVersion, s.resourceVersion))
		}
		klog.ErrorS(nil, "Snapshot resourceVersion must be greater than current resourceVersion", "snapshotResourceVersion", snapshot.resourceVersion, "currentResourceVersion", s.resourceVersion)
	}
	s.snapshots.ReplaceOrInsert(snapshot)
	s.resourceVersion = max(s.resourceVersion, snapshot.resourceVersion)
}

func (s *snapshotter) UpdateResourceVersion(rv uint64) {
	if !s.enabled {
		return
	}
	if rv < s.resourceVersion {
		if consistency.PanicOnCacheInconsistency {
			panic(fmt.Sprintf("updated resourceVersion (%d) must be greater than or equal to current resourceVersion (%d)", rv, s.resourceVersion))
		}
		klog.ErrorS(nil, "Updated resourceVersion must be greater than or equal to current resourceVersion", "updatedResourceVersion", rv, "currentResourceVersion", s.resourceVersion)
	}
	s.resourceVersion = max(s.resourceVersion, rv)
}

func (s *snapshotter) RemoveLess(rv uint64) {
	if !s.enabled {
		return
	}
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
