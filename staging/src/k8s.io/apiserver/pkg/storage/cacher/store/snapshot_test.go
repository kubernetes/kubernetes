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

package store

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSnapshotListPrefix(t *testing.T) {
	// Elements are deliberately unordered; snapshots must return keys in order.
	elements := []*Element{
		testStorageElement("/pods/ns1/b", "b", 2),
		testStorageElement("/pods/ns2/d", "d", 4),
		testStorageElement("/pods/ns1/a", "a", 1),
		testStorageElement("/pods/ns1/c", "c", 3),
	}
	// orderedListSnapshot is excluded: it serves a pre-computed range and
	// ignores prefix and continueKey by contract. Prefixes are "/"-terminated
	// as the cacher produces them; the implementations differ on other
	// prefixes (strings.HasPrefix in the btree, path segments in
	// listSnapshot).
	snapshots := []struct {
		name        string
		newSnapshot func(t *testing.T) Snapshot
	}{
		{
			name: "Indexer",
			newSnapshot: func(t *testing.T) Snapshot {
				indexer := newThreadedBtreeStoreIndexer(nil, btreeDegree)
				for _, elem := range elements {
					require.NoError(t, indexer.Add(elem))
				}
				return indexer.Clone()
			},
		},
		{
			name: "btreeStore",
			newSnapshot: func(t *testing.T) Snapshot {
				store := newBtreeStore(btreeDegree)
				for _, elem := range elements {
					require.NoError(t, store.Add(elem))
				}
				return &store
			},
		},
		{
			name: "listSnapshot",
			newSnapshot: func(t *testing.T) Snapshot {
				items := make([]interface{}, 0, len(elements))
				for _, elem := range elements {
					items = append(items, elem)
				}
				return listSnapshot{Items: items}
			},
		},
	}
	testCases := []struct {
		name        string
		prefix      string
		continueKey string
		expectKeys  []string
	}{
		{name: "whole range", prefix: "/pods/", expectKeys: []string{"/pods/ns1/a", "/pods/ns1/b", "/pods/ns1/c", "/pods/ns2/d"}},
		{name: "namespace prefix", prefix: "/pods/ns1/", expectKeys: []string{"/pods/ns1/a", "/pods/ns1/b", "/pods/ns1/c"}},
		{name: "continue token", prefix: "/pods/ns1/", continueKey: "/pods/ns1/a\x00", expectKeys: []string{"/pods/ns1/b", "/pods/ns1/c"}},
		// A continue token is the inclusive lower bound of the next page:
		// the apiserver encodes it as lastReturnedKey+"\x00" (the successor
		// string, which no real key equals) and etcd ranges start inclusively
		// at the given key, so an element whose key equals continueKey is
		// returned.
		{name: "continue from existing key", prefix: "/pods/ns1/", continueKey: "/pods/ns1/b", expectKeys: []string{"/pods/ns1/b", "/pods/ns1/c"}},
		{name: "continue past last key", prefix: "/pods/ns1/", continueKey: "/pods/ns1/c\x00", expectKeys: nil},
		{name: "no match", prefix: "/pods/ns3/", expectKeys: nil},
	}
	for _, s := range snapshots {
		t.Run(s.name, func(t *testing.T) {
			snapshot := s.newSnapshot(t)
			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					items, err := snapshot.OrderedListPrefix(tc.prefix, tc.continueKey)
					require.NoError(t, err)
					var listed []string
					for _, item := range items {
						listed = append(listed, item.(*Element).Key)
					}
					assert.Equal(t, tc.expectKeys, listed, "OrderedListPrefix")

					var ranged []string
					for elem, err := range snapshot.RangePrefix(tc.prefix, tc.continueKey) {
						require.NoError(t, err)
						ranged = append(ranged, elem.Key)
					}
					assert.Equal(t, tc.expectKeys, ranged, "RangePrefix")

					assert.Equal(t, len(tc.expectKeys), snapshot.Count(tc.prefix, tc.continueKey), "Count")
				})
			}
		})
	}
}
