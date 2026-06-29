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

package cache

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

func makeMutationTestPod(name string, uid types.UID, rv string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			UID:             uid,
			ResourceVersion: rv,
		},
	}
}

func TestMutationCacheOnAddOrUpdate(t *testing.T) {
	tests := map[string]struct {
		mutationRV  string
		storeUID    types.UID // UID passed to OnAddOrUpdate
		storeRV     string    // RV passed to OnAddOrUpdate
		wantCleared bool
	}{
		"equal-rv-clears": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "2",
			wantCleared: true,
		},
		"newer-store-clears": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "3",
			wantCleared: true,
		},
		"older-store-keeps": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "1",
			wantCleared: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			store := NewStore(MetaNamespaceKeyFunc)
			indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{})
			mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Minute, false)

			mutated := makeMutationTestPod("pod", "uid-1", tc.mutationRV)
			mc.Mutation(mutated)

			mc.OnAddOrUpdate(makeMutationTestPod("pod", tc.storeUID, tc.storeRV))

			// Add the pod to the backing store at RV "1" (always older than the
			// mutation's RV "2"), so we can tell whether GetByKey returns the
			// mutation or the backing copy.
			require.NoError(t, store.Add(makeMutationTestPod("pod", "uid-1", "1")))

			got, exists, err := mc.GetByKey("pod")
			require.NoError(t, err)
			require.True(t, exists)

			gotRV := got.(*v1.Pod).ResourceVersion
			if tc.wantCleared {
				assert.Equal(t, "1", gotRV, "backing store version expected after mutation cleared")
			} else {
				assert.Equal(t, tc.mutationRV, gotRV, "mutation version expected")
			}
		})
	}
}

// TestMutationCacheOnDelete checks the behavior of OnDelete
// when invoked after Mutation.
func TestMutationCacheOnDelete(t *testing.T) {
	tests := map[string]struct {
		deleteUID  types.UID
		deleteRV   string
		wantExists bool
	}{
		"old-rv-same-UID": {
			deleteUID:  "uid-1",
			deleteRV:   "1", // Could be from a stale object in DeletedFinalStateUnknown.
			wantExists: false,
		},
		"new-rv-different-UID": {
			deleteUID:  "uid-2",
			deleteRV:   "3",
			wantExists: false,
		},
		"old-rv-different-uid": {
			deleteUID:  "uid-other",
			deleteRV:   "1",
			wantExists: true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			store := NewStore(MetaNamespaceKeyFunc)
			indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{})
			mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Minute, true)

			// Backing store has the pod at RV "2".
			require.NoError(t, store.Add(makeMutationTestPod("pod", "uid-1", "2")))

			// Mutation brings it to RV "3".
			mc.Mutation(makeMutationTestPod("pod", "uid-1", "3"))
			deletedPod := makeMutationTestPod("pod", tc.deleteUID, tc.deleteRV)
			require.NoError(t, store.Delete(deletedPod))
			mc.OnDelete(deletedPod)

			_, exists, err := mc.GetByKey("pod")
			require.NoError(t, err)
			require.Equal(t, tc.wantExists, exists)
		})
	}
}

// TestMutationCacheUpdateConcurrentDelete checks the behavior when
// Mutation is called after some informer events.
func TestMutationCacheUpdateConcurrentDelete(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{})
	mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Hour, true)

	// Backing store has the pod at RV "1".
	require.NoError(t, store.Add(makeMutationTestPod("pod", "uid-1", "1")))

	// Client starts an update leading to RV "2", which immediately gets
	// deleted by some other client. That deletion is received before the
	// client finishes its update.
	updatedPod := makeMutationTestPod("pod", "uid-1", "2")
	require.NoError(t, store.Update(updatedPod))
	require.NoError(t, store.Delete(updatedPod))

	// Informer events get delivered with a delay.
	mc.OnAddOrUpdate(updatedPod)
	mc.OnDelete(updatedPod)

	// This Mutation call is stale, which gets detected because
	// the mutation cache contains a tombstone object.
	mc.Mutation(updatedPod)

	_, exists, err := mc.GetByKey("pod")
	require.NoError(t, err)
	require.False(t, exists)
	require.Equal(t, []any{"pod"}, mc.(*mutationCache).mutationCache.Keys())
}

// TestMutationCacheUpdateConcurrentRecreate checks the behavior when
// Mutation is called after some informer events.
func TestMutationCacheUpdateConcurrentRecreate(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{})
	mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Hour, true)

	// Backing store has the pod at RV "1".
	require.NoError(t, store.Add(makeMutationTestPod("pod", "uid-1", "1")))

	// Client starts an update leading to RV "2", which immediately gets
	// replaced by some other pod using the same name. Those changes are
	// received before the client finishes its update.
	updatedPod := makeMutationTestPod("pod", "uid-1", "2")
	require.NoError(t, store.Update(updatedPod))
	require.NoError(t, store.Delete(updatedPod))
	replacementPod := makeMutationTestPod("pod", "uid-2", "3")
	require.NoError(t, store.Add(replacementPod))

	// Informer events get delivered with a delay.
	mc.OnAddOrUpdate(updatedPod)
	mc.OnDelete(updatedPod)
	mc.OnAddOrUpdate(replacementPod)

	// This Mutation call is stale, which gets detected because there is a more
	// recent object in the store.
	mc.Mutation(updatedPod)

	got, exists, err := mc.GetByKey("pod")
	require.NoError(t, err)
	require.True(t, exists)
	require.Equal(t, replacementPod, got)
	require.Empty(t, mc.(*mutationCache).mutationCache.Keys())
}

// TestMutationCacheOnDeleteClearsStaleMutation exercises the core bug that motivated
// the OnDelete method: a mutation stored after an update must not survive once the
// informer reports the object deleted.
//
// The scenario:
//  1. The object is added by the caller (includeAdds=true) so mutation cache
//     returns it even when the backing store is empty.
//  2. The object is deleted on the server; the backing store is now empty.
//  3. Without OnDelete the mutation would linger in the cache and ByIndex
//     would still return it, preventing the caller from recreating it.
func TestMutationCacheOnDeleteClearsStaleMutation(t *testing.T) {
	store := NewStore(MetaNamespaceKeyFunc)
	byNameIndex := "by-name"
	indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{
		byNameIndex: func(obj interface{}) ([]string, error) {
			return []string{obj.(*v1.Pod).Name}, nil
		},
	})
	mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Minute, true /* includeAdds */)

	// Simulate an update: mutation cache holds the pod at RV "2" while the
	// backing store is still empty (informer hasn't caught up yet).
	mc.Mutation(makeMutationTestPod("pod", "uid-1", "2"))

	items, err := mc.ByIndex(byNameIndex, "pod")
	require.NoError(t, err)
	assert.Len(t, items, 1, "mutation should be visible via ByIndex when backing store is empty")

	// Informer reports the delete: OnDelete must clear the stale mutation so
	// that the next ByIndex call returns nothing and lets the caller recreate
	// the object.
	mc.OnDelete(makeMutationTestPod("pod", "uid-1", "1"))

	items, err = mc.ByIndex(byNameIndex, "pod")
	require.NoError(t, err)
	assert.Empty(t, items, "OnDelete must clear the mutation; ByIndex must return nothing")
}
