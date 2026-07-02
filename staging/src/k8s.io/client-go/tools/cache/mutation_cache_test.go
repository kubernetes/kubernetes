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
		"equal-rv-same-uid-clears": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "2",
			wantCleared: true,
		},
		"newer-store-same-uid-clears": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "3",
			wantCleared: true,
		},
		"older-store-same-uid-keeps": {
			mutationRV:  "2",
			storeUID:    "uid-1",
			storeRV:     "1",
			wantCleared: false,
		},
		"equal-rv-different-uid-keeps": {
			mutationRV:  "2",
			storeUID:    "uid-other",
			storeRV:     "2",
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

func TestMutationCacheOnDelete(t *testing.T) {
	tests := map[string]struct {
		deleteUID   types.UID
		deleteRV    string
		wantCleared bool
	}{
		"matching-uid-clears": {
			deleteUID:   "uid-1",
			deleteRV:    "2",
			wantCleared: true,
		},
		"tombstone-old-rv-still-clears": {
			// OnDelete does not compare resource versions: even a tombstone
			// object with an older RV than the mutation clears the entry.
			deleteUID:   "uid-1",
			deleteRV:    "1",
			wantCleared: true,
		},
		"different-uid-keeps": {
			deleteUID:   "uid-other",
			deleteRV:    "2",
			wantCleared: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			store := NewStore(MetaNamespaceKeyFunc)
			indexer := NewIndexer(MetaNamespaceKeyFunc, Indexers{})
			mc := NewIntegerResourceVersionMutationCache(klog.Background(), store, indexer, time.Minute, false)

			// Backing store has the pod at RV "1".
			require.NoError(t, store.Add(makeMutationTestPod("pod", "uid-1", "1")))

			// Mutation brings it to RV "2".
			mc.Mutation(makeMutationTestPod("pod", "uid-1", "2"))

			mc.OnDelete(makeMutationTestPod("pod", tc.deleteUID, tc.deleteRV))

			got, exists, err := mc.GetByKey("pod")
			require.NoError(t, err)
			require.True(t, exists)

			gotRV := got.(*v1.Pod).ResourceVersion
			if tc.wantCleared {
				assert.Equal(t, "1", gotRV, "backing store version expected after mutation cleared")
			} else {
				assert.Equal(t, "2", gotRV, "mutation version expected")
			}
		})
	}
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
