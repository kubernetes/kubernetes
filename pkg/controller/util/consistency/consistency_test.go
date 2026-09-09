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

package consistency

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
)

func TestOwnerRecord_WroteAt(t *testing.T) {
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	assert.Equal(t, uid, or.ownerUID)
	require.NotNil(t, or.versions)

	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	grDs := schema.GroupResource{Group: "apps", Resource: "daemonsets"}

	// First write
	or.WroteAt(grPod, "5")
	assert.Equal(t, "5", or.versions[grPod])

	// Second write (higher)
	or.WroteAt(grPod, "10")
	assert.Equal(t, "10", or.versions[grPod])

	// Third write (lower)
	or.WroteAt(grPod, "8")
	assert.Equal(t, "10", or.versions[grPod])

	// Write to different resource
	or.WroteAt(grDs, "1")
	assert.Equal(t, "1", or.versions[grDs])
	assert.Equal(t, "10", or.versions[grPod], "Pod version should be unchanged")
}

func TestOwnerRecord_IsReady(t *testing.T) {
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	grDs := schema.GroupResource{Group: "apps", Resource: "daemonsets"}
	podStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	dsStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	resourceStores := map[schema.GroupResource]LastSyncRVGetter{
		grPod: podStore,
		grDs:  dsStore,
	}

	store := NewConsistencyStore(resourceStores)

	// Case 1: No writes. Should be ready.
	require.NoError(t, or.EnsureReady(store), "Should be ready if no writes are recorded")

	// Add a write
	or.WroteAt(grPod, "10")

	// Case 2: Write exists, but no reads. Should stay ready.
	require.NoError(t, or.EnsureReady(store), "Should stay ready if write exists and no read")

	// Add a read, but it's lower
	podStore.Bookmark("5")

	// Case 3: Write exists, read is lower. Not ready.
	require.Error(t, or.EnsureReady(store), "Not ready if read < write")

	// Add a read, equal
	podStore.Bookmark("10")

	// Case 4: Write exists, read is equal. Ready.
	require.NoError(t, or.EnsureReady(store), "Ready if read == write")

	// Add a read, higher
	podStore.Bookmark("15")

	// Case 5: Write exists, read is higher. Ready.
	require.NoError(t, or.EnsureReady(store), "Ready if read > write")

	// Add a second write and read
	or.WroteAt(grDs, "100")
	dsStore.Bookmark("50")

	// Case 6: One resource ready, one not. Not ready.
	require.Error(t, or.EnsureReady(store), "Not ready if one of multiple writes is not ready (no read)")

	// Make the second one ready
	dsStore.Bookmark("100")

	// Case 7: All resources ready. Ready.
	require.NoError(t, or.EnsureReady(store), "Ready if all writes are ready")
}

func TestConsistencyStore_New(t *testing.T) {
	store := NewConsistencyStore(nil)
	require.NotNil(t, store)
	require.NotNil(t, store.writes)
	assert.Empty(t, store.writes)
}

func TestConsistencyStore_EnsureWrittenRecord(t *testing.T) {
	store := NewConsistencyStore(nil)
	owner := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	uid2 := types.UID("uid-2")

	// Create new
	r1 := store.ensureWrittenRecord(owner, uid1)
	require.NotNil(t, r1)
	assert.Equal(t, uid1, r1.ownerUID)
	assert.Same(t, r1, store.writes[owner])

	// Get existing with same UID
	r2 := store.ensureWrittenRecord(owner, uid1)
	assert.Same(t, r1, r2, "Should return existing record for same UID")

	// Get existing with different UID (should replace)
	r3 := store.ensureWrittenRecord(owner, uid2)
	require.NotNil(t, r3)
	assert.NotSame(t, r1, r3, "Should be a new record")
	assert.Equal(t, uid2, r3.ownerUID)
	assert.Same(t, r3, store.writes[owner], "New record should replace old one in map")
	assert.Empty(t, r3.versions, "New record should be empty")

	// Check that old record is detached
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	r1.WroteAt(grPod, "10") // Write to old record
	assert.Empty(t, r3.versions, "Write to old record should not affect new record")
}

func TestConsistencyStore_EnsureWrittenRecord_Concurrent(t *testing.T) {
	store := NewConsistencyStore(nil)
	owner := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	uid2 := types.UID("uid-2")

	wg := sync.WaitGroup{}
	numGoroutines := 50

	// Concurrent creation with same UID
	var firstRecord *ownerRecord
	var once sync.Once
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := store.ensureWrittenRecord(owner, uid1)
			assert.Equal(t, uid1, r.ownerUID)
			once.Do(func() {
				firstRecord = r
			})
			assert.Same(t, firstRecord, r)
		}()
	}
	wg.Wait()
	require.NotNil(t, firstRecord)
	assert.Len(t, store.writes, 1)

	// Concurrent replacement with new UID
	var replacementRecord *ownerRecord
	var replaceOnce sync.Once
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := store.ensureWrittenRecord(owner, uid2)
			assert.Equal(t, uid2, r.ownerUID)
			replaceOnce.Do(func() {
				replacementRecord = r
			})
			assert.Same(t, replacementRecord, r)
		}()
	}
	wg.Wait()
	require.NotNil(t, replacementRecord)
	assert.Len(t, store.writes, 1)
	assert.Same(t, replacementRecord, store.writes[owner])
	assert.NotSame(t, firstRecord, replacementRecord)
}

func TestConsistencyStore_WroteAt(t *testing.T) {
	store := NewConsistencyStore(nil)
	owner := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	grPod := schema.GroupResource{Group: "", Resource: "pods"}

	store.WroteAt(owner, uid1, grPod, "10")

	record := store.getWrittenRecord(owner)
	require.NotNil(t, record)
	assert.Equal(t, uid1, record.ownerUID)

	assert.Equal(t, "10", record.versions[grPod])

	// Write again
	store.WroteAt(owner, uid1, grPod, "20")
	assert.Equal(t, "20", record.versions[grPod])
}

func TestConsistencyStore_Clear(t *testing.T) {
	store := NewConsistencyStore(nil)
	owner1 := types.NamespacedName{Name: "owner1"}
	owner2 := types.NamespacedName{Name: "owner2"}
	uid1 := types.UID("uid-1")
	uid2 := types.UID("uid-2")

	// Setup
	r1 := store.ensureWrittenRecord(owner1, uid1)
	r2 := store.ensureWrittenRecord(owner2, uid2)
	require.Len(t, store.writes, 2)

	// Clear non-existent
	store.Clear(types.NamespacedName{Name: "non-existent"}, uid1)
	assert.Len(t, store.writes, 2)

	// Clear with wrong UID
	store.Clear(owner1, uid2)
	assert.Len(t, store.writes, 2, "Should not clear with wrong UID")
	assert.Same(t, r1, store.writes[owner1])

	// Clear with correct UID
	store.Clear(owner1, uid1)
	assert.Len(t, store.writes, 1, "Should clear with correct UID")
	assert.Nil(t, store.writes[owner1])
	assert.Same(t, r2, store.writes[owner2], "Other record should remain")

	// Re-add r1
	store.ensureWrittenRecord(owner1, uid1)
	require.Len(t, store.writes, 2)

	// Clear with empty UID
	store.Clear(owner1, "")
	assert.Len(t, store.writes, 1, "Should clear with empty UID")
	assert.Nil(t, store.writes[owner1])
	assert.Same(t, r2, store.writes[owner2])
}

func TestConsistencyStore_IsReady(t *testing.T) {
	owner1 := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	podStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	resourceStores := map[schema.GroupResource]LastSyncRVGetter{
		grPod: podStore,
	}

	store := NewConsistencyStore(resourceStores)

	// Case 1: No record. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if no record exists")

	// Add a write and initial read rv
	podStore.Bookmark("5")
	store.WroteAt(owner1, uid1, grPod, "10")

	// Case 2: Record exists, read < write. Not ready.
	require.Error(t, store.EnsureReady(owner1), "Not ready if read < write")

	// Add read, equal
	podStore.Bookmark("10")

	// Case 3: Record exists, read == write. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if read == write")

	// Add read, higher
	podStore.Bookmark("15")

	// Case 4: Record exists, read > write. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if read > write")

	// Assert that the record no longer exists, we no longer need to track the
	// reads as long as the read has been higher than the latest write.
	assert.Nil(t, store.getWrittenRecord(owner1), "Written record should no longer exist")
}
