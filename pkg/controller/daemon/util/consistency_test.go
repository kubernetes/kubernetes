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

package util

import (
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
)

func TestHighWaterResourceVersion_New(t *testing.T) {
	rv := "123"
	h := newHighWaterResourceVersion(rv)
	require.NotNil(t, h)
	assert.Equal(t, rv, *h.version.Load())
	assert.Equal(t, rv, h.String())
}

func TestHighWaterResourceVersion_RaiseTo(t *testing.T) {
	t.Parallel()
	h := newHighWaterResourceVersion("10")

	// Raise to higher
	h.RaiseTo("20")
	assert.Equal(t, "20", h.String())

	// Raise to lower (should not change)
	h.RaiseTo("10")
	assert.Equal(t, "20", h.String())

	// Raise to same (should not change)
	h.RaiseTo("20")
	assert.Equal(t, "20", h.String())

	// Raise to higher again
	h.RaiseTo("100")
	assert.Equal(t, "100", h.String())

	// Raise with invalid RV (should not change)
	h.RaiseTo("abc")
	assert.Equal(t, "100", h.String())

	// Raise from invalid RV
	hInvalid := newHighWaterResourceVersion("xyz")
	hInvalid.RaiseTo("10")
	assert.Equal(t, "10", hInvalid.String())
}

func TestHighWaterResourceVersion_RaiseTo_Concurrent(t *testing.T) {
	t.Parallel()
	h := newHighWaterResourceVersion("0")
	wg := sync.WaitGroup{}
	numGoroutines := 100
	numIterations := 50

	for i := range numGoroutines {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			for j := range numIterations {
				// Each goroutine writes its own set of numbers
				rv := strconv.Itoa(start*numIterations + j + 1)
				h.RaiseTo(rv)
			}
		}(i)
	}

	wg.Wait()
	// The highest version should be numGoroutines * numIterations
	expectedMax := strconv.Itoa(numGoroutines * numIterations)
	assert.Equal(t, expectedMax, h.String())
}

func TestHighWaterResourceVersion_CompareTo(t *testing.T) {
	t.Parallel()
	h10 := newHighWaterResourceVersion("10")
	h20 := newHighWaterResourceVersion("20")

	// 10 < 20
	i, err := h10.CompareTo("20")
	require.NoError(t, err)
	assert.Equal(t, -1, i)

	// 20 > 10
	i, err = h20.CompareTo("10")
	require.NoError(t, err)
	assert.Equal(t, 1, i)

	// 10 == 10
	i, err = h10.CompareTo("10")
	require.NoError(t, err)
	assert.Equal(t, 0, i)
}

func TestResourceVersions_GetOrCreate(t *testing.T) {
	t.Parallel()
	rvs := newResourceVersions()
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	initialRV := "10"

	// Create new
	h1 := rvs.getOrCreate(gr, initialRV)
	require.NotNil(t, h1)
	assert.Equal(t, initialRV, h1.String())
	assert.Same(t, h1, rvs.versions[gr], "Should be stored in the map")

	// Get existing
	h2 := rvs.getOrCreate(gr, "20") // "20" should be ignored
	assert.Same(t, h1, h2, "Should return the existing record")
	assert.Equal(t, initialRV, h1.String(), "Initial RV should not be overwritten by getOrCreate")
}

func TestResourceVersions_GetOrCreate_Concurrent(t *testing.T) {
	t.Parallel()
	rvs := newResourceVersions()
	gr := schema.GroupResource{Group: "apps", Resource: "deployments"}
	initialRV := "10"
	wg := sync.WaitGroup{}
	numGoroutines := 100

	var firstRecord *highWaterResourceVersion
	var once sync.Once

	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			h := rvs.getOrCreate(gr, initialRV)

			once.Do(func() {
				firstRecord = h
			})

			assert.Same(t, firstRecord, h, "All goroutines should get the same record instance")
		}()
	}

	wg.Wait()
	require.NotNil(t, firstRecord)
	assert.Equal(t, initialRV, firstRecord.String())
	assert.Len(t, rvs.versions, 1, "Only one record should be created")
}

func TestOwnerRecord_WroteAt(t *testing.T) {
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	assert.Equal(t, uid, or.ownerUID)
	require.NotNil(t, or.versions)

	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	grDs := schema.GroupResource{Group: "apps", Resource: "daemonsets"}

	// First write
	or.WroteAt(grPod, "5")
	hPod := or.versions.get(grPod)
	require.NotNil(t, hPod)
	assert.Equal(t, "5", hPod.String())

	// Second write (higher)
	or.WroteAt(grPod, "10")
	assert.Equal(t, "10", hPod.String())

	// Third write (lower)
	or.WroteAt(grPod, "8")
	assert.Equal(t, "10", hPod.String())

	// Write to different resource
	or.WroteAt(grDs, "1")
	hSvc := or.versions.get(grDs)
	require.NotNil(t, hSvc)
	assert.Equal(t, "1", hSvc.String())
	assert.Equal(t, "10", hPod.String(), "Pod version should be unchanged")
}

func TestOwnerRecord_IsReady(t *testing.T) {
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	grDs := schema.GroupResource{Group: "apps", Resource: "daemonsets"}
	resourceStores := map[schema.GroupResource]cache.Store{
		grPod: cache.NewStore(cache.MetaNamespaceKeyFunc),
		grDs:  cache.NewStore(cache.MetaNamespaceKeyFunc),
	}

	store := NewConsistencyStore(resourceStores)

	// Case 1: No writes. Should be ready.
	require.NoError(t, or.EnsureReady(store), "Should be ready if no writes are recorded")

	// Add a write
	or.WroteAt(grPod, "10")

	// Case 2: Write exists, but no read. Not ready.
	require.Error(t, or.EnsureReady(store), "Not ready if write exists but no read")

	// Add a read, but it's lower
	resourceStores[grPod].Bookmark("5")

	// Case 3: Write exists, read is lower. Not ready.
	require.Error(t, or.EnsureReady(store), "Not ready if read < write")

	// Add a read, equal
	resourceStores[grPod].Bookmark("10")

	// Case 4: Write exists, read is equal. Ready.
	require.NoError(t, or.EnsureReady(store), "Ready if read == write")

	// Add a read, higher
	resourceStores[grPod].Bookmark("15")

	// Case 5: Write exists, read is higher. Ready.
	require.NoError(t, or.EnsureReady(store), "Ready if read > write")

	// Add a second write
	or.WroteAt(grDs, "100")

	// Case 6: One resource ready, one not. Not ready.
	require.Error(t, or.EnsureReady(store), "Not ready if one of multiple writes is not ready (no read)")

	// Make the second one ready
	resourceStores[grDs].Bookmark("100")

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
	assert.Empty(t, r3.versions.versions, "New record should be empty")

	// Check that old record is detached
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	r1.WroteAt(grPod, "10") // Write to old record
	assert.Empty(t, r3.versions.versions, "Write to old record should not affect new record")
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

	h := record.versions.get(grPod)
	require.NotNil(t, h)
	assert.Equal(t, "10", h.String())

	// Write again
	store.WroteAt(owner, uid1, grPod, "20")
	assert.Equal(t, "20", h.String())
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
	resourceStores := map[schema.GroupResource]cache.Store{
		grPod: cache.NewStore(cache.MetaNamespaceKeyFunc),
	}

	store := NewConsistencyStore(resourceStores)

	// Case 1: No record. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if no record exists")

	// Add a write
	store.WroteAt(owner1, uid1, grPod, "10")

	// Case 2: Record exists, but no read. Not ready.
	require.Error(t, store.EnsureReady(owner1), "Not ready if write exists but no read")

	// Add a read, but lower
	resourceStores[grPod].Bookmark("5")

	// Case 3: Record exists, read < write. Not ready.
	require.Error(t, store.EnsureReady(owner1), "Not ready if read < write")

	// Add read, equal
	resourceStores[grPod].Bookmark("10")

	// Case 4: Record exists, read == write. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if read == write")

	// Add read, higher
	resourceStores[grPod].Bookmark("15")

	// Case 5: Record exists, read > write. Ready.
	require.NoError(t, store.EnsureReady(owner1), "Ready if read > write")

	// Assert that the record no longer exists, we no longer need to track the
	// reads as long as the read has been higher than the latest write.
	assert.Nil(t, store.getWrittenRecord(owner1), "Written record should no longer exist")
}

func (r *resourceVersions) get(resource schema.GroupResource) *highWaterResourceVersion {
	r.versionsLock.RLock()
	defer r.versionsLock.RUnlock()
	return r.versions[resource]
}
