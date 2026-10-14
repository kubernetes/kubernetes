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
	"errors"
	"sync"
	"testing"

	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/ktesting"
)

func TestOwnerRecord_WroteAt(t *testing.T) {
	tCtx := ktesting.Init(t)
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	tCtx.Assert(or.ownerUID).Should(gomega.Equal(uid))
	tCtx.Require(or.versions).ShouldNot(gomega.BeNil())

	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	grDs := schema.GroupResource{Group: "apps", Resource: "daemonsets"}

	// Empty or malformed initial write is ignored
	or.WroteAt(grPod, "")
	tCtx.Assert(or.versions).Should(gomega.BeEmpty())
	or.WroteAt(grPod, "invalid")
	tCtx.Assert(or.versions).Should(gomega.BeEmpty())

	// First write
	or.WroteAt(grPod, "5")
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("5"))

	// Second write (higher)
	or.WroteAt(grPod, "10")
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("10"))

	// Third write (lower)
	or.WroteAt(grPod, "8")
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("10"))

	// Empty or malformed write does not overwrite existing valid RV
	or.WroteAt(grPod, "")
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("10"))
	or.WroteAt(grPod, "invalid")
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("10"))

	// Write to different resource
	or.WroteAt(grDs, "1")
	tCtx.Assert(or.versions[grDs]).Should(gomega.Equal("1"))
	tCtx.Assert(or.versions[grPod]).Should(gomega.Equal("10"), "Pod version should be unchanged")
}

func TestOwnerRecord_IsReady(t *testing.T) {
	tCtx := ktesting.Init(t)
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

	store := newConsistencyStore(resourceStores)

	// Case 1: No writes. Should be ready.
	tCtx.ExpectNoError(or.EnsureReady(store), "Should be ready if no writes are recorded")

	// Add a write
	or.WroteAt(grPod, "10")

	// Case 2: Write exists, but no reads observed yet (readRV == ""). Not ready (fails in safe direction).
	var consistencyErr *ConsistencyError
	err := or.EnsureReady(store)
	tCtx.Require(errors.As(err, &consistencyErr)).Should(gomega.BeTrueBecause("error must match target type"), "Not ready if write exists and no read observed")
	tCtx.Assert(consistencyErr.ReadRV).Should(gomega.BeEmpty())
	tCtx.Assert(consistencyErr.WroteRV).Should(gomega.Equal("10"))
	tCtx.Assert(consistencyErr.GroupResource).Should(gomega.Equal(grPod))

	// Add a read, but it's lower
	podStore.Bookmark("5")

	// Case 3: Write exists, read is lower. Not ready.
	tCtx.Require(or.EnsureReady(store)).Should(gomega.HaveOccurred(), "Not ready if read < write")

	// Add a read, equal
	podStore.Bookmark("10")

	// Case 4: Write exists, read is equal. Ready.
	tCtx.ExpectNoError(or.EnsureReady(store), "Ready if read == write")

	// Add a read, higher
	podStore.Bookmark("15")

	// Case 5: Write exists, read is higher. Ready.
	tCtx.ExpectNoError(or.EnsureReady(store), "Ready if read > write")

	// Add a second write and read
	or.WroteAt(grDs, "100")
	dsStore.Bookmark("50")

	// Case 6: One resource ready, one not. Not ready.
	tCtx.Require(or.EnsureReady(store)).Should(gomega.HaveOccurred(), "Not ready if one of multiple writes is not ready")

	// Make the second one ready
	dsStore.Bookmark("100")

	// Case 7: All resources ready. Ready.
	tCtx.ExpectNoError(or.EnsureReady(store), "Ready if all writes are ready")
}

func TestOwnerRecord_EnsureReady_MissingOrNilStore(t *testing.T) {
	tCtx := ktesting.Init(t)
	uid := types.UID("owner-uid-1")
	or := newOwnerRecord(uid)
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	or.WroteAt(grPod, "10")

	// Missing store in map
	storeWithoutPod := newConsistencyStore(map[schema.GroupResource]LastSyncRVGetter{})
	err := or.EnsureReady(storeWithoutPod)
	tCtx.Require(err).Should(gomega.MatchError(gomega.ContainSubstring("no store registered for group resource pods")))

	// Nil store in map
	storeWithNilPod := newConsistencyStore(map[schema.GroupResource]LastSyncRVGetter{
		grPod: nil,
	})
	err = or.EnsureReady(storeWithNilPod)
	tCtx.Require(err).Should(gomega.MatchError(gomega.ContainSubstring("no store registered for group resource pods")))
}

func TestConsistencyError_Error(t *testing.T) {
	tCtx := ktesting.Init(t)
	gr := schema.GroupResource{Group: "apps", Resource: "statefulsets"}
	errUnsynced := &ConsistencyError{
		WroteRV:       "10",
		ReadRV:        "",
		GroupResource: gr,
	}
	tCtx.Assert(errUnsynced.Error()).Should(gomega.Equal("read version is not synced (written version: 10) for group resource statefulsets.apps"))

	errStale := &ConsistencyError{
		WroteRV:       "10",
		ReadRV:        "5",
		GroupResource: gr,
	}
	tCtx.Assert(errStale.Error()).Should(gomega.Equal("read version: 5 is not as new as written version: 10 for group resource statefulsets.apps"))
}

func TestConsistencyStore_New(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := NewConsistencyStore(nil)
	tCtx.Require(store).ShouldNot(gomega.BeNil())
	internalStore, ok := store.(*consistencyStore)
	tCtx.Require(ok).Should(gomega.BeTrueBecause("NewConsistencyStore must return a *consistencyStore"))
	tCtx.Require(internalStore.writes).ShouldNot(gomega.BeNil())
	tCtx.Assert(internalStore.writes).Should(gomega.BeEmpty())
}

func TestNoopConsistencyStore(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := NewNoopConsistencyStore()
	tCtx.Require(store).ShouldNot(gomega.BeNil())

	owner := types.NamespacedName{Namespace: "default", Name: "owner1"}
	uid := types.UID("uid-1")
	grPod := schema.GroupResource{Group: "", Resource: "pods"}

	store.WroteAt(owner, uid, grPod, "10")
	tCtx.ExpectNoError(store.EnsureReady(owner))
	store.Clear(owner, uid)
	tCtx.ExpectNoError(store.EnsureReady(owner))
}

func TestConsistencyStore_EnsureWrittenRecord(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := newConsistencyStore(nil)
	owner := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	uid2 := types.UID("uid-2")

	// Create new
	r1 := store.ensureWrittenRecord(owner, uid1)
	tCtx.Require(r1).ShouldNot(gomega.BeNil())
	tCtx.Assert(r1.ownerUID).Should(gomega.Equal(uid1))
	tCtx.Assert(store.writes[owner]).Should(gomega.BeIdenticalTo(r1))

	// Get existing with same UID
	r2 := store.ensureWrittenRecord(owner, uid1)
	tCtx.Assert(r2).Should(gomega.BeIdenticalTo(r1), "Should return existing record for same UID")

	// Get existing with different UID (should replace)
	r3 := store.ensureWrittenRecord(owner, uid2)
	tCtx.Require(r3).ShouldNot(gomega.BeNil())
	tCtx.Assert(r3).ShouldNot(gomega.BeIdenticalTo(r1), "Should be a new record")
	tCtx.Assert(r3.ownerUID).Should(gomega.Equal(uid2))
	tCtx.Assert(store.writes[owner]).Should(gomega.BeIdenticalTo(r3), "New record should replace old one in map")
	tCtx.Assert(r3.versions).Should(gomega.BeEmpty(), "New record should be empty")

	// Check that old record is detached
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	r1.WroteAt(grPod, "10") // Write to old record
	tCtx.Assert(r3.versions).Should(gomega.BeEmpty(), "Write to old record should not affect new record")
}

func TestConsistencyStore_EnsureWrittenRecord_Concurrent(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := newConsistencyStore(nil)
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
			tCtx.Assert(r.ownerUID).Should(gomega.Equal(uid1))
			once.Do(func() {
				firstRecord = r
			})
			tCtx.Assert(r).Should(gomega.BeIdenticalTo(firstRecord))
		}()
	}
	wg.Wait()
	tCtx.Require(firstRecord).ShouldNot(gomega.BeNil())
	tCtx.Assert(store.writes).Should(gomega.HaveLen(1))

	// Concurrent replacement with new UID
	var replacementRecord *ownerRecord
	var replaceOnce sync.Once
	for range numGoroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r := store.ensureWrittenRecord(owner, uid2)
			tCtx.Assert(r.ownerUID).Should(gomega.Equal(uid2))
			replaceOnce.Do(func() {
				replacementRecord = r
			})
			tCtx.Assert(r).Should(gomega.BeIdenticalTo(replacementRecord))
		}()
	}
	wg.Wait()
	tCtx.Require(replacementRecord).ShouldNot(gomega.BeNil())
	tCtx.Assert(store.writes).Should(gomega.HaveLen(1))
	tCtx.Assert(store.writes[owner]).Should(gomega.BeIdenticalTo(replacementRecord))
	tCtx.Assert(replacementRecord).ShouldNot(gomega.BeIdenticalTo(firstRecord))
}

func TestConsistencyStore_WroteAt(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := newConsistencyStore(nil)
	owner := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	grPod := schema.GroupResource{Group: "", Resource: "pods"}

	// Empty RV allocates a record but does not add any versions
	store.WroteAt(owner, uid1, grPod, "")
	tCtx.Assert(store.getWrittenRecord(owner).versions[grPod]).Should(gomega.BeEmpty())

	store.WroteAt(owner, uid1, grPod, "10")

	record := store.getWrittenRecord(owner)
	tCtx.Require(record).ShouldNot(gomega.BeNil())
	tCtx.Assert(record.ownerUID).Should(gomega.Equal(uid1))

	tCtx.Assert(record.versions[grPod]).Should(gomega.Equal("10"))

	// Write again
	store.WroteAt(owner, uid1, grPod, "20")
	tCtx.Assert(record.versions[grPod]).Should(gomega.Equal("20"))
}

func TestConsistencyStore_Clear(t *testing.T) {
	tCtx := ktesting.Init(t)
	store := newConsistencyStore(nil)
	owner1 := types.NamespacedName{Name: "owner1"}
	owner2 := types.NamespacedName{Name: "owner2"}
	uid1 := types.UID("uid-1")
	uid2 := types.UID("uid-2")

	// Setup
	r1 := store.ensureWrittenRecord(owner1, uid1)
	r2 := store.ensureWrittenRecord(owner2, uid2)
	tCtx.Require(store.writes).Should(gomega.HaveLen(2))

	// Clear non-existent
	store.Clear(types.NamespacedName{Name: "non-existent"}, uid1)
	tCtx.Assert(store.writes).Should(gomega.HaveLen(2))

	// Clear with wrong UID
	store.Clear(owner1, uid2)
	tCtx.Assert(store.writes).Should(gomega.HaveLen(2), "Should not clear with wrong UID")
	tCtx.Assert(store.writes[owner1]).Should(gomega.BeIdenticalTo(r1))

	// Clear with correct UID
	store.Clear(owner1, uid1)
	tCtx.Assert(store.writes).Should(gomega.HaveLen(1), "Should clear with correct UID")
	tCtx.Assert(store.writes[owner1]).Should(gomega.BeNil())
	tCtx.Assert(store.writes[owner2]).Should(gomega.BeIdenticalTo(r2), "Other record should remain")

	// Re-add r1
	store.ensureWrittenRecord(owner1, uid1)
	tCtx.Require(store.writes).Should(gomega.HaveLen(2))

	// Clear with empty UID
	store.Clear(owner1, "")
	tCtx.Assert(store.writes).Should(gomega.HaveLen(1), "Should clear with empty UID")
	tCtx.Assert(store.writes[owner1]).Should(gomega.BeNil())
	tCtx.Assert(store.writes[owner2]).Should(gomega.BeIdenticalTo(r2))
}

func TestConsistencyStore_IsReady(t *testing.T) {
	tCtx := ktesting.Init(t)
	owner1 := types.NamespacedName{Name: "owner1"}
	uid1 := types.UID("uid-1")
	grPod := schema.GroupResource{Group: "", Resource: "pods"}
	podStore := cache.NewStore(cache.MetaNamespaceKeyFunc)
	resourceStores := map[schema.GroupResource]LastSyncRVGetter{
		grPod: podStore,
	}

	store := newConsistencyStore(resourceStores)

	// Case 1: No record. Ready.
	tCtx.ExpectNoError(store.EnsureReady(owner1), "Ready if no record exists")

	// Add a write before any reads
	store.WroteAt(owner1, uid1, grPod, "10")

	// Case 2: Record exists, but store has not synced any reads yet. Not ready.
	tCtx.Require(store.EnsureReady(owner1)).Should(gomega.HaveOccurred(), "Not ready if readRV is empty")

	// Add read, but lower
	podStore.Bookmark("5")

	// Case 3: Record exists, read < write. Not ready.
	tCtx.Require(store.EnsureReady(owner1)).Should(gomega.HaveOccurred(), "Not ready if read < write")

	// Add read, equal
	podStore.Bookmark("10")

	// Case 4: Record exists, read == write. Ready.
	tCtx.ExpectNoError(store.EnsureReady(owner1), "Ready if read == write")

	// Add read, higher
	podStore.Bookmark("15")

	// Case 5: Record exists, read > write. Ready.
	tCtx.ExpectNoError(store.EnsureReady(owner1), "Ready if read > write")

	// Assert that the record no longer exists, we no longer need to track the
	// reads as long as the read has been higher than the latest write.
	tCtx.Assert(store.getWrittenRecord(owner1)).Should(gomega.BeNil(), "Written record should no longer exist")
}
