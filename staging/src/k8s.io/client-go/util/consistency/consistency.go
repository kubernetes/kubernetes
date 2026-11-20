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

// Package consistency provides a store for tracking written resource versions and
// ensuring that local informer caches have observed resource versions at least as new
// as those written before reconciling owners.
//
// Tracking requires informer stores that implement LastSyncRVGetter (such as cache.Store
// when the AtomicFIFO feature gate is enabled, which has been the default since v1.36).
package consistency

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/resourceversion"
)

// ConsistencyStore allows tracking written resource versions for owned resources
// and verifying that local informer stores have caught up to those versions.
type ConsistencyStore interface {
	// WroteAt records a written RV for an owned resource.
	WroteAt(owner types.NamespacedName, ownerUID types.UID, resource schema.GroupResource, rv string)
	// Clear wipes the owner if the UID matches, if left empty it will wipe no
	// matter what the UID is.
	Clear(owner types.NamespacedName, ownerUID types.UID)
	// EnsureReady queries the ConsistencyStore to check whether or not the
	// stores records are up to date, returning an error if they are not.
	// Must not be called concurrently with WroteAt for the same owner.
	EnsureReady(owner types.NamespacedName) error
}

// ConsistencyError is an error type returned by EnsureReady with information
// about the resource versions and GroupResource that caused the error.
type ConsistencyError struct {
	ReadRV        string
	WroteRV       string
	GroupResource schema.GroupResource
}

// Error implements the error interface for a ConsistencyError.
func (c *ConsistencyError) Error() string {
	if c.ReadRV == "" {
		return fmt.Sprintf("read version is not synced (written version: %s) for group resource %s", c.WroteRV, c.GroupResource.String())
	}
	return fmt.Sprintf("read version: %s is not as new as written version: %s for group resource %s", c.ReadRV, c.WroteRV, c.GroupResource.String())
}

var _ ConsistencyStore = &consistencyStore{}

// LastSyncRVGetter is a minimal interface that provides the latest resource version observed by a store.
type LastSyncRVGetter interface {
	// LastStoreSyncResourceVersion returns the latest resource version that the store has seen.
	LastStoreSyncResourceVersion() string
}

// consistencyStore is a ConsistencyStore implementation that compares recorded
// write resource versions against cache stores.
type consistencyStore struct {
	// writesLock guards reads/additions/deletions to the writes map.
	// individual records are responsible for managing their own thread safety.
	writesLock sync.RWMutex
	// writes is a map of owner -> ownerRecord
	writes map[types.NamespacedName]*ownerRecord

	stores map[schema.GroupResource]LastSyncRVGetter
}

// NewConsistencyStore creates a new ConsistencyStore configured with stores
// implementing LastSyncRVGetter for each tracked GroupResource.
//
// Note: This requires the AtomicFIFO feature gate to be enabled in client-go (default
// since v1.36) so that stores can track and report their latest synced resource version via
// LastStoreSyncResourceVersion. If AtomicFIFO is disabled, or if a store has not observed
// any sync yet, EnsureReady will return a ConsistencyError.
func NewConsistencyStore(stores map[schema.GroupResource]LastSyncRVGetter) ConsistencyStore {
	return newConsistencyStore(stores)
}

func newConsistencyStore(stores map[schema.GroupResource]LastSyncRVGetter) *consistencyStore {
	return &consistencyStore{
		writes: map[types.NamespacedName]*ownerRecord{},
		stores: stores,
	}
}

// getWrittenRecord returns the record for the given owner, or nil if no record exists.
func (c *consistencyStore) getWrittenRecord(owner types.NamespacedName) *ownerRecord {
	c.writesLock.RLock()
	defer c.writesLock.RUnlock()
	return c.writes[owner]
}

// ensureWrittenRecord returns a ownerRecord for the given owner and ownerUID.
// If there is no current record, one is created.
// If there is a current record with a different ownerUID, it is replaced with an empty record for the specified ownerUID.
func (c *consistencyStore) ensureWrittenRecord(owner types.NamespacedName, ownerUID types.UID) *ownerRecord {
	// fast path, already exists
	if record := c.getWrittenRecord(owner); record != nil && record.ownerUID == ownerUID {
		return record
	}

	// slow path, init
	c.writesLock.Lock()
	defer c.writesLock.Unlock()
	// check again after write lock
	if record := c.writes[owner]; record != nil && record.ownerUID == ownerUID {
		return record
	}
	// initialize to the given uid
	record := newOwnerRecord(ownerUID)
	c.writes[owner] = record
	return record
}

// WroteAt writes the latest written RV if it is greater than the currently
// written RV for the owner.
func (c *consistencyStore) WroteAt(owner types.NamespacedName, ownerUID types.UID, resource schema.GroupResource, rv string) {
	c.ensureWrittenRecord(owner, ownerUID).WroteAt(resource, rv)
}

// Clear deletes the record for owner if it exists and matches the specified
// ownerUID (or the specified ownerUID is empty)
func (c *consistencyStore) Clear(owner types.NamespacedName, ownerUID types.UID) {
	// deleted owners typically have an existing record, not worth checking the fast path for missing records
	c.writesLock.Lock()
	defer c.writesLock.Unlock()
	if record := c.writes[owner]; record != nil && (len(ownerUID) == 0 || record.ownerUID == ownerUID) {
		delete(c.writes, owner)
	}
}

// EnsureReady returns nil if observed resource versions are at least as new as
// any recorded versions for the given owner, otherwise returning the error of
// what happened. Must not be called concurrent with WroteAt for the same owner.
func (c *consistencyStore) EnsureReady(owner types.NamespacedName) error {
	record := c.getWrittenRecord(owner)
	if record == nil {
		return nil
	}
	err := record.EnsureReady(c)
	if err == nil {
		c.Clear(owner, record.ownerUID)
		return nil
	}
	return err
}

type ownerRecord struct {
	// ownerUID must not be mutated after creation
	ownerUID types.UID

	versionsLock sync.Mutex
	versions     map[schema.GroupResource]string
}

func newOwnerRecord(ownerUID types.UID) *ownerRecord {
	return &ownerRecord{ownerUID: ownerUID, versions: map[schema.GroupResource]string{}}
}

// WroteAt increments the written resource version of an ownerRecord if it is
// the newest seen resource version for that resource.
func (w *ownerRecord) WroteAt(resource schema.GroupResource, rv string) {
	w.versionsLock.Lock()
	defer w.versionsLock.Unlock()
	if _, ok := w.versions[resource]; !ok {
		if _, err := resourceversion.CompareResourceVersion(rv, rv); err != nil {
			return
		}
		w.versions[resource] = rv
		return
	}
	cmp, err := resourceversion.CompareResourceVersion(w.versions[resource], rv)
	if err != nil || cmp >= 0 {
		return
	}
	w.versions[resource] = rv
}

// EnsureReady checks whether or not the ownerRecord is ready compared to the
// read resource versions in the consistency store.
func (w *ownerRecord) EnsureReady(c *consistencyStore) error {
	w.versionsLock.Lock()
	defer w.versionsLock.Unlock()
	for gr, wroteRV := range w.versions {
		store, exists := c.stores[gr]
		if !exists || store == nil {
			return fmt.Errorf("no store registered for group resource %s", gr.String())
		}
		readRV := store.LastStoreSyncResourceVersion()
		if readRV == "" {
			// Store has not observed any sync or bookmark yet (e.g. cache not synced or
			// AtomicFIFO disabled). Fail in the safe direction by reporting not ready.
			return &ConsistencyError{
				WroteRV:       wroteRV,
				ReadRV:        readRV,
				GroupResource: gr,
			}
		}
		i, err := resourceversion.CompareResourceVersion(wroteRV, readRV)
		if err != nil {
			// comparison errors indicate there's a data problem with resource versions, continue so we don't block syncing
			continue
		}
		if i > 0 {
			// read version is not as new as owner version, not ready
			return &ConsistencyError{
				WroteRV:       wroteRV,
				ReadRV:        readRV,
				GroupResource: gr,
			}
		}
	}
	return nil
}

type noopConsistencyStore struct{}

var _ ConsistencyStore = &noopConsistencyStore{}

func (*noopConsistencyStore) WroteAt(owner types.NamespacedName, ownerUID types.UID, resource schema.GroupResource, rv string) {
}

func (*noopConsistencyStore) Clear(owner types.NamespacedName, ownerUID types.UID) {}

func (*noopConsistencyStore) EnsureReady(owner types.NamespacedName) error {
	return nil
}

// NewNoopConsistencyStore creates a ConsistencyStore that records nothing and always
// returns nil from EnsureReady. It can be used when consistency tracking is disabled
// (for example, behind a feature gate or configuration flag) or in unit tests where
// informer stores are not wired up, allowing callers to invoke ConsistencyStore
// methods unconditionally without nil checks.
func NewNoopConsistencyStore() ConsistencyStore {
	return &noopConsistencyStore{}
}
