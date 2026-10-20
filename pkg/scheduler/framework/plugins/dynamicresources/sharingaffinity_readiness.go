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

package dynamicresources

import (
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

type exclusionListener func(klog.Logger, sets.Set[poolName], bool)

func (c *claimTracker) subscribeExclusionChanges(listener exclusionListener) func() {
	c.listenersMu.Lock()
	if c.listeners == nil {
		c.listeners = make(map[uint64]exclusionListener)
	}
	c.nextListener++
	id := c.nextListener
	c.listeners[id] = listener
	c.listenersMu.Unlock()
	return func() {
		c.listenersMu.Lock()
		delete(c.listeners, id)
		c.listenersMu.Unlock()
	}
}

// Callers release accounting and publication locks before notifying profiles:
// activation may synchronously enter scheduler queue and plugin code.
func (c *claimTracker) notifyExclusionChanges(logger klog.Logger, changed sets.Set[poolName], startupDone bool) {
	if len(changed) == 0 && !startupDone {
		return
	}
	c.listenersMu.Lock()
	listeners := make([]exclusionListener, 0, len(c.listeners))
	for _, listener := range c.listeners {
		listeners = append(listeners, listener)
	}
	c.listenersMu.Unlock()
	for _, listener := range listeners {
		listener(logger, changed.Clone(), startupDone)
	}
}

type affinityCycleSnapshot struct {
	state    *structured.AllocatedState
	excluded sets.Set[poolName]
	reasons  map[poolName]string
	captured map[poolName]poolMeta
}

// gatherAffinityStateLocked requires inFlightMutex and allocatedDevices.mutex.
// Affinity already includes both owner kinds; capacity intentionally does not
// deduplicate the observed/in-flight overlap, preserving legacy accounting.
func (c *claimTracker) gatherAffinityStateLocked() *structured.AllocatedState {
	a := c.allocatedDevices
	state := &structured.AllocatedState{
		AllocatedDevices:         a.ids.Clone(),
		AllocatedSharedDeviceIDs: sets.New[structured.DeviceID](),
		AggregatedCapacity:       a.capacities.Clone(),
		DeviceSharingAffinity: structured.SharingAffinityState{
			DeviceStates: make(structured.AffinityStateCollection),
			StalePools:   sets.New[schedulerapi.PoolID](),
		},
	}
	for shared := range a.shareIDs {
		state.AllocatedSharedDeviceIDs.Insert(shared.GetDeviceID())
	}
	capacityEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity)
	if !capacityEnabled {
		state.AllocatedDevices.Insert(state.AllocatedSharedDeviceIDs.UnsortedList()...)
		state.AllocatedSharedDeviceIDs = sets.New[structured.DeviceID]()
		state.AggregatedCapacity = structured.NewConsumedCapacityCollection()
	}
	for _, entry := range c.inFlightAllocations {
		foreachAllocatedDevice(entry.claim,
			func(device structured.DeviceID) { state.AllocatedDevices.Insert(device) },
			capacityEnabled,
			func(device structured.SharedDeviceID) { state.AllocatedSharedDeviceIDs.Insert(device.GetDeviceID()) },
			func(capacity structured.DeviceConsumedCapacity) { state.AggregatedCapacity.Insert(capacity) },
		)
	}
	for device, record := range a.affinityDevices {
		if affinity, present := foldDeviceAffinity(record); present {
			state.DeviceSharingAffinity.DeviceStates[device] = affinity
		}
	}
	for device := range a.fenced {
		state.DeviceSharingAffinity.DeviceStates[device] = structured.DeviceAffinityState{Status: structured.AffinityStatusUnreconstructable}
	}
	keys := sets.New[poolName]()
	if c.metadata != nil {
		for key := range c.metadata.pools {
			keys.Insert(key)
		}
	}
	for key := range a.owners {
		keys.Insert(key.pool)
	}
	for key := range keys {
		if !a.poolReadyLocked(key) {
			state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}] = struct{}{}
		}
	}
	return state
}

func (c *claimTracker) gatherAllocatedStateForSlices(all []*resourceapi.ResourceSlice) (affinityCycleSnapshot, error) {
	listed := make(map[poolName]poolMeta)
	for key, slices := range groupSlicesByPool(all) {
		listed[key] = groupedMetadataView(slices, key)
	}
	c.inFlightMutex.RLock()
	defer c.inFlightMutex.RUnlock()
	a := c.allocatedDevices
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if c.metadata == nil || !c.metadata.synced || !a.startupReady {
		return affinityCycleSnapshot{}, errAffinityMetadataNotCoherent
	}
	snapshot := affinityCycleSnapshot{
		state:    c.gatherAffinityStateLocked(),
		excluded: sets.New[poolName](),
		reasons:  make(map[poolName]string),
		captured: make(map[poolName]poolMeta),
	}
	for pool := range snapshot.state.DeviceSharingAffinity.StalePools {
		key := poolName{pool.Driver, pool.Pool}
		snapshot.excluded.Insert(key)
		snapshot.reasons[key] = "notReady"
	}
	for key := range a.affinityMismatchedViewsLocked(listed) {
		snapshot.excluded.Insert(key)
		snapshot.reasons[key] = "listerAhead"
		snapshot.state.DeviceSharingAffinity.StalePools[schedulerapi.PoolID{Driver: key.driver, Pool: key.pool}] = struct{}{}
	}
	for key, meta := range c.metadata.pools {
		if !snapshot.excluded.Has(key) {
			snapshot.captured[key] = meta
		}
	}
	return snapshot, nil
}

func (a *allocatedDevices) affinityMismatchedPoolsLocked(all []*resourceapi.ResourceSlice) sets.Set[poolName] {
	listed := make(map[poolName]poolMeta)
	for key, slices := range groupSlicesByPool(all) {
		listed[key] = groupedMetadataView(slices, key)
	}
	return a.affinityMismatchedViewsLocked(listed)
}

func (a *allocatedDevices) affinityMismatchedViewsLocked(listed map[poolName]poolMeta) sets.Set[poolName] {
	keys := sets.New[poolName]()
	for key := range listed {
		keys.Insert(key)
	}
	for key := range a.metadata.pools {
		keys.Insert(key)
	}
	mismatched := sets.New[poolName]()
	for key := range keys {
		view, found := listed[key]
		published, publishedOK := a.metadata.pools[key]
		if (!found || !publishedOK || !sameMetadata(view, published)) && a.affinityRelevantLocked(key, view, found) {
			mismatched.Insert(key)
		}
	}
	return mismatched
}

func (a *allocatedDevices) affinityRelevantLocked(key poolName, listed poolMeta, listedOK bool) bool {
	hasExtractors := func(view poolMeta) bool {
		return view.digest != "" && view.digest != structured.EmptyExtractorDigest ||
			!view.complete && view.lastCompleteDigest != "" && view.lastCompleteDigest != structured.EmptyExtractorDigest
	}
	if listedOK && hasExtractors(listed) || a.metadata != nil && hasExtractors(a.metadata.pools[key]) {
		return true
	}
	for device, record := range a.affinityDevices {
		if device.Driver.String() == key.driver && device.Pool.String() == key.pool &&
			(len(record.established) != 0 || record.ambiguous || len(record.owners) != 0 || len(record.backers) != 0) {
			return true
		}
	}
	for device := range a.fenced {
		if device.Driver.String() == key.driver && device.Pool.String() == key.pool {
			return true
		}
	}
	return false
}

func (c *claimTracker) exclusionResolved(key poolName) (bool, error) {
	if key == (poolName{}) {
		c.allocatedDevices.mutex.RLock()
		defer c.allocatedDevices.mutex.RUnlock()
		return c.metadata != nil && c.metadata.synced && c.allocatedDevices.startupReady, nil
	}
	all, err := c.metadataSlices.ListPatchedResourceSlices()
	if err != nil {
		return false, err
	}
	view := metadataView(all, key)
	found := false
	for _, slice := range all {
		if slice.Spec.Driver == key.driver && slice.Spec.Pool.Name == key.pool {
			found = true
			break
		}
	}
	a := c.allocatedDevices
	a.mutex.RLock()
	defer a.mutex.RUnlock()
	if c.metadata == nil || !c.metadata.synced || !a.startupReady || !a.poolReadyLocked(key) {
		return false, nil
	}
	published, publishedOK := c.metadata.pools[key]
	return !a.affinityRelevantLocked(key, view, found) || found && publishedOK && sameMetadata(published, view), nil
}
