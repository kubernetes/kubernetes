/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/ptr"
)

// consumedCapacityError checks the consumed capacity of each allocation result.
// Checking only the aggregated capacity would not be enough, because a positive
// value from another claim for the same device can hide a negative value.
// The stored names get checked, because NewDeviceConsumedCapacity drops an
// unqualified name when the qualified name is also set.
func consumedCapacityError(claim *resourceapi.ResourceClaim) error {
	if claim.Status.Allocation == nil {
		return nil
	}
	for _, result := range claim.Status.Allocation.Devices.Results {
		for _, name := range slices.Sorted(maps.Keys(result.ConsumedCapacity)) {
			if quantity := result.ConsumedCapacity[name]; quantity.Sign() < 0 {
				deviceID := structured.MakeDeviceID(result.Driver, result.Pool, result.Device)
				return fmt.Errorf("claim %s, device %s: consumed capacity %q must not be negative, got %s", klog.KObj(claim), deviceID, name, quantity.String())
			}
		}
	}
	return nil
}

// allocatedDevices reacts to events in a cache and maintains a set of all allocated devices.
// This is cheaper than repeatedly calling List, making strings unique, and building the set
// each time PreFilter is called.
//
// To simplify detecting concurrent changes, each modification bumps a revision counter,
// similar to ResourceVersion in the apiserver. get and getCapacities include the
// current value in their result. A caller than can compare againt the current value
// to determine whether some prior results are still up-to-date, without having to get
// and compare them.
//
// All methods are thread-safe. get returns a cloned set.
type allocatedDevices struct {
	logger klog.Logger

	mutex                     sync.RWMutex
	revision                  int64
	ids                       sets.Set[structured.DeviceID]
	shareIDs                  sets.Set[structured.SharedDeviceID]
	capacities                structured.ConsumedCapacityCollection
	invalidCapacities         map[types.UID]error
	enabledConsumableCapacity bool
}

func newAllocatedDevices(logger klog.Logger) *allocatedDevices {
	return &allocatedDevices{
		logger:                    logger,
		ids:                       sets.New[structured.DeviceID](),
		shareIDs:                  sets.New[structured.SharedDeviceID](),
		capacities:                structured.NewConsumedCapacityCollection(),
		invalidCapacities:         make(map[types.UID]error),
		enabledConsumableCapacity: utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity),
	}
}

func (a *allocatedDevices) get() (sets.Set[structured.DeviceID], int64) {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	return a.ids.Clone(), a.revision
}

func (a *allocatedDevices) getSharedDeviceIDs() (sets.Set[structured.DeviceID], int64) {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	result := make(sets.Set[structured.DeviceID], a.shareIDs.Len())
	for sharedDeviceID := range a.shareIDs {
		result.Insert(sharedDeviceID.GetDeviceID())
	}
	return result, a.revision
}

// getCapacities returns an error instead of the capacities if some allocated claim
// has a negative consumed capacity, because the aggregated capacities then
// under-count what is in use. The error remains until all such claims are removed.
func (a *allocatedDevices) getCapacities() (structured.ConsumedCapacityCollection, int64, error) {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	if len(a.invalidCapacities) > 0 {
		errs := slices.SortedFunc(maps.Values(a.invalidCapacities), func(x, y error) int {
			return strings.Compare(x.Error(), y.Error())
		})
		if len(errs) > 1 {
			return nil, a.revision, fmt.Errorf("%w (and %d more claims with invalid consumed capacity)", errs[0], len(errs)-1)
		}
		return nil, a.revision, errs[0]
	}
	return a.capacities.Clone(), a.revision, nil
}

func (a *allocatedDevices) getRevision() int64 {
	a.mutex.RLock()
	defer a.mutex.RUnlock()

	return a.revision
}

func (a *allocatedDevices) handlers() cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		AddFunc:    a.onAdd,
		UpdateFunc: a.onUpdate,
		DeleteFunc: a.onDelete,
	}
}

func (a *allocatedDevices) onAdd(obj any) {
	claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onAdd")
		return
	}

	if claim.Status.Allocation != nil {
		a.addDevices(claim)
	}
}

func (a *allocatedDevices) onUpdate(oldObj, newObj any) {
	originalClaim, modifiedClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onUpdate")
		return
	}

	if originalClaim.UID == modifiedClaim.UID && a.allocationsEqual(originalClaim.Status.Allocation, modifiedClaim.Status.Allocation) {
		return
	}

	// Informers can skip deletion/recreation or the cleared allocation between
	// uses of the same claim, so two non-nil allocations may still differ.
	a.updateDevices(originalClaim, modifiedClaim)
}

// allocationsEqual compares the allocation fields used by collectDevices and
// consumedCapacityError. onUpdate checks the claim UID separately.
// Only device identity, admin access, share IDs, and consumed quantities affect
// accounting or capacity validation. Configuration, node selectors, and other
// allocation fields do not affect the tracker, so a semantic deep comparison of
// the entire allocation is unnecessary. Direct comparisons avoid reflection;
// Quantity.Equal preserves numeric equality across quantity representations.
// Capacity fields are irrelevant when consumable capacity is disabled.
// Keep this comparison in sync with the fields consumed by those helpers.
func (a *allocatedDevices) allocationsEqual(old, new *resourceapi.AllocationResult) bool {
	if old == new {
		return true
	}
	if old == nil || new == nil {
		return false
	}
	return slices.EqualFunc(old.Devices.Results, new.Devices.Results, func(x, y resourceapi.DeviceRequestAllocationResult) bool {
		if x.Driver != y.Driver || x.Pool != y.Pool || x.Device != y.Device ||
			ptr.Deref(x.AdminAccess, false) != ptr.Deref(y.AdminAccess, false) {
			return false
		}
		return !a.enabledConsumableCapacity ||
			(ptr.Equal(x.ShareID, y.ShareID) && maps.EqualFunc(x.ConsumedCapacity, y.ConsumedCapacity, resource.Quantity.Equal))
	})
}

func (a *allocatedDevices) onDelete(obj any) {
	claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
	if err != nil {
		// Shouldn't happen.
		a.logger.Error(err, "unexpected object in allocatedDevices.onDelete")
		return
	}

	a.removeDevices(claim)
}

// addDevices calls updateDevices with no old claim, so only the new allocation
// is added to the tracker.
func (a *allocatedDevices) addDevices(claim *resourceapi.ResourceClaim) {
	a.updateDevices(nil, claim)
}

// removeDevices calls updateDevices with no new claim, so only the old allocation
// is removed from the tracker.
func (a *allocatedDevices) removeDevices(claim *resourceapi.ResourceClaim) {
	a.updateDevices(claim, nil)
}

type allocatedClaimDevices struct {
	deviceIDs   []structured.DeviceID
	shareIDs    []structured.SharedDeviceID
	capacities  []structured.DeviceConsumedCapacity
	capacityErr error
}

// collectDevices prepares a claim's device IDs, share IDs and consumed capacity
// without changing the tracker, so this work can happen outside its lock.
// If capacity is invalid, it records the error and omits the capacity amounts.
func (a *allocatedDevices) collectDevices(claim *resourceapi.ResourceClaim, operation string) allocatedClaimDevices {
	var devices allocatedClaimDevices
	if claim == nil || claim.Status.Allocation == nil {
		return devices
	}
	devices.deviceIDs = make([]structured.DeviceID, 0, 20)
	if a.enabledConsumableCapacity {
		devices.shareIDs = make([]structured.SharedDeviceID, 0, 20)
		devices.capacities = make([]structured.DeviceConsumedCapacity, 0, 20)
		devices.capacityErr = consumedCapacityError(claim)
	}
	structured.ForEachAllocatedDevice(claim.Status.Allocation,
		func(deviceID structured.DeviceID) {
			a.logger.V(6).Info("Observed device "+operation, "device", deviceID, "claim", klog.KObj(claim))
			devices.deviceIDs = append(devices.deviceIDs, deviceID)
		},
		a.enabledConsumableCapacity,
		func(sharedDeviceID structured.SharedDeviceID) {
			a.logger.V(6).Info("Observed shared device "+operation, "shared device", sharedDeviceID, "claim", klog.KObj(claim))
			devices.shareIDs = append(devices.shareIDs, sharedDeviceID)
		},
		func(capacity structured.DeviceConsumedCapacity) {
			a.logger.V(6).Info("Observed consumed capacity "+operation, "device", capacity.DeviceID, "consumed capacity", capacity.ConsumedCapacity, "claim", klog.KObj(claim))
			// A negative value can hide valid consumption during aggregation.
			// Invalid claims must neither add nor subtract capacity.
			if devices.capacityErr == nil {
				devices.capacities = append(devices.capacities, capacity)
			}
		},
	)
	return devices
}

// updateDevices atomically replaces oldClaim's accounting with newClaim's.
// Either claim may be nil for an add or removal. Collection happens before
// locking, removal and insertion share one lock and revision change so readers
// cannot observe a temporarily free device during replacement.
func (a *allocatedDevices) updateDevices(oldClaim, newClaim *resourceapi.ResourceClaim) {
	oldDevices := a.collectDevices(oldClaim, "deallocation")
	newDevices := a.collectDevices(newClaim, "allocation")
	if len(oldDevices.deviceIDs)+len(newDevices.deviceIDs) == 0 &&
		len(oldDevices.shareIDs)+len(newDevices.shareIDs) == 0 &&
		len(oldDevices.capacities)+len(newDevices.capacities) == 0 &&
		oldDevices.capacityErr == nil && newDevices.capacityErr == nil {
		return
	}

	a.mutex.Lock()
	defer a.mutex.Unlock()
	a.revision++
	for _, deviceID := range oldDevices.deviceIDs {
		a.ids.Delete(deviceID)
	}
	for _, shareID := range oldDevices.shareIDs {
		a.shareIDs.Delete(shareID)
	}
	for _, capacity := range oldDevices.capacities {
		a.capacities.Remove(capacity)
	}
	if oldClaim != nil {
		delete(a.invalidCapacities, oldClaim.UID)
	}
	for _, deviceID := range newDevices.deviceIDs {
		a.ids.Insert(deviceID)
	}
	for _, shareID := range newDevices.shareIDs {
		a.shareIDs.Insert(shareID)
	}
	for _, capacity := range newDevices.capacities {
		a.capacities.Insert(capacity)
	}
	if newDevices.capacityErr != nil {
		a.invalidCapacities[newClaim.UID] = newDevices.capacityErr
	}
}
