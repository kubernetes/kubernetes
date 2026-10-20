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

package schedulerapi

import (
	"fmt"
	"maps"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

// DeviceID represents a unique identifier for a device in the DRA system.
// This type is used in the scheduler and autoscaler contract.
type DeviceID struct {
	Driver, Pool, Device draapi.UniqueString
}

func (d DeviceID) String() string {
	return d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String()
}

// MakeDeviceID creates a new DeviceID from driver, pool, and device names.
// This function is used in the scheduler and autoscaler contract.
func MakeDeviceID(driver, pool, device string) DeviceID {
	return DeviceID{
		Driver: draapi.MakeUniqueString(driver),
		Pool:   draapi.MakeUniqueString(pool),
		Device: draapi.MakeUniqueString(device),
	}
}

// SharedDeviceID represents a shared device allocation.
// This type is used in consumable capacity features and the scheduler.
type SharedDeviceID struct {
	Driver, Pool, Device, ShareID draapi.UniqueString
}

func (d SharedDeviceID) String() string {
	deviceIDStr := d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String()
	if d.ShareID.String() != "" {
		deviceIDStr += "/" + d.ShareID.String()
	}
	return deviceIDStr
}

func (d SharedDeviceID) GetDeviceID() DeviceID {
	return DeviceID{d.Driver, d.Pool, d.Device}
}

// MakeSharedDeviceID creates a new SharedDeviceID from a DeviceID and share ID.
// This function is used in consumable capacity features and the scheduler.
func MakeSharedDeviceID(deviceID DeviceID, shareID *types.UID) SharedDeviceID {
	// This function avoids disruptive changes to MakeDeviceID
	// while enabling ShareID as part of the device key.
	var shareIDStr string
	if shareID != nil {
		shareIDStr = string(*shareID)
	}
	return SharedDeviceID{
		Driver:  deviceID.Driver,
		Pool:    deviceID.Pool,
		Device:  deviceID.Device,
		ShareID: draapi.MakeUniqueString(shareIDStr),
	}
}

// AffinityStatus is scheduler-internal state, never serialized.
// Its zero value is AffinityStatusClean, so missing AffinityStateCollection entries
// represent clean devices.
type AffinityStatus int

const (
	AffinityStatusClean AffinityStatus = iota
	AffinityStatusLocked
	// AffinityStatusUnreconstructable means a consistent affinity lock cannot
	// be reconstructed for the device.
	AffinityStatusUnreconstructable
)

func (s AffinityStatus) String() string {
	switch s {
	case AffinityStatusClean:
		return "Clean"
	case AffinityStatusLocked:
		return "Locked"
	case AffinityStatusUnreconstructable:
		return "Unreconstructable"
	default:
		return fmt.Sprintf("AffinityStatus(%d)", s)
	}
}

// DeviceAffinityState is the sharing-affinity state of one device.
type DeviceAffinityState struct {
	Status AffinityStatus
	// LockedAffinity is nonempty only when Status is Locked; nil otherwise.
	LockedAffinity map[string]string
}

// Clone returns a state with an independently owned lock map.
func (s DeviceAffinityState) Clone() DeviceAffinityState {
	cloned := s
	cloned.LockedAffinity = maps.Clone(s.LockedAffinity)
	return cloned
}

// AffinityStateCollection records sharing-affinity state by device.
// Devices without an entry are considered clean.
type AffinityStateCollection map[DeviceID]DeviceAffinityState

// Clone preserves nil and isolates every device's lock map.
func (c AffinityStateCollection) Clone() AffinityStateCollection {
	if c == nil {
		return nil
	}
	cloned := make(AffinityStateCollection, len(c))
	for device, state := range c {
		cloned[device] = state.Clone()
	}
	return cloned
}

// SharingAffinityState holds device affinity and pool readiness for allocation.
// Its zero value represents clean devices and no stale pools.
type SharingAffinityState struct {
	// DeviceStates records shared-device affinity when enabled. Missing entries
	// are Clean; the collection is nil when sharing affinity is disabled.
	DeviceStates AffinityStateCollection
	// StalePools lists pools whose sharing affinity metadata is not yet
	// coherent with the slices. Backends that receive the full AllocatedState
	// treat them as incomplete. In-memory only; nil means none.
	StalePools sets.Set[PoolID]
}

// Clone preserves nil collections and isolates device locks and stale pools.
func (s SharingAffinityState) Clone() SharingAffinityState {
	cloned := s
	cloned.DeviceStates = s.DeviceStates.Clone()
	cloned.StalePools = maps.Clone(s.StalePools)
	return cloned
}

// AllocatedState represents the current state of allocated resources.
// This type is used in the scheduler and autoscaler contract.
// AllocatedState packs information of allocated devices which is gathered from allocated resource claims.
type AllocatedState struct {
	// AllocatedDevices contains device IDs that are exclusively allocated to a claim.
	AllocatedDevices sets.Set[DeviceID]
	// AllocatedSharedDeviceIDs contains device IDs that have one or more shared allocations
	// when the DRAConsumableCapacity feature is enabled.
	AllocatedSharedDeviceIDs sets.Set[DeviceID]
	// AggregatedCapacity records the consumed capacity per device ID when
	// the DRAConsumableCapacity feature is enabled.
	AggregatedCapacity ConsumedCapacityCollection
	// DeviceSharingAffinity records sharing-affinity state when the
	// DRASharingAffinity feature is enabled.
	DeviceSharingAffinity SharingAffinityState
}

// PoolID identifies a pool by driver and pool name.
type PoolID struct {
	Driver, Pool string
}

func (p PoolID) String() string {
	return p.Driver + "/" + p.Pool
}

// ConsumedCapacity represents the consumed capacity of a specific resource.
// This type is used in consumable capacity features and the scheduler.
// ConsumedCapacity defines consumable capacity values.
//
// Keys are fully-qualified names, so capacity in different domains is never conflated
// regardless of whether a name's domain was given explicitly or left implicit.
// Values are pointers to support in-place updates, for example via Add.
type ConsumedCapacity map[draapi.FullyQualifiedName]*resource.Quantity

// NewConsumedCapacity creates a new ConsumedCapacity.
// This function is used in consumable capacity features and the scheduler.
// NewConsumedCapacity initiates a new map of consumable capacity values
func NewConsumedCapacity() ConsumedCapacity {
	return make(ConsumedCapacity)
}

// Clone makes a copy of consumed capacity values
func (s ConsumedCapacity) Clone() ConsumedCapacity {
	clone := make(ConsumedCapacity)
	for name, quantity := range s {
		q := quantity.DeepCopy()
		clone[name] = &q
	}
	return clone
}

// Add adds quantity to corresponding consumable capacity,
// and creates a new entry if no capacity created yet.
func (s ConsumedCapacity) Add(addedCapacity ConsumedCapacity) {
	for name, quantity := range addedCapacity {
		val := quantity.DeepCopy()
		if _, found := s[name]; found {
			s[name].Add(val)
		} else {
			s[name] = &val
		}
	}
}

// Sub subtracts quantity,
// and ignore if no capacity entry found.
func (s ConsumedCapacity) Sub(subtractedCapacity ConsumedCapacity) {
	for name, quantity := range subtractedCapacity {
		if _, found := s[name]; found {
			s[name].Sub(*quantity)
		}
	}
}

// Empty return true if all quantity is zero.
func (s ConsumedCapacity) Empty() bool {
	for _, quantity := range s {
		if !quantity.IsZero() {
			return false
		}
	}
	return true
}

// ConsumedCapacityCollection represents a collection of consumed capacities.
// This type is used in consumable capacity features and the scheduler.
// ConsumedCapacityCollection collects consumable capacity values of each device
type ConsumedCapacityCollection map[DeviceID]ConsumedCapacity

// NewConsumedCapacityCollection creates a new ConsumedCapacityCollection.
// This function is used in consumable capacity features and the scheduler.
// NewConsumedCapacityCollection initiates a new map of device's consumable capacity values
func NewConsumedCapacityCollection() ConsumedCapacityCollection {
	return make(ConsumedCapacityCollection)
}

// Clone makes a copy of ConsumedCapacity of each capacity.
func (c ConsumedCapacityCollection) Clone() ConsumedCapacityCollection {
	clone := NewConsumedCapacityCollection()
	for deviceID, share := range c {
		clone[deviceID] = share.Clone()
	}
	return clone
}

// Insert adds a new allocated capacity to the collection.
func (c ConsumedCapacityCollection) Insert(cap DeviceConsumedCapacity) {
	consumedCapacity := cap.ConsumedCapacity
	if _, found := c[cap.DeviceID]; found {
		c[cap.DeviceID].Add(consumedCapacity)
	} else {
		c[cap.DeviceID] = consumedCapacity.Clone()
	}
}

// Remove removes an allocated capacity from the collection.
func (c ConsumedCapacityCollection) Remove(cap DeviceConsumedCapacity) {
	if _, found := c[cap.DeviceID]; found {
		c[cap.DeviceID].Sub(cap.ConsumedCapacity)
		if c[cap.DeviceID].Empty() {
			delete(c, cap.DeviceID)
		}
	}
}

// DeviceConsumedCapacity represents the consumed capacity of a device.
// This type is used in consumable capacity features and the scheduler.
// DeviceConsumedCapacity contains consumed capacity result within device allocation.
type DeviceConsumedCapacity struct {
	DeviceID
	ConsumedCapacity
}

// NewDeviceConsumedCapacity creates a new DeviceConsumedCapacity for deviceID from
// consumedCapacity as found in a DeviceRequestAllocationResult.
// Each key is resolved against deviceID.Driver so that the returned
// DeviceConsumedCapacity, like the rest of the internal ConsumedCapacity tracking, is
// always keyed by fully-qualified names.
//
// consumedCapacity is not guaranteed to be free of names that resolve to the same
// fully-qualified name: a 1.37 allocator could persist both "cap" and "<driver>/cap" for a
// device that published both. Resolving each name independently and inserting it into
// normalized would then make the outcome depend on map iteration order, whichever entry
// is visited last would win. To make the result deterministic without iterating
// consumedCapacity twice: an entry explicitly qualified with deviceID.Driver always
// overwrites whatever is already recorded for its key, while an unqualified entry is
// dropped if an entry (qualified or not) was already recorded for its key.
//
// Callers that already have a ConsumedCapacity (for example, the allocators
// themselves, while computing what a request would consume) do not need this
// conversion and can construct a DeviceConsumedCapacity directly instead.
func NewDeviceConsumedCapacity(deviceID DeviceID, consumedCapacity map[resourceapi.QualifiedName]resource.Quantity) DeviceConsumedCapacity {
	normalized := make(ConsumedCapacity, len(consumedCapacity))
	driverPrefix := deviceID.Driver.String() + "/"
	for name, val := range consumedCapacity {
		identifier, ok := strings.CutPrefix(string(name), driverPrefix)
		if ok {
			key := draapi.FullyQualifiedName{Domain: deviceID.Driver.String(), Identifier: identifier}
			normalized[key] = new(val)
			continue
		}
		key := draapi.MakeFullyQualifiedName(name, deviceID.Driver.String())
		if _, alreadySet := normalized[key]; alreadySet {
			continue
		}
		normalized[key] = new(val)
	}
	return DeviceConsumedCapacity{
		DeviceID:         deviceID,
		ConsumedCapacity: normalized,
	}
}

// Clone makes a copy of DeviceConsumedCapacity.
func (a DeviceConsumedCapacity) Clone() DeviceConsumedCapacity {
	return DeviceConsumedCapacity{
		DeviceID:         a.DeviceID,
		ConsumedCapacity: a.ConsumedCapacity.Clone(),
	}
}

// String returns formatted device ID.
func (a DeviceConsumedCapacity) String() string {
	return a.DeviceID.String()
}
