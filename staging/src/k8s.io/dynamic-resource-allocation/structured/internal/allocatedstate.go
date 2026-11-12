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

package internal

// This file should be moved under pkg/scheduler/framework/autoscaler_constract.
// See https://github.com/kubernetes/kubernetes/issues/133161.

import (
	"iter"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/utils/ptr"
)

// Type aliases pointing to the schedulerapi package where the actual
// definitions are maintained. This ensures that any changes to these types
// require autoscaler approval.
type DeviceID = schedulerapi.DeviceID
type AllocatedState = schedulerapi.AllocatedState
type ConsumedCapacity = schedulerapi.ConsumedCapacity
type ConsumedCapacityCollection = schedulerapi.ConsumedCapacityCollection
type DeviceConsumedCapacity = schedulerapi.DeviceConsumedCapacity

// Wrapper functions that delegate to the schedulerapi package
func MakeDeviceID(driver, pool, device string) DeviceID {
	return schedulerapi.MakeDeviceID(driver, pool, device)
}

func NewConsumedCapacity() ConsumedCapacity {
	return schedulerapi.NewConsumedCapacity()
}

func NewConsumedCapacityCollection() ConsumedCapacityCollection {
	return schedulerapi.NewConsumedCapacityCollection()
}

// IsDeviceAllocated checks if a device is allocated, considering both fully allocated devices
// and partially consumed devices when consumable capacity is enabled.
func IsDeviceAllocated(deviceID DeviceID, allocatedState *AllocatedState) bool {
	// Check if device is fully allocated (traditional case).
	if allocatedState.AllocatedDevices.Has(deviceID) {
		return true
	}

	// Check if device is partially consumed via shared allocations (consumable capacity case).
	if allocatedState.AllocatedSharedDeviceIDs.Has(deviceID) {
		return true
	}

	// For scheduler-generated state, consumed capacity is recorded together with
	// a shared device ID. Keep this check to preserve IsDeviceAllocated semantics
	// and handle manually constructed or future AllocatedState producers.
	if _, hasConsumedCapacity := allocatedState.AggregatedCapacity[deviceID]; hasConsumedCapacity {
		return true
	}

	return false
}

// GenerateShareID is a helper function that generates a new share ID.
// This remains in the internal package as it's a utility function.
func GenerateShareID() *types.UID {
	newUID := uuid.NewUUID()
	return &newUID
}

// ForEachAllocatedDevice implements structured.ForEachAllocatedDevice.
func ForEachAllocatedDevice(allocation *resourceapi.AllocationResult,
	dedicatedDeviceCallback func(deviceID DeviceID),
	enabledConsumableCapacity bool,
	sharedDeviceCallback func(schedulerapi.SharedDeviceID),
	consumedCapacityCallback func(DeviceConsumedCapacity)) {
	if allocation == nil {
		return
	}
	for _, result := range allocation.Devices.Results {
		// Kubernetes 1.31 did not set this, 1.32 always does.
		// Supporting 1.31 is not worth the additional code that
		// would have to be written (= looking up in request) because
		// it is extremely unlikely that there really is a result
		// that still exists in a cluster from 1.31 where this matters.
		if ptr.Deref(result.AdminAccess, false) {
			// Is not considered as allocated.
			continue
		}
		deviceID := MakeDeviceID(result.Driver, result.Pool, result.Device)

		// None of the users of this helper need to abort iterating,
		// therefore it's not supported as it only would add overhead.

		// Execute sharedDeviceCallback and consumedCapacityCallback correspondingly
		// if DRAConsumableCapacity feature is enabled
		if enabledConsumableCapacity {
			shared := result.ShareID != nil
			if shared {
				sharedDeviceID := schedulerapi.MakeSharedDeviceID(deviceID, result.ShareID)
				sharedDeviceCallback(sharedDeviceID)
				if result.ConsumedCapacity != nil {
					deviceConsumedCapacity := schedulerapi.NewDeviceConsumedCapacity(deviceID, result.ConsumedCapacity)
					consumedCapacityCallback(deviceConsumedCapacity)
				}
				continue
			}
		}

		// Otherwise, execute dedicatedDeviceCallback
		dedicatedDeviceCallback(deviceID)
	}
}

// AddToAllocatedState implements structured.AddToAllocatedState.
func AddToAllocatedState(state *AllocatedState, allocations iter.Seq[*resourceapi.AllocationResult], features Features) {
	if state.AllocatedDevices == nil {
		state.AllocatedDevices = sets.New[DeviceID]()
	}
	if state.AllocatedSharedDeviceIDs == nil {
		state.AllocatedSharedDeviceIDs = sets.New[DeviceID]()
	}
	if state.AggregatedCapacity == nil {
		state.AggregatedCapacity = NewConsumedCapacityCollection()
	}
	for allocation := range allocations {
		ForEachAllocatedDevice(allocation,
			func(deviceID DeviceID) {
				state.AllocatedDevices.Insert(deviceID)
			},
			features.ConsumableCapacity,
			func(sharedDeviceID schedulerapi.SharedDeviceID) {
				state.AllocatedSharedDeviceIDs.Insert(sharedDeviceID.GetDeviceID())
			},
			func(capacity DeviceConsumedCapacity) {
				state.AggregatedCapacity.Insert(capacity)
			},
		)
	}
}
