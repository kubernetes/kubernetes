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
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
)

// Type aliases pointing to the schedulerapi package where the actual
// definitions are maintained. This ensures that any changes to these types
// require autoscaler approval.
type DeviceID = schedulerapi.DeviceID
type SharedDeviceID = schedulerapi.SharedDeviceID
type AllocatedState = schedulerapi.AllocatedState
type ConsumedCapacity = schedulerapi.ConsumedCapacity
type ConsumedCapacityCollection = schedulerapi.ConsumedCapacityCollection
type DeviceConsumedCapacity = schedulerapi.DeviceConsumedCapacity

// Wrapper functions that delegate to the schedulerapi package
func MakeDeviceID(driver, pool, device string) DeviceID {
	return schedulerapi.MakeDeviceID(driver, pool, device)
}

func MakeSharedDeviceID(deviceID DeviceID, shareID *types.UID) SharedDeviceID {
	return schedulerapi.MakeSharedDeviceID(deviceID, shareID)
}

func NewConsumedCapacity() ConsumedCapacity {
	return schedulerapi.NewConsumedCapacity()
}

func NewConsumedCapacityCollection() ConsumedCapacityCollection {
	return schedulerapi.NewConsumedCapacityCollection()
}

func NewDeviceConsumedCapacity(deviceID DeviceID, consumedCapacity map[resourceapi.QualifiedName]resource.Quantity) DeviceConsumedCapacity {
	return schedulerapi.NewDeviceConsumedCapacity(deviceID, consumedCapacity)
}

// IsDeviceAllocated checks if a device is allocated, considering both fully allocated devices
// and partially consumed devices when consumable capacity is enabled.
func IsDeviceAllocated(deviceID DeviceID, allocatedState *AllocatedState) bool {
	// Check if device is fully allocated (traditional case).
	if allocatedState.AllocatedDevices.Has(deviceID) {
		return true
	}

	// Check if device is partially consumed via shared allocations (consumable capacity case).
	// We need to check if any shared device ID corresponds to our device.
	for sharedDeviceID := range allocatedState.AllocatedSharedDeviceIDs {
		if sharedDeviceID.GetDeviceID() == deviceID {
			return true
		}
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
