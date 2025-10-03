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
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
)

// DeviceID represents a unique identifier for a device in the DRA system.
// This type is used in the scheduler and autoscaler contract.
type DeviceID = structured.DeviceID

// AllocatedState represents the current state of allocated resources.
// This type is used in the scheduler and autoscaler contract.
type AllocatedState = structured.AllocatedState

// SharedDeviceID represents a shared device allocation.
// This type is used in consumable capacity features and the scheduler.

type SharedDeviceID = structured.SharedDeviceID

// DeviceConsumedCapacity represents the consumed capacity of a device.
// This type is used in consumable capacity features and the scheduler.
type DeviceConsumedCapacity = structured.DeviceConsumedCapacity

// ConsumedCapacityCollection represents a collection of consumed capacities.
// This type is used in consumable capacity features and the scheduler.
type ConsumedCapacityCollection = structured.ConsumedCapacityCollection

// ConsumedCapacity represents the consumed capacity of a specific resource.
// This type is used in consumable capacity features and the scheduler.
type ConsumedCapacity = structured.ConsumedCapacity

// MakeDeviceID creates a new DeviceID from driver, pool, and device names.
// This function is used in the scheduler and autoscaler contract.
func MakeDeviceID(driver, pool, device string) DeviceID {
	return structured.MakeDeviceID(driver, pool, device)
}

// MakeSharedDeviceID creates a new SharedDeviceID from a DeviceID and share ID.
// This function is used in consumable capacity features and the scheduler.
func MakeSharedDeviceID(deviceID DeviceID, shareID *types.UID) SharedDeviceID {
	return structured.MakeSharedDeviceID(deviceID, shareID)
}

// NewConsumedCapacityCollection creates a new ConsumedCapacityCollection.
// This function is used in consumable capacity features and the scheduler.
func NewConsumedCapacityCollection() ConsumedCapacityCollection {
	return structured.NewConsumedCapacityCollection()
}

// NewConsumedCapacity creates a new ConsumedCapacity.
// This function is used in consumable capacity features and the scheduler.
func NewConsumedCapacity() ConsumedCapacity {
	return make(ConsumedCapacity)
}

// NewDeviceConsumedCapacity creates a new DeviceConsumedCapacity.
// This function is used in consumable capacity features and the scheduler.
func NewDeviceConsumedCapacity(deviceID DeviceID, consumedCapacity map[resourceapi.QualifiedName]resource.Quantity) DeviceConsumedCapacity {
	return structured.NewDeviceConsumedCapacity(deviceID, consumedCapacity)
}

// SchedulerContractTypes returns a list of all types in this package that are
// part of the scheduler contract and require SIG Autoscaling review.
//
// This function is used for validation and documentation purposes.
func SchedulerContractTypes() []string {
	return []string{
		"DeviceID",
		"AllocatedState",
		"SharedDeviceID",
		"DeviceConsumedCapacity",
		"ConsumedCapacityCollection",
		"ConsumedCapacity",
		"MakeDeviceID",
		"MakeSharedDeviceID",
		"NewConsumedCapacityCollection",
		"NewConsumedCapacity",
		"NewDeviceConsumedCapacity",
	}
}

// ValidateSchedulerContract validates that all scheduler contract types
// are accessible and compatible.
//
// This function should be called in tests to ensure that changes to the
// underlying structured package don't break the scheduler contract.
func ValidateSchedulerContract() error {
	// Create instances of all types to ensure they are accessible
	deviceID := MakeDeviceID("test-driver", "test-pool", "test-device")
	_ = deviceID.String()

	allocatedState := AllocatedState{
		AllocatedDevices:         sets.New[DeviceID](),
		AllocatedSharedDeviceIDs: sets.New[SharedDeviceID](),
		AggregatedCapacity:       NewConsumedCapacityCollection(),
	}
	_ = allocatedState

	sharedDeviceID := MakeSharedDeviceID(deviceID, nil)
	_ = sharedDeviceID

	consumedCapacity := NewConsumedCapacity()
	_ = consumedCapacity

	consumedCapacityCollection := NewConsumedCapacityCollection()
	_ = consumedCapacityCollection

	deviceConsumedCapacity := NewDeviceConsumedCapacity(deviceID, make(map[resourceapi.QualifiedName]resource.Quantity))
	_ = deviceConsumedCapacity

	return nil
}
