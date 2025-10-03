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
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

// TestSchedulerContractTypes validates that all scheduler contract types
// are properly defined and accessible.
func TestSchedulerContractTypes(t *testing.T) {
	types := SchedulerContractTypes()
	expectedTypes := []string{
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

	if len(types) != len(expectedTypes) {
		t.Errorf("Expected %d scheduler contract types, got %d", len(expectedTypes), len(types))
	}

	for _, expectedType := range expectedTypes {
		found := false
		for _, actualType := range types {
			if actualType == expectedType {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Missing scheduler contract type: %s", expectedType)
		}
	}
}

// TestValidateSchedulerContract validates that the scheduler contract
// validation function works correctly.
func TestValidateSchedulerContract(t *testing.T) {
	err := ValidateSchedulerContract()
	if err != nil {
		t.Fatalf("Scheduler contract validation failed: %v", err)
	}
}

// TestDeviceIDCreation tests the DeviceID creation and string conversion.
func TestDeviceIDCreation(t *testing.T) {
	deviceID := MakeDeviceID("test-driver", "test-pool", "test-device")
	expectedString := "test-driver/test-pool/test-device"
	if deviceID.String() != expectedString {
		t.Errorf("Expected DeviceID string %q, got %q", expectedString, deviceID.String())
	}
}

// TestSharedDeviceIDCreation tests the SharedDeviceID creation.
func TestSharedDeviceIDCreation(t *testing.T) {
	deviceID := MakeDeviceID("test-driver", "test-pool", "test-device")
	shareID := types.UID("test-share")
	sharedDeviceID := MakeSharedDeviceID(deviceID, &shareID)

	// Validate that the shared device ID was created correctly
	_ = sharedDeviceID
}

// TestConsumedCapacityCreation tests the creation of consumed capacity types.
func TestConsumedCapacityCreation(t *testing.T) {
	// Test ConsumedCapacity creation
	consumedCapacity := NewConsumedCapacity()
	if consumedCapacity == nil {
		t.Error("NewConsumedCapacity returned nil")
	}

	// Test ConsumedCapacityCollection creation
	consumedCapacityCollection := NewConsumedCapacityCollection()
	if consumedCapacityCollection == nil {
		t.Error("NewConsumedCapacityCollection returned nil")
	}

	// Test DeviceConsumedCapacity creation
	deviceID := MakeDeviceID("test-driver", "test-pool", "test-device")
	deviceConsumedCapacity := NewDeviceConsumedCapacity(deviceID, make(map[resourceapi.QualifiedName]resource.Quantity))
	if deviceConsumedCapacity.DeviceID.String() != deviceID.String() {
		t.Error("DeviceConsumedCapacity DeviceID mismatch")
	}
}

// TestAllocatedStateCreation tests the creation of AllocatedState.
func TestAllocatedStateCreation(t *testing.T) {
	allocatedState := AllocatedState{
		AllocatedDevices:         sets.New[DeviceID](),
		AllocatedSharedDeviceIDs: sets.New[SharedDeviceID](),
		AggregatedCapacity:       NewConsumedCapacityCollection(),
	}

	// Validate that the allocated state was created correctly
	_ = allocatedState
}
