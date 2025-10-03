/*
Copyright 2022 The Kubernetes Authors.

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

// This file helps to detect whether we changed the interface of Listers.
// This is important for downstream projects who import the Listers to not
// breaking their codes.

package contract

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	fwk "k8s.io/kube-scheduler/framework"
)

var _ fwk.NodeInfoLister = &nodeInfoListerContract{}
var _ fwk.StorageInfoLister = &storageInfoListerContract{}
var _ fwk.SharedLister = &shareListerContract{}
var _ fwk.ResourceSliceLister = &resourceSliceListerContract{}
var _ fwk.DeviceClassLister = &deviceClassListerContract{}
var _ fwk.ResourceClaimTracker = &resourceClaimTrackerContract{}
var _ fwk.SharedDRAManager = &sharedDRAManagerContract{}

type nodeInfoListerContract struct{}

func (c *nodeInfoListerContract) List() ([]fwk.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) HavePodsWithAffinityList() ([]fwk.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) HavePodsWithRequiredAntiAffinityList() ([]fwk.NodeInfo, error) {
	return nil, nil
}

func (c *nodeInfoListerContract) Get(_ string) (fwk.NodeInfo, error) {
	return nil, nil
}

type storageInfoListerContract struct{}

func (c *storageInfoListerContract) IsPVCUsedByPods(_ string) bool {
	return false
}

type shareListerContract struct{}

func (c *shareListerContract) NodeInfos() fwk.NodeInfoLister {
	return nil
}

func (c *shareListerContract) StorageInfos() fwk.StorageInfoLister {
	return nil
}

type resourceSliceListerContract struct{}

func (c *resourceSliceListerContract) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return nil, nil
}

type deviceClassListerContract struct{}

func (c *deviceClassListerContract) List() ([]*resourceapi.DeviceClass, error) {
	return nil, nil
}

func (c *deviceClassListerContract) Get(_ string) (*resourceapi.DeviceClass, error) {
	return nil, nil
}

type resourceClaimTrackerContract struct{}

func (r *resourceClaimTrackerContract) List() ([]*resourceapi.ResourceClaim, error) {
	return nil, nil
}

func (r *resourceClaimTrackerContract) Get(_, _ string) (*resourceapi.ResourceClaim, error) {
	return nil, nil
}

func (r *resourceClaimTrackerContract) ListAllAllocatedDevices() (sets.Set[schedulerapi.DeviceID], error) {
	return nil, nil
}

func (r *resourceClaimTrackerContract) GatherAllocatedState() (*schedulerapi.AllocatedState, error) {
	return nil, nil
}

func (r *resourceClaimTrackerContract) SignalClaimPendingAllocation(_ types.UID, _ *resourceapi.ResourceClaim) error {
	return nil
}

func (r *resourceClaimTrackerContract) ClaimHasPendingAllocation(_ types.UID) bool {
	return false
}

func (r *resourceClaimTrackerContract) RemoveClaimPendingAllocation(_ types.UID) (deleted bool) {
	return false
}

func (r *resourceClaimTrackerContract) AssumeClaimAfterAPICall(_ *resourceapi.ResourceClaim) error {
	return nil
}

func (r *resourceClaimTrackerContract) AssumedClaimRestore(_, _ string) {
}

type sharedDRAManagerContract struct{}

func (s *sharedDRAManagerContract) ResourceClaims() fwk.ResourceClaimTracker {
	return nil
}

func (s *sharedDRAManagerContract) ResourceSlices() fwk.ResourceSliceLister {
	return nil
}

func (s *sharedDRAManagerContract) DeviceClasses() fwk.DeviceClassLister {
	return nil
}

// TestSchedulerAPIContract validates that the schedulerapi package types
// used in the autoscaler contract are accessible and compatible.
//
// This test ensures that changes to the schedulerapi package are visible to
// SIG Autoscaling during code review, as these types are part of the
// autoscaler contract.
func TestSchedulerAPIContract(t *testing.T) {
	// Validate that the schedulerapi contract is valid
	err := schedulerapi.ValidateSchedulerContract()
	if err != nil {
		t.Fatalf("SchedulerAPI contract validation failed: %v", err)
	}

	// Validate that all schedulerapi types used in the autoscaler contract
	// are properly tracked
	types := schedulerapi.SchedulerContractTypes()
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
		t.Errorf("Expected %d schedulerapi types, got %d", len(expectedTypes), len(types))
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
			t.Errorf("Missing schedulerapi type in contract: %s", expectedType)
		}
	}
}

func TestResourceClaimTrackerSchedulerAPITypes(t *testing.T) {
	// Create a contract instance to test the interface
	tracker := &resourceClaimTrackerContract{}

	// Test ListAllAllocatedDevices method with schedulerapi.DeviceID
	devices, err := tracker.ListAllAllocatedDevices()
	if err != nil {
		t.Errorf("ListAllAllocatedDevices failed: %v", err)
	}
	if devices == nil {
		// This is expected for the contract implementation, but ensures the method is accessible
		devices = sets.New[schedulerapi.DeviceID]()
	}

	// Test GatherAllocatedState method with schedulerapi.AllocatedState
	state, err := tracker.GatherAllocatedState()
	if err != nil {
		t.Errorf("GatherAllocatedState failed: %v", err)
	}
	if state == nil {
		// This is expected for the contract implementation, but ensures the method is accessible
		state = &schedulerapi.AllocatedState{
			AllocatedDevices:         sets.New[schedulerapi.DeviceID](),
			AllocatedSharedDeviceIDs: sets.New[schedulerapi.SharedDeviceID](),
			AggregatedCapacity:       schedulerapi.NewConsumedCapacityCollection(),
		}
	}

	// Validate that the schedulerapi package types are properly accessible
	_ = devices
	_ = state
}
