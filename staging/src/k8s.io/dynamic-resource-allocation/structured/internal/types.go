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

package internal

import (
	"context"
	"errors"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// ErrFailedAllocationOnNode is an empty sentinel for errors.Is.
// See allocator.go for details.
var ErrFailedAllocationOnNode = errors.New("")

type DeviceClassLister interface {
	// List returns a list of all DeviceClasses.
	List() ([]*resourceapi.DeviceClass, error)
	// Get returns the DeviceClass with the given className.
	Get(className string) (*resourceapi.DeviceClass, error)
}

// Allocator is intentionally not documented here. See the main package for docs.
//
// This interface is also broader than the public one.
type Allocator interface {
	Allocate(ctx context.Context, node *v1.Node, claims []*resourceapi.ResourceClaim) (finalResult []resourceapi.AllocationResult, finalErr error)
}

// AllocatorExtended is an optional interface. Not all variants implement it.
type AllocatorExtended interface {
	// Stats shows statistics from the allocation process.
	// May return nil if not implemented.
	GetStats() Stats
}

// Stats shows statistics from the allocation process.
type Stats struct {
	// NumAllocateOneInvocations counts the number of times the allocateOne function
	// got called.
	NumAllocateOneInvocations int64
}

// Features control optional functionality during ResourceClaim allocation.
// Each entry must correspond to at least one control flow change. Entries can
// be removed when the control flow change is no longer necessary (= feature is
// considered stable and always enabled).
//
// This often corresponds to feature gates, but not always: if a KEP implementation
// depends on a set of feature gates, then a single entry here should control whether
// that implementation is active.
type Features struct {
	// Sorted alphabetically. When adding a new entry, also extend Set and FeaturesAll.

	AdminAccess            bool
	ConsumableCapacity     bool
	DeviceBindingAndStatus bool
	DeviceTaints           bool
	PartitionableDevices   bool
	PrioritizedList        bool
}

// Set returns all features which are set to true.
// The names of the features happen to match the Kubernetes
// feature gates where applicable. Plain strings are used
// because not all allocator features necessarily have to
// be Kubernetes feature gates and this package must not
// depend on those definitions.
func (f Features) Set() sets.Set[string] {
	enabled := sets.New[string]()
	if f.AdminAccess {
		enabled.Insert("DRAAdminAccess")
	}
	if f.ConsumableCapacity {
		enabled.Insert("DRAConsumableCapacity")
	}
	if f.DeviceTaints {
		enabled.Insert("DRADeviceTaints")
	}
	if f.PartitionableDevices {
		enabled.Insert("DRAPartitionableDevices")
	}
	if f.PrioritizedList {
		enabled.Insert("DRAPrioritizedList")
	}
	if f.DeviceBindingAndStatus {
		enabled.Insert("DRADeviceBindingConditions+DRAResourceClaimDeviceStatus")
	}
	return enabled
}

var FeaturesAll = Features{
	AdminAccess:            true,
	ConsumableCapacity:     true,
	DeviceBindingAndStatus: true,
	DeviceTaints:           true,
	PartitionableDevices:   true,
	PrioritizedList:        true,
}
