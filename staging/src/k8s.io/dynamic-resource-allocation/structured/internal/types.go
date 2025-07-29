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

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

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

// Features contains all feature gates that may influence the behavior of ResourceClaim allocation.
type Features struct {
	// Sorted alphabetically. When adding a new entry, also extend Set and FeaturesAll.

	AdminAccess          bool
	DeviceTaints         bool
	PartitionableDevices bool
	PrioritizedList      bool
	// ResourceSliceMixins specified whether the DRA ResourceSlice Mixins feature
	// should be enabled.
	//
	// The allocator handles mixins by merging the values defined directly on devices
	// or counter sets with the values in the mixins when needed. This means that
	// capacity and attributes from device mixins are merged when the selectors for
	// a device is evaluated. For counters, either for consumption in devices or in counter
	// sets, values from mixins are merged when the allocator needs to determine if there
	// are sufficient counters available for a specifc device.
	//
	// To avoid unnecessary memory consumption, we decided against merging the mixins
	// into all devices before the allocation process.
	ResourceSliceMixins bool
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
	if f.DeviceTaints {
		enabled.Insert("DRADeviceTaints")
	}
	if f.PartitionableDevices {
		enabled.Insert("DRAPartitionableDevices")
	}
	if f.PrioritizedList {
		enabled.Insert("DRAPrioritizedList")
	}
	if f.ResourceSliceMixins {
		enabled.Insert("DRAResourceSliceMixins")
	}
	return enabled
}

var FeaturesAll = Features{
	AdminAccess:          true,
	DeviceTaints:         true,
	PartitionableDevices: true,
	PrioritizedList:      true,
	ResourceSliceMixins:  true,
}

type DeviceID struct {
	Driver, Pool, Device draapi.UniqueString
}

func (d DeviceID) String() string {
	return d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String()
}

func MakeDeviceID(driver, pool, device string) DeviceID {
	return DeviceID{
		Driver: draapi.MakeUniqueString(driver),
		Pool:   draapi.MakeUniqueString(pool),
		Device: draapi.MakeUniqueString(device),
	}
}
