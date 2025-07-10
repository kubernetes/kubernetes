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
	resourceapi "k8s.io/api/resource/v1beta1"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

type DeviceClassLister interface {
	// List returns a list of all DeviceClasses.
	List() ([]*resourceapi.DeviceClass, error)
	// Get returns the DeviceClass with the given className.
	Get(className string) (*resourceapi.DeviceClass, error)
}

// Allocator is intentionally not documented here. See the main package for docs.
type Allocator interface {
	ClaimsToAllocate() []*resourceapi.ResourceClaim
	Allocate(ctx context.Context, node *v1.Node) (finalResult []resourceapi.AllocationResult, finalErr error)
}

type Features struct {
	AdminAccess          bool
	PrioritizedList      bool
	PartitionableDevices bool
	DeviceTaints         bool
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
