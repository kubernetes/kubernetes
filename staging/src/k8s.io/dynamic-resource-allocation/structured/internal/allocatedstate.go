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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

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

type SharedDeviceID struct {
	Driver, Pool, Device, ShareID draapi.UniqueString
}

// MakeSharedDeviceID creates a SharedDeviceID by extending MakeDeviceID with shareID.
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

func (d SharedDeviceID) String() string {
	deviceIDStr := d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String()
	if d.ShareID.String() != "" {
		deviceIDStr += "/" + d.ShareID.String()
	}
	return deviceIDStr
}

func GenerateShareID() *types.UID {
	newUID := uuid.NewUUID()
	return &newUID
}

// AllocatedState packs information of allocated devices which is gathered from allocated resource claims.
type AllocatedState struct {
	AllocatedDevices         sets.Set[DeviceID]
	AllocatedSharedDeviceIDs sets.Set[SharedDeviceID]
	AggregatedCapacity       ConsumedCapacityCollection
}

// ConsumedCapacity defines consumable capacity values
type ConsumedCapacity map[resourceapi.QualifiedName]*resource.Quantity

// ConsumedCapacityCollection collects consumable capacity values of each device
type ConsumedCapacityCollection map[DeviceID]ConsumedCapacity

// NewConsumedCapacity initiates a new map of consumable capacity values
func NewConsumedCapacity() ConsumedCapacity {
	return make(ConsumedCapacity)
}

// NewConsumedCapacity initiates a new map of device's consumable capacity values
func NewConsumedCapacityCollection() ConsumedCapacityCollection {
	return make(ConsumedCapacityCollection)
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

// DeviceConsumedCapacity contains consumed capacity result within device allocation.
type DeviceConsumedCapacity struct {
	DeviceID
	ConsumedCapacity
}

// NewDeviceConsumedCapacity creates DeviceConsumedCapacity instance from device ID and its consumed capacity.
func NewDeviceConsumedCapacity(deviceID DeviceID, consumedCapacity map[resourceapi.QualifiedName]resource.Quantity) DeviceConsumedCapacity {
	allocatedCapacity := NewConsumedCapacity()
	for name, quantity := range consumedCapacity {
		allocatedCapacity[name] = &quantity
	}
	return DeviceConsumedCapacity{
		DeviceID:         deviceID,
		ConsumedCapacity: allocatedCapacity,
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
