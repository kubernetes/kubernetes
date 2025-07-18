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

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"

	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

// DRAConsumableCapacity types

type SharedDeviceID struct {
	Driver, Pool, Device, ShareID draapi.UniqueString
}

func MakeSharedDeviceID(deviceID DeviceID, shareID string) SharedDeviceID {
	return SharedDeviceID{
		Driver:  deviceID.Driver,
		Pool:    deviceID.Pool,
		Device:  deviceID.Device,
		ShareID: draapi.MakeUniqueString(shareID),
	}
}

func (d SharedDeviceID) String() string {
	return d.Driver.String() + "/" + d.Pool.String() + "/" + d.Device.String() + "/" + d.ShareID.String()
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
	allocatedCapacity := make(ConsumedCapacity)
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

type UniqueHexStringFactory struct {
	mu      sync.Mutex
	usedIDs sets.Set[SharedDeviceID]
	nBytes  int
}

func NewUniqueHexStringFactory(nBytes int) *UniqueHexStringFactory {
	return &UniqueHexStringFactory{
		usedIDs: sets.New[SharedDeviceID](),
		nBytes:  nBytes,
	}
}

func (f *UniqueHexStringFactory) SetUsedShareIDs(usedIDs sets.Set[SharedDeviceID]) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if usedIDs != nil {
		f.usedIDs = usedIDs
	}
}

// GenerateNewShareID generates a new random hexadecimal string of length nBytes*2.
// It combines the generated string with the given driver, pool, and device identifiers
// to form a composite key, ensuring uniqueness within the factory's usedIDs map.
//
// The function attempts up to maxTry times to generate a unique ID. If a unique ID
// is found, it is added to the usedIDs map and returned. If all attempts fail,
// an error is returned.
func (f *UniqueHexStringFactory) GenerateNewShareID(deviceID DeviceID, maxTry int) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	count := 0
	for {
		b := make([]byte, f.nBytes)
		_, err := rand.Read(b)
		if err != nil {
			return "", fmt.Errorf("failed to generate random bytes: %w", err)
		}
		ShareID := hex.EncodeToString(b)
		sharedDeviceID := MakeSharedDeviceID(deviceID, ShareID)
		if _, exists := f.usedIDs[sharedDeviceID]; !exists {
			f.usedIDs[sharedDeviceID] = struct{}{} // Mark UID as used
			return ShareID, nil
		}
		count += 1
		if count > maxTry {
			return "", fmt.Errorf("failed to find unique hex string within %d try", maxTry)
		}
	}
}

func (f *UniqueHexStringFactory) DeleteShareID(deviceID DeviceID, shareID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	sharedDeviceID := MakeSharedDeviceID(deviceID, shareID)
	if _, exists := f.usedIDs[sharedDeviceID]; !exists {
		delete(f.usedIDs, sharedDeviceID)
	}
}
