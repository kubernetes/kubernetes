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

package structured

import (
	"errors"

	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
)

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

// CmpRequestOverCapacity checks whether the new request can be added within the given capacity,
// and checks whether the requested value is against the capacity sharing policy.
func (s ConsumedCapacity) CmpRequestOverCapacity(request *resourceapi.DeviceRequest,
	capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, allocatingCapacity *ConsumedCapacity) (bool, error) {
	if requestsContainNonExistCapacity(request, capacity) {
		return false, errors.New("some requested capacity has not been defined")
	}
	clone := s.Clone()
	for name, cap := range capacity {
		var requestedValPtr *resource.Quantity
		if request.CapacityRequests != nil && request.CapacityRequests.Minimum != nil {
			if requestedVal, requestedFound := request.CapacityRequests.Minimum[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		if isConsumableCapacity(cap) {
			consumedCapacity := calculateConsumedCapacity(requestedValPtr, *cap.SharingPolicy)
			if violatePolicy(*consumedCapacity, cap.SharingPolicy) {
				return false, nil
			}
			_, allocatedFound := clone[name]
			if !allocatedFound {
				clone[name] = consumedCapacity
			} else {
				clone[name].Add(*consumedCapacity)
			}
			if allocatingCapacity != nil {
				if allocatingVal, allocatingFound := (*allocatingCapacity)[name]; allocatingFound {
					clone[name].Add(*allocatingVal)
				}
			}
			if clone[name].Cmp(cap.Value) > 0 {
				return false, nil
			}
		} else if requestedValPtr != nil {
			if (*requestedValPtr).Cmp(cap.Value) > 0 {
				return false, nil
			}
		}
	}
	return true, nil
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
	clone := cap.ConsumedCapacity.Clone()
	if _, found := c[cap.DeviceID]; found {
		c[cap.DeviceID].Add(clone)
	} else {
		c[cap.DeviceID] = clone
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

// requestsNonExistCapacity returns true if requests contain non-exist capacity.
func requestsContainNonExistCapacity(request *resourceapi.DeviceRequest,
	capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity) bool {
	if request.CapacityRequests == nil || request.CapacityRequests.Minimum == nil {
		return false
	}
	for name := range request.CapacityRequests.Minimum {
		if _, found := capacity[name]; !found {
			return true
		}
	}
	return false
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

// isConsumableCapacity returns true if capacity has consumable spec defined.
func isConsumableCapacity(cap resourceapi.DeviceCapacity) bool {
	return cap.SharingPolicy != nil
}

// calculateConsumedCapacity returns valid capacity to be consumed regarding the requested capacity and consumable spec.
func calculateConsumedCapacity(requestedVal *resource.Quantity, consumable resourceapi.CapacitySharingPolicy) *resource.Quantity {
	if consumable.ValidRange != nil {
		if requestedVal == nil {
			returnedVal := consumable.Default.DeepCopy()
			return &returnedVal
		}
		if requestedVal.Cmp(consumable.ValidRange.Minimum) < 0 {
			returnedVal := consumable.ValidRange.Minimum.DeepCopy()
			return &returnedVal
		}
		if consumable.ValidRange.ChunkSize != nil {
			requestedInt64 := requestedVal.Value()
			chunksize := consumable.ValidRange.ChunkSize.Value()
			min := consumable.ValidRange.Minimum.Value()
			added := (requestedInt64 - min)
			n := added / chunksize
			mod := added % chunksize
			if mod != 0 {
				n += 1
			}
			val := min + chunksize*n
			return resource.NewQuantity(val, resource.BinarySI)
		}
	} else if consumable.ValidValues != nil {
		if requestedVal == nil {
			returnedVal := consumable.Default.DeepCopy()
			return &returnedVal
		}
	}
	return requestedVal
}

// GetConsumedCapacityFromRequest returns valid consumed capacity,
// according to claim request and defined capacity.
func GetConsumedCapacityFromRequest(requestedCapacity *resourceapi.CapacityRequirements,
	consumableCapacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity) map[resourceapi.QualifiedName]resource.Quantity {
	consumedCapacities := make(map[resourceapi.QualifiedName]resource.Quantity)
	for name, cap := range consumableCapacity {
		if isConsumableCapacity(cap) {
			var requestedValPtr *resource.Quantity
			if requestedCapacity != nil && requestedCapacity.Minimum != nil {
				if requestedVal, requestedFound := requestedCapacity.Minimum[name]; requestedFound {
					requestedValPtr = &requestedVal
				}
			}
			consumedCapacity := calculateConsumedCapacity(requestedValPtr, *cap.SharingPolicy)
			consumedCapacities[name] = *consumedCapacity
		}
	}
	return consumedCapacities
}

// violatePolicy checks whether the request violate the sharing policy.
func violatePolicy(requestedVal resource.Quantity, policy *resourceapi.CapacitySharingPolicy) bool {
	if policy == nil {
		return false
	}
	if requestedVal == policy.Default {
		return false
	}
	if policy.ValidRange != nil {
		if policy.ValidRange.Maximum != nil &&
			requestedVal.Cmp(*policy.ValidRange.Maximum) > 0 {
			return true
		}
		if policy.ValidRange.ChunkSize != nil {
			requestedInt64 := requestedVal.Value()
			chunksize := policy.ValidRange.ChunkSize.Value()
			min := policy.ValidRange.Minimum.Value()
			added := (requestedInt64 - min)
			mod := added % chunksize
			if mod != 0 {
				return true
			}
		}
		return false
	}
	for _, validVal := range policy.ValidValues {
		if requestedVal.Cmp(validVal) == 0 {
			return false
		}
	}
	return true
}
