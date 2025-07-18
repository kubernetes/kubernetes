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

package experimental

import (
	"errors"
	"fmt"

	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/structured/internal"
)

type ConsumedCapacity = internal.ConsumedCapacity

// CmpRequestOverCapacity checks whether the new capacity request can be added within the given capacity,
// and checks whether the requested value is against the capacity sharing policy.
func CmpRequestOverCapacity(currentConsumedCapacity ConsumedCapacity, capacityRequests *resourceapi.CapacityRequirements,
	allowMultipleAllocations *bool, capacity map[draapi.QualifiedName]draapi.DeviceCapacity, allocatingCapacity ConsumedCapacity) (bool, error) {
	if requestsContainNonExistCapacity(capacityRequests, capacity) {
		return false, errors.New("some requested capacity has not been defined")
	}
	clone := currentConsumedCapacity.Clone()
	for name, cap := range capacity {
		convertedName := resourceapi.QualifiedName(name)
		var convertedCapacity resourceapi.DeviceCapacity
		err := draapi.Convert_api_DeviceCapacity_To_v1beta1_DeviceCapacity(&cap, &convertedCapacity, nil)
		if err != nil {
			return false, fmt.Errorf("failed to convert DeviceCapacity %w", err)
		}
		var requestedValPtr *resource.Quantity
		if capacityRequests != nil && capacityRequests.Requests != nil {
			if requestedVal, requestedFound := capacityRequests.Requests[convertedName]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		if isConsumableCapacity(convertedCapacity) {
			consumedCapacity := calculateConsumedCapacity(requestedValPtr, *convertedCapacity.SharingPolicy)
			if violatePolicy(*consumedCapacity, convertedCapacity.SharingPolicy) {
				return false, nil
			}
			_, allocatedFound := clone[convertedName]
			if !allocatedFound {
				clone[convertedName] = consumedCapacity
			} else {
				clone[convertedName].Add(*consumedCapacity)
			}
			if allocatingCapacity != nil {
				if allocatingVal, allocatingFound := allocatingCapacity[convertedName]; allocatingFound {
					clone[convertedName].Add(*allocatingVal)
				}
			}
			if clone[convertedName].Cmp(cap.Value) > 0 {
				return false, nil
			}
		} else if requestedValPtr != nil {
			// request resource of non-consumable capacity
			if allowMultipleAllocations != nil && *allowMultipleAllocations {
				// no guarantee
				return false, fmt.Errorf("device is allowMultipleAllocations without sharingPolicy guarantee on the requested resource")
			}
			if (*requestedValPtr).Cmp(cap.Value) > 0 {
				return false, nil
			}
		}
	}
	return true, nil
}

// requestsNonExistCapacity returns true if requests contain non-exist capacity.
func requestsContainNonExistCapacity(capacityRequests *resourceapi.CapacityRequirements,
	capacity map[draapi.QualifiedName]draapi.DeviceCapacity) bool {
	if capacityRequests == nil || capacityRequests.Requests == nil {
		return false
	}
	for name := range capacityRequests.Requests {
		convertedName := draapi.QualifiedName(name)
		if _, found := capacity[convertedName]; !found {
			return true
		}
	}
	return false
}

// isConsumableCapacity returns true if capacity has consumable spec defined.
func isConsumableCapacity(cap resourceapi.DeviceCapacity) bool {
	return cap.SharingPolicy != nil
}

// calculateConsumedCapacity returns valid capacity to be consumed regarding the requested capacity and consumable spec.
// The default consumable capacity is used if requestedValPtr is nil.
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
			if requestedCapacity != nil && requestedCapacity.Requests != nil {
				if requestedVal, requestedFound := requestedCapacity.Requests[name]; requestedFound {
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
