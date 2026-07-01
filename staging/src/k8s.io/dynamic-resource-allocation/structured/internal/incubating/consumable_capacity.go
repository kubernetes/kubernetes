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

package incubating

import (
	"errors"
	"math"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

// CmpRequestOverCapacity checks whether the new capacity request can be added within the given capacity,
// and checks whether the requested value is against the capacity requestPolicy.
func CmpRequestOverCapacity(currentConsumedCapacity ConsumedCapacity, deviceRequestCapacity *resourceapi.CapacityRequirements,
	allowMultipleAllocations *bool, capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, allocatingCapacity ConsumedCapacity, fractionalCapacityRange bool) (bool, error) {
	if requestsContainNonExistCapacity(deviceRequestCapacity, capacity) {
		return false, errors.New("some requested capacity has not been defined")
	}
	clone := currentConsumedCapacity.Clone()
	for name, cap := range capacity {
		var requestedValPtr *resource.Quantity
		if deviceRequestCapacity != nil && deviceRequestCapacity.Requests != nil {
			if requestedVal, requestedFound := deviceRequestCapacity.Requests[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		consumedCapacity := calculateConsumedCapacity(requestedValPtr, cap, fractionalCapacityRange)
		if violatesPolicy(consumedCapacity, cap.RequestPolicy, fractionalCapacityRange) {
			return false, nil
		}
		// If the current clone already contains an entry for this capacity, add the consumedCapacity to it.
		// Otherwise, initialize it with calculated consumedCapacity.
		if _, allocatedFound := clone[name]; allocatedFound {
			clone[name].Add(consumedCapacity)
		} else {
			clone[name] = ptr.To(consumedCapacity)
		}
		// If allocatingCapacity contains an entry for this capacity, add its value to clone as well.
		if allocatingVal, allocatingFound := allocatingCapacity[name]; allocatingFound {
			clone[name].Add(*allocatingVal)
		}
		if clone[name].Cmp(cap.Value) > 0 {
			return false, nil
		}
	}
	return true, nil
}

// requestsNonExistCapacity returns true if requests contain non-exist capacity.
func requestsContainNonExistCapacity(deviceRequestCapacity *resourceapi.CapacityRequirements,
	capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity) bool {
	if deviceRequestCapacity == nil || deviceRequestCapacity.Requests == nil {
		return false
	}
	for name := range deviceRequestCapacity.Requests {
		if _, found := capacity[name]; !found {
			return true
		}
	}
	return false
}

// calculateConsumedCapacity returns valid capacity to be consumed regarding the requested capacity and device capacity policy.
//
// If no requestPolicy, return capacity.Value.
// If no requestVal, fill the quantity by fillEmptyRequest function
// Otherwise, use requestPolicy to calculate the consumed capacity from request if applicable.
func calculateConsumedCapacity(requestedVal *resource.Quantity, capacity resourceapi.DeviceCapacity, fractionalCapacityRange bool) resource.Quantity {
	if requestedVal == nil {
		return fillEmptyRequest(capacity)
	}
	if capacity.RequestPolicy == nil {
		return requestedVal.DeepCopy()
	}
	switch {
	case capacity.RequestPolicy.ValidRange != nil && capacity.RequestPolicy.ValidRange.Min != nil:
		return roundUpRange(requestedVal, capacity.RequestPolicy.ValidRange, fractionalCapacityRange)
	case capacity.RequestPolicy.ValidValues != nil:
		return roundUpValidValues(requestedVal, capacity.RequestPolicy.ValidValues)
	}
	return *requestedVal
}

// fillEmptyRequest
// return requestPolicy.default if defined.
// Otherwise, return capacity value.
func fillEmptyRequest(capacity resourceapi.DeviceCapacity) resource.Quantity {
	if capacity.RequestPolicy != nil && capacity.RequestPolicy.Default != nil {
		return capacity.RequestPolicy.Default.DeepCopy()
	}
	return capacity.Value.DeepCopy()
}

// roundUpRange rounds the requestedVal up to fit within the specified validRange.
//   - If requestedVal is less than Min, it returns Min.
//   - If Step is specified, it rounds requestedVal up to the nearest multiple of Step
//     starting from Min.
//   - If no Step is specified and requestedVal >= Min, it returns requestedVal as is.
//
// When fractionalCapacityRange is true and any of min/max/step are fractional and all
// fit within the milli-value int64 range and step >= 1m, milli-value arithmetic is used.
// Otherwise Value() arithmetic is used.
//
// If the rounded milli-value exceeds the int64 milli-value range, the result is
// capped at the maximum representable milli value.
func roundUpRange(requestedVal *resource.Quantity, validRange *resourceapi.CapacityRequestPolicyRange, fractionalCapacityRange bool) resource.Quantity {
	if requestedVal.Cmp(*validRange.Min) < 0 {
		return validRange.Min.DeepCopy()
	}
	if validRange.Step == nil {
		return *requestedVal
	}
	if useMilli(validRange, fractionalCapacityRange) {
		requestedMilli, err := safeMilliValue(*requestedVal)
		format := validRange.Step.Format
		if err != nil {
			// This is violated value.
			// It will be rejected by violateValidRange check
			return *requestedVal
		}
		stepMilli := validRange.Step.MilliValue()
		minMilli := validRange.Min.MilliValue()
		added := requestedMilli - minMilli
		n := added / stepMilli
		if added%stepMilli != 0 {
			n++
		}
		if n > (math.MaxInt64-minMilli)/stepMilli {
			// Round to maximum value
			return *resource.NewMilliQuantity(math.MaxInt64, format)
		}
		valMilli := minMilli + stepMilli*n
		// Return in the same format as the step quantity. If the result is a
		// whole number, use NewQuantity to keep the representation compact and
		// compatible with quantities parsed from whole-number strings.
		if valMilli%1000 == 0 {
			return *resource.NewQuantity(valMilli/1000, format)
		}
		return *resource.NewMilliQuantity(valMilli, format)
	}
	// Integer arithmetic path.
	requestedInt := requestedVal.Value()
	stepInt := validRange.Step.Value()
	minInt := validRange.Min.Value()
	added := requestedInt - minInt
	n := added / stepInt
	if added%stepInt != 0 {
		n++
	}
	// TODO (#140441): minInt+stepInt*n can overflow int64 for a large capacity request,
	// and allocate a device which it should not.
	return *resource.NewQuantity(minInt+stepInt*n, validRange.Step.Format)
}

// roundUpValidValues returns the first value in validValues that is greater than or equal to requestedVal.
// If no such value exists, it returns requestedVal itself.
func roundUpValidValues(requestedVal *resource.Quantity, validValues []resource.Quantity) resource.Quantity {
	// Simple sequential search is used as the maximum entry of validValues is finite and small (≤10),
	// and the list must already be sorted in ascending order, ensured by API validation.
	// Note: A binary search could alternatively be used for better efficiency if the list grows larger.
	for _, validValue := range validValues {
		if requestedVal.Cmp(validValue) <= 0 {
			return validValue.DeepCopy()
		}
	}
	return *requestedVal
}

// GetConsumedCapacityFromRequest returns valid consumed capacity,
// according to claim request and defined capacity.
func GetConsumedCapacityFromRequest(requestedCapacity *resourceapi.CapacityRequirements,
	consumableCapacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, fractionalCapacityRange bool) map[resourceapi.QualifiedName]resource.Quantity {
	consumedCapacity := make(map[resourceapi.QualifiedName]resource.Quantity)
	for name, cap := range consumableCapacity {
		var requestedValPtr *resource.Quantity
		if requestedCapacity != nil && requestedCapacity.Requests != nil {
			if requestedVal, requestedFound := requestedCapacity.Requests[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		capacity := calculateConsumedCapacity(requestedValPtr, cap, fractionalCapacityRange)
		consumedCapacity[name] = capacity
	}
	return consumedCapacity
}

// violatesPolicy checks whether the request violate the requestPolicy.
func violatesPolicy(requestedVal resource.Quantity, policy *resourceapi.CapacityRequestPolicy, fractionalCapacityRange bool) bool {
	if policy == nil {
		// no policy to check
		return false
	}
	if policy.Default != nil && requestedVal == *policy.Default {
		return false
	}
	switch {
	case policy.ValidRange != nil:
		return violateValidRange(requestedVal, *policy.ValidRange, fractionalCapacityRange)
	case len(policy.ValidValues) > 0:
		return violateValidValues(requestedVal, policy.ValidValues)
	}
	// no policy violated through to completion.
	return false
}

func violateValidRange(requestedVal resource.Quantity, validRange resourceapi.CapacityRequestPolicyRange, fractionalCapacityRange bool) bool {
	if validRange.Max != nil &&
		requestedVal.Cmp(*validRange.Max) > 0 {
		return true
	}
	if validRange.Step != nil {
		var requested, step, min int64
		var err error
		if useMilli(&validRange, fractionalCapacityRange) {
			if requested, err = safeMilliValue(requestedVal); err != nil {
				return true
			}
			step = validRange.Step.MilliValue()
			min = validRange.Min.MilliValue()
		} else {
			requested = requestedVal.Value()
			step = validRange.Step.Value()
			min = validRange.Min.Value()
		}
		// must be a multiple of step from min
		if (requested-min)%step != 0 {
			return true
		}
	}
	return false
}

// useMilli reports whether milli-value arithmetic should be used for the given range.
// Conditions: fractionalCapacityRange enabled AND any of min/max/step is fractional
// AND all non-nil fields fit within the milli-value int64 range AND step >= 1m (i.e.
// non-zero after MilliValue()).
func useMilli(validRange *resourceapi.CapacityRequestPolicyRange, fractionalCapacityRange bool) bool {
	if !fractionalCapacityRange {
		return false
	}
	for i, q := range []*resource.Quantity{validRange.Min, validRange.Max, validRange.Step} {
		if q == nil {
			continue
		}
		if _, err := safeMilliValue(*q); err != nil {
			return false
		}
		if i == 2 {
			// Step must be at least 1m.
			if step, err := safeMilliValue(*validRange.Step); err == nil && step < 1 {
				return false
			}
		}
	}
	return true
}

func violateValidValues(requestedVal resource.Quantity, validValues []resource.Quantity) bool {
	for _, validVal := range validValues {
		if requestedVal.Cmp(validVal) == 0 {
			return false
		}
	}
	return true
}

// safeMilliValue returns q as a milli value if the conversion is lossless.
func safeMilliValue(q resource.Quantity) (int64, error) {
	milli := q.MilliValue()
	milliQuantity := resource.NewMilliQuantity(milli, q.Format)
	if q.Cmp(*milliQuantity) != 0 {
		return 0, errors.New("value must be representable in milli units")
	}
	return milli, nil
}
