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
	"math"

	inf "gopkg.in/inf.v0"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

// CmpRequestOverCapacity checks whether the new capacity request can be added within the given capacity,
// and checks whether the requested value is against the capacity requestPolicy.
func CmpRequestOverCapacity(currentConsumedCapacity ConsumedCapacity, deviceRequestCapacity *resourceapi.CapacityRequirements,
	allowMultipleAllocations *bool, capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, allocatingCapacity ConsumedCapacity) (bool, error) {
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
		consumedCapacity := calculateConsumedCapacity(requestedValPtr, cap)
		if violatesPolicy(consumedCapacity, cap.RequestPolicy) {
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
func calculateConsumedCapacity(requestedVal *resource.Quantity, capacity resourceapi.DeviceCapacity) resource.Quantity {
	if requestedVal == nil {
		return fillEmptyRequest(capacity)
	}
	if capacity.RequestPolicy == nil {
		return requestedVal.DeepCopy()
	}
	switch {
	case capacity.RequestPolicy.ValidRange != nil && capacity.RequestPolicy.ValidRange.Min != nil:
		return roundUpRange(requestedVal, capacity.RequestPolicy.ValidRange)
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
func roundUpRange(requestedVal *resource.Quantity, validRange *resourceapi.CapacityRequestPolicyRange) resource.Quantity {
	if requestedVal.Cmp(*validRange.Min) < 0 {
		return validRange.Min.DeepCopy()
	}
	if validRange.Step == nil {
		return *requestedVal
	}
	// Integer arithmetic fast path, guarded against int64 overflow and against a
	// step whose Value() is 0 (a quantity larger than MaxInt64, for example "100E",
	// which would otherwise divide by zero). Fall back to exact arithmetic when a
	// value does not fit int64 or when min+step*n would overflow.
	if fitsInt64Value(requestedVal) && fitsInt64Value(validRange.Min) && fitsInt64Value(validRange.Step) {
		if step := validRange.Step.Value(); step > 0 {
			requestedInt64 := requestedVal.Value()
			min := validRange.Min.Value()
			added := requestedInt64 - min
			n := added / step
			if added%step != 0 {
				n++
			}
			if val, ok := safeMinPlusStepN(min, step, n); ok {
				return *resource.NewQuantity(val, resource.BinarySI)
			}
		}
	}
	return roundUpRangeArbitrary(requestedVal, validRange.Min, validRange.Step)
}

// fitsInt64Value reports whether q is non-negative and small enough that q.Value()
// is exact. A quantity larger than MaxInt64 wraps, for example "100E".Value() == 0.
func fitsInt64Value(q *resource.Quantity) bool {
	return q.Sign() >= 0 && q.CmpInt64(math.MaxInt64) <= 0
}

// safeMinPlusStepN returns base+step*n and reports whether that int64 computation
// stays within range. base, step and n are assumed non-negative.
func safeMinPlusStepN(base, step, n int64) (int64, bool) {
	if n != 0 && step > (math.MaxInt64-base)/n {
		return 0, false
	}
	return base + step*n, true
}

// roundUpRangeArbitrary rounds requestedVal up to minVal+ceil((requestedVal-minVal)/step)*step
// with arbitrary precision, for the cases the int64 fast path cannot represent
// without overflowing or dividing by zero.
func roundUpRangeArbitrary(requestedVal, minVal, step *resource.Quantity) resource.Quantity {
	added := new(inf.Dec).Sub(requestedVal.AsDec(), minVal.AsDec())
	n := new(inf.Dec).QuoRound(added, step.AsDec(), 0, inf.RoundCeil)
	result := new(inf.Dec).Add(minVal.AsDec(), new(inf.Dec).Mul(n, step.AsDec()))
	return *resource.NewDecimalQuantity(*result, step.Format)
}

// isStepMultiple reports whether requestedVal-minVal is an exact multiple of step,
// using arbitrary precision so it neither overflows nor divides by zero.
func isStepMultiple(requestedVal, minVal, step *resource.Quantity) bool {
	added := new(inf.Dec).Sub(requestedVal.AsDec(), minVal.AsDec())
	n := new(inf.Dec).QuoRound(added, step.AsDec(), 0, inf.RoundDown)
	remainder := new(inf.Dec).Sub(added, new(inf.Dec).Mul(n, step.AsDec()))
	return remainder.Sign() == 0
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
	consumableCapacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity) map[resourceapi.QualifiedName]resource.Quantity {
	consumedCapacity := make(map[resourceapi.QualifiedName]resource.Quantity)
	for name, cap := range consumableCapacity {
		var requestedValPtr *resource.Quantity
		if requestedCapacity != nil && requestedCapacity.Requests != nil {
			if requestedVal, requestedFound := requestedCapacity.Requests[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		capacity := calculateConsumedCapacity(requestedValPtr, cap)
		consumedCapacity[name] = capacity
	}
	return consumedCapacity
}

// violatesPolicy checks whether the request violate the requestPolicy.
func violatesPolicy(requestedVal resource.Quantity, policy *resourceapi.CapacityRequestPolicy) bool {
	if policy == nil {
		// no policy to check
		return false
	}
	if policy.Default != nil && requestedVal == *policy.Default {
		return false
	}
	switch {
	case policy.ValidRange != nil:
		return violateValidRange(requestedVal, *policy.ValidRange)
	case len(policy.ValidValues) > 0:
		return violateValidValues(requestedVal, policy.ValidValues)
	}
	// no policy violated through to completion.
	return false
}

func violateValidRange(requestedVal resource.Quantity, validRange resourceapi.CapacityRequestPolicyRange) bool {
	if validRange.Max != nil &&
		requestedVal.Cmp(*validRange.Max) > 0 {
		return true
	}
	if validRange.Step != nil {
		// Guard against int64 overflow and against a step whose Value() is 0 (a
		// quantity larger than MaxInt64); fall back to exact arithmetic otherwise.
		if fitsInt64Value(&requestedVal) && fitsInt64Value(validRange.Min) && fitsInt64Value(validRange.Step) {
			if step := validRange.Step.Value(); step > 0 {
				added := requestedVal.Value() - validRange.Min.Value()
				return added%step != 0
			}
		}
		// must be a multiple of step
		return !isStepMultiple(&requestedVal, validRange.Min, validRange.Step)
	}
	return false
}

func violateValidValues(requestedVal resource.Quantity, validValues []resource.Quantity) bool {
	for _, validVal := range validValues {
		if requestedVal.Cmp(validVal) == 0 {
			return false
		}
	}
	return true
}
