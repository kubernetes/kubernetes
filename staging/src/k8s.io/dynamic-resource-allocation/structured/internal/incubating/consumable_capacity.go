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
	"fmt"
	"math"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

// errCapacityRequestNotRepresentable signals that a capacity request, or the
// value it rounds up to, cannot be represented by a device's range arithmetic
// (it passes MaxInt64 on the integer path, or the milli-value range on the
// fractional path). The allocator treats this as a fatal error that aborts the
// allocation rather than a per-device mismatch to skip, so an unrepresentable
// request is rejected outright instead of driving a device-by-device retry.
var errCapacityRequestNotRepresentable = errors.New("capacity request cannot be represented in the device's range arithmetic")

// errNegativeCapacity signals that a resolved consumed capacity is negative. A negative
// value is not valid capacity and would under-count the device and over-allocate it. It
// can reach the allocator from a stored ResourceSlice or ResourceClaim created before
// stricter validation, which admission does not re-check, so the allocator rejects it
// rather than recording it.
var errNegativeCapacity = errors.New("capacity value is negative")

// CmpRequestOverCapacity checks whether the new capacity request can be added within the given capacity,
// and checks whether the requested value is against the capacity requestPolicy.
func CmpRequestOverCapacity(currentConsumedCapacity ConsumedCapacity, deviceRequestCapacity *resourceapi.CapacityRequirements,
	allowMultipleAllocations *bool, capacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, allocatingCapacity ConsumedCapacity, fractionalCapacityRange bool) (bool, error) {
	if requestsContainNonExistCapacity(deviceRequestCapacity, capacity) {
		// This device does not define a requested capacity. A different device may,
		// so report it as not satisfiable rather than a fatal error: false with a nil
		// error means "skip this device", a non-nil error means "abort allocation".
		return false, nil
	}
	// First resolve every requested capacity. A request that cannot be represented is a
	// fatal error that must abort regardless of map iteration order, so representability
	// is checked for all capacities before any soft "not satisfiable" return below.
	// Interleaving the two in one loop would let the outcome, skip or abort, depend on
	// which capacity the map happens to visit first.
	consumed := make(map[resourceapi.QualifiedName]resource.Quantity, len(capacity))
	for name, cap := range capacity {
		var requestedValPtr *resource.Quantity
		if deviceRequestCapacity != nil && deviceRequestCapacity.Requests != nil {
			if requestedVal, requestedFound := deviceRequestCapacity.Requests[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		consumedCapacity, err := calculateConsumedCapacity(requestedValPtr, cap, fractionalCapacityRange)
		if err != nil {
			// The request is not representable in this capacity's range arithmetic.
			// Surface a fatal error so the allocator aborts instead of retrying other
			// devices, which a user-controlled out-of-range request could abuse.
			return false, fmt.Errorf("capacity %q: %w", name, err)
		}
		if consumedCapacity.Sign() < 0 {
			// A negative capacity cannot come from a request the allocator should honor;
			// recording it would over-allocate the device. Abort rather than skip, since a
			// stored object from before stricter validation is a data problem, not a
			// per-device mismatch that another device could satisfy.
			return false, fmt.Errorf("capacity %q: %w", name, errNegativeCapacity)
		}
		consumed[name] = consumedCapacity
	}
	// The policy and capacity checks are soft: they skip this device rather than abort.
	clone := currentConsumedCapacity.Clone()
	for name, cap := range capacity {
		consumedCapacity := consumed[name]
		if violatesPolicy(consumedCapacity, cap.RequestPolicy, fractionalCapacityRange) {
			return false, nil
		}
		// If the current clone already contains an entry for this capacity, add the consumedCapacity to it.
		// Otherwise, initialize it with calculated consumedCapacity.
		if _, allocatedFound := clone[name]; allocatedFound {
			clone[name].Add(consumedCapacity)
		} else {
			clone[name] = ptr.To(consumedCapacity.DeepCopy())
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
func calculateConsumedCapacity(requestedVal *resource.Quantity, capacity resourceapi.DeviceCapacity, fractionalCapacityRange bool) (resource.Quantity, error) {
	if requestedVal == nil {
		return fillEmptyRequest(capacity), nil
	}
	if capacity.RequestPolicy == nil {
		return requestedVal.DeepCopy(), nil
	}
	switch {
	case capacity.RequestPolicy.ValidRange != nil && capacity.RequestPolicy.ValidRange.Min != nil:
		return roundUpRange(requestedVal, capacity.RequestPolicy.ValidRange, fractionalCapacityRange)
	case capacity.RequestPolicy.ValidValues != nil:
		return roundUpValidValues(requestedVal, capacity.RequestPolicy.ValidValues), nil
	}
	return *requestedVal, nil
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
// It returns errCapacityRequestNotRepresentable when the request, or the value it
// rounds up to, cannot be represented: past MaxInt64 on the integer path, or past
// the milli-value range on the fractional path. The caller aborts allocation with
// that error rather than acting on a wrapped read or a silently capped value.
func roundUpRange(requestedVal *resource.Quantity, validRange *resourceapi.CapacityRequestPolicyRange, fractionalCapacityRange bool) (resource.Quantity, error) {
	if requestedVal.Cmp(*validRange.Min) < 0 {
		return validRange.Min.DeepCopy(), nil
	}
	if validRange.Step == nil {
		return *requestedVal, nil
	}
	if useMilli(validRange, fractionalCapacityRange) {
		requestedMilli, err := safeMilliValue(*requestedVal)
		format := validRange.Step.Format
		if err != nil {
			return resource.Quantity{}, fmt.Errorf("%w: request %s is not a representable milli value", errCapacityRequestNotRepresentable, requestedVal.String())
		}
		stepMilli := validRange.Step.MilliValue()
		minMilli := validRange.Min.MilliValue()
		added := requestedMilli - minMilli
		n := added / stepMilli
		if added%stepMilli != 0 {
			n++
		}
		if n > (math.MaxInt64-minMilli)/stepMilli {
			return resource.Quantity{}, fmt.Errorf("%w: rounding request %s up to the next step passes the milli-value range", errCapacityRequestNotRepresentable, requestedVal.String())
		}
		valMilli := minMilli + stepMilli*n
		// Return in the same format as the step quantity. If the result is a
		// whole number, use NewQuantity to keep the representation compact and
		// compatible with quantities parsed from whole-number strings.
		if valMilli%1000 == 0 {
			return *resource.NewQuantity(valMilli/1000, format), nil
		}
		return *resource.NewMilliQuantity(valMilli, format), nil
	}
	// Integer arithmetic path.
	// A request above MaxInt64 cannot be read with Value() without wrapping, so it
	// cannot be rounded in int64. Report that it is not representable rather than
	// acting on the wrapped read.
	if requestedVal.CmpInt64(math.MaxInt64) > 0 {
		return resource.Quantity{}, fmt.Errorf("%w: request %s exceeds MaxInt64", errCapacityRequestNotRepresentable, requestedVal.String())
	}
	requestedInt := requestedVal.Value()
	stepInt := validRange.Step.Value()
	minInt := validRange.Min.Value()
	added := requestedInt - minInt
	n := added / stepInt
	if added%stepInt != 0 {
		n++
	}
	// minInt+stepInt*n overflows int64 once the rounded value passes MaxInt64. It cannot
	// be represented or compared reliably, so report the request as not representable. The
	// guard needs minInt >= 0 so that MaxInt64-minInt does not itself overflow; a negative
	// min is out of the non-negative API contract and keeps the existing arithmetic here.
	if minInt >= 0 && n > (math.MaxInt64-minInt)/stepInt {
		return resource.Quantity{}, fmt.Errorf("%w: rounding request %s up to the next step passes MaxInt64", errCapacityRequestNotRepresentable, requestedVal.String())
	}
	return *resource.NewQuantity(minInt+stepInt*n, validRange.Step.Format), nil
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
	consumableCapacity map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, fractionalCapacityRange bool) (map[resourceapi.QualifiedName]resource.Quantity, error) {
	consumedCapacity := make(map[resourceapi.QualifiedName]resource.Quantity)
	for name, cap := range consumableCapacity {
		var requestedValPtr *resource.Quantity
		if requestedCapacity != nil && requestedCapacity.Requests != nil {
			if requestedVal, requestedFound := requestedCapacity.Requests[name]; requestedFound {
				requestedValPtr = &requestedVal
			}
		}
		capacity, err := calculateConsumedCapacity(requestedValPtr, cap, fractionalCapacityRange)
		if err != nil {
			// Feasibility already ran, so this should not happen. Return the error
			// rather than a raw request so a future caller cannot record an
			// unrepresentable value as consumed capacity.
			return nil, fmt.Errorf("capacity %q: %w", name, err)
		}
		consumedCapacity[name] = capacity
	}
	return consumedCapacity, nil
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
