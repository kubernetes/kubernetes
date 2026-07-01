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
	"math"
	"testing"

	. "github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	driverA   = "driver-a"
	pool1     = "pool-1"
	device1   = "device-1"
	capacity0 = "capacity-0"
	capacity1 = "capacity-1"
)

var (
	one   = resource.MustParse("1")
	two   = resource.MustParse("2")
	three = resource.MustParse("3")

	pointTwoFive = resource.MustParse("250m")
	pointFour    = resource.MustParse("400m")
	pointThree   = resource.MustParse("300m")
	pointTwo     = resource.MustParse("200m")
	pointOne     = resource.MustParse("100m")

	// tooBigForMilli is a value whose MilliValue() overflows int64 (> MaxInt64/1000).
	// resource.MustParse uses DecimalSI by default for large integers.
	tooBigForMilli = resource.MustParse("9224372036854776E3") // ~9.22e21, well above MaxInt64 (9.22e18)
)

func deviceConsumedCapacity(deviceID DeviceID) DeviceConsumedCapacity {
	capaicty := map[resourceapi.QualifiedName]resource.Quantity{
		capacity0: one,
	}
	return NewDeviceConsumedCapacity(deviceID, capaicty)
}

func TestConsumableCapacity(t *testing.T) {

	t.Run("add-sub-allocating-consumed-capacity", func(t *testing.T) {
		g := NewWithT(t)
		allocatedCapacity := NewConsumedCapacity()
		g.Expect(allocatedCapacity.Empty()).To(BeTrueBecause("allocated capacity should start from zero"))
		oneAllocated := ConsumedCapacity{
			capacity0: &one,
		}
		allocatedCapacity.Add(oneAllocated)
		g.Expect(allocatedCapacity.Empty()).To(BeFalseBecause("capacity is added"))
		allocatedCapacity.Sub(oneAllocated)
		g.Expect(allocatedCapacity.Empty()).To(BeTrueBecause("capacity is subtracted to zero"))
	})

	t.Run("insert-remove-allocating-consumed-capacity-collection", func(t *testing.T) {
		g := NewWithT(t)
		deviceID := MakeDeviceID(driverA, pool1, device1)
		aggregatedCapacity := NewConsumedCapacityCollection()
		aggregatedCapacity.Insert(deviceConsumedCapacity(deviceID))
		aggregatedCapacity.Insert(deviceConsumedCapacity(deviceID))
		allocatedCapacity, found := aggregatedCapacity[deviceID]
		g.Expect(found).To(BeTrueBecause("expected deviceID to be found"))
		g.Expect(allocatedCapacity[capacity0].Cmp(two)).To(BeZero())
		aggregatedCapacity.Remove(deviceConsumedCapacity(deviceID))
		g.Expect(allocatedCapacity[capacity0].Cmp(one)).To(BeZero())
	})

	t.Run("get-consumed-capacity-from-request", func(t *testing.T) {
		requestedCapacity := &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]resource.Quantity{
				capacity0: one,
				"dummy":   one,
			},
		}
		consumableCapacity := map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			capacity0: { // with request and with default, expect requested value
				Value: two,
				RequestPolicy: &resourceapi.CapacityRequestPolicy{
					Default:    &two,
					ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one},
				},
			},
			capacity1: { // no request but with default, expect default
				Value: two,
				RequestPolicy: &resourceapi.CapacityRequestPolicy{
					Default:    &one,
					ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one},
				},
			},
			"dummy": {
				Value: one, // no request and no policy (no default), expect capacity value
			},
		}
		consumedCapacity := GetConsumedCapacityFromRequest(requestedCapacity, consumableCapacity, false)
		g := NewWithT(t)
		g.Expect(consumedCapacity).To(HaveLen(3))
		for name, val := range consumedCapacity {
			g.Expect(string(name)).Should(BeElementOf([]string{capacity0, capacity1, "dummy"}))
			g.Expect(val.Cmp(one)).To(BeZero())
		}
	})

	t.Run("violate-capacity-sharing", testViolateCapacityRequestPolicy)

	t.Run("calculate-consumed-capacity", testCalculateConsumedCapacity)

	t.Run("safe-milli-value", testSafeMilliValue)

	t.Run("use-milli", testUseMilli)
}

func testSafeMilliValue(t *testing.T) {
	testcases := map[string]struct {
		q           resource.Quantity
		expectMilli int64
		expectErr   bool
	}{
		"whole number": {
			q:           one,
			expectMilli: 1000,
		},
		"fractional milli": {
			q:           pointOne,
			expectMilli: 100,
		},
		"too large for milli (overflows int64)": {
			q:         tooBigForMilli,
			expectErr: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			got, err := safeMilliValue(tc.q)
			if tc.expectErr {
				g.Expect(err).To(HaveOccurred())
			} else {
				g.Expect(err).NotTo(HaveOccurred())
				g.Expect(got).To(Equal(tc.expectMilli))
			}
		})
	}
}

func testUseMilli(t *testing.T) {
	testcases := map[string]struct {
		validRange              resourceapi.CapacityRequestPolicyRange
		fractionalCapacityRange bool
		expectUseMilli          bool
	}{
		"fractional disabled": {
			validRange: resourceapi.CapacityRequestPolicyRange{
				Min: &pointTwo, Max: &one, Step: &pointOne,
			},
			fractionalCapacityRange: false,
			expectUseMilli:          false,
		},
		"all fractional, fits in milli": {
			validRange: resourceapi.CapacityRequestPolicyRange{
				Min: &pointTwo, Max: &one, Step: &pointOne,
			},
			fractionalCapacityRange: true,
			expectUseMilli:          true,
		},
		"all integer, fits in milli (no hasFractional gate any more)": {
			validRange: resourceapi.CapacityRequestPolicyRange{
				Min: &one, Max: &three, Step: &one,
			},
			fractionalCapacityRange: true,
			expectUseMilli:          true,
		},
		"min overflows milli, falls back to integer": {
			validRange: resourceapi.CapacityRequestPolicyRange{
				Min: &tooBigForMilli, Step: &one,
			},
			fractionalCapacityRange: true,
			expectUseMilli:          false,
		},
		"step too small (sub-milli)": {
			// 1u = 1 micro = 0.001m; safeMilliValue succeeds (rounds to 0) but step<1m
			validRange: resourceapi.CapacityRequestPolicyRange{
				Min:  &one,
				Step: func() *resource.Quantity { q := resource.MustParse("0"); return &q }(),
			},
			fractionalCapacityRange: true,
			expectUseMilli:          false,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			got := useMilli(&tc.validRange, tc.fractionalCapacityRange)
			g.Expect(got).To(Equal(tc.expectUseMilli))
		})
	}
}

func testViolateCapacityRequestPolicy(t *testing.T) {
	testcases := map[string]struct {
		requestedVal            resource.Quantity
		requestPolicy           *resourceapi.CapacityRequestPolicy
		fractionalCapacityRange bool
		expectResult            bool
	}{
		"no constraint": {requestedVal: one, expectResult: false},
		"less than maximum": {
			requestedVal: one,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default:    &one,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one, Max: &two},
			},
			expectResult: false,
		},
		"more than maximum": {
			requestedVal: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default:    &one,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one, Max: &one},
			},
			expectResult: true,
		},
		"in set": {
			requestedVal: one,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default:     &one,
				ValidValues: []resource.Quantity{one},
			},
			expectResult: false,
		},
		"not in set": {
			requestedVal: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default:     &one,
				ValidValues: []resource.Quantity{one},
			},
			expectResult: true,
		},
		// fractional step: min=0.2, step=0.1, max=1
		"fractional step aligned (0.3 = min+1*step)": {
			requestedVal: pointThree,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointTwo,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointTwo,
					Max:  &one,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            false,
		},
		"fractional step not aligned (0.25 is not a multiple of 0.1 from 0.2)": {
			requestedVal: pointTwoFive,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointTwo,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointTwo,
					Max:  &one,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            true,
		},
		// requested value cannot be losslessly converted to milli, treated as violation.
		// Range values are all milli-representable (no Max) so useMilli returns true; the
		// oversized requested value then hits the safeMilliValue error path in violateValidRange.
		"requested value overflows milli representation": {
			requestedVal: tooBigForMilli,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &one,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &one,
					Step: &one,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			violate := violatesPolicy(tc.requestedVal, tc.requestPolicy, tc.fractionalCapacityRange)
			g.Expect(violate).To(BeEquivalentTo(tc.expectResult))
		})
	}
}

func testCalculateConsumedCapacity(t *testing.T) {
	testcases := map[string]struct {
		requestedVal            *resource.Quantity
		capacityValue           resource.Quantity
		requestPolicy           *resourceapi.CapacityRequestPolicy
		fractionalCapacityRange bool
		expectResult            resource.Quantity
	}{
		"empty": {requestedVal: nil, capacityValue: one, requestPolicy: &resourceapi.CapacityRequestPolicy{}, expectResult: one},
		"min in range": {
			requestedVal:  nil,
			capacityValue: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one}},
			expectResult:  one,
		},
		"default in set": {
			requestedVal:  nil,
			capacityValue: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidValues: []resource.Quantity{one}},
			expectResult:  one,
		},
		"more than min in range": {
			requestedVal:  &two,
			capacityValue: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one}},
			expectResult:  two,
		},
		"less than min in range": {
			requestedVal:  &one,
			capacityValue: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &two}},
			expectResult:  two,
		},
		"with step (round up)": {
			requestedVal:  &two,
			capacityValue: three,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one, Step: &two}},
			expectResult:  three,
		},
		"with step (no remaining)": {
			requestedVal:  &two,
			capacityValue: two,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: &one, Step: &one}},
			expectResult:  two,
		},
		// fractional step: min=0.2, step=0.1, max=1; request=0.25; rounds up to 0.3
		"fractional step round up (0.25 to 0.3)": {
			requestedVal:  &pointTwoFive,
			capacityValue: resource.MustParse("1"),
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointTwo,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointTwo,
					Max:  &one,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            resource.MustParse("300m"),
		},
		// fractional step: request already aligned; no rounding
		"fractional step already aligned (0.4 = min+2*step)": {
			requestedVal:  &pointFour,
			capacityValue: resource.MustParse("1"),
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointTwo,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointTwo,
					Max:  &one,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            resource.MustParse("400m"),
		},
		"valid value in set": {
			requestedVal:  &two,
			capacityValue: three,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidValues: []resource.Quantity{one, two, three}},
			expectResult:  two,
		},
		"set (round up)": {
			requestedVal:  &two,
			capacityValue: three,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidValues: []resource.Quantity{one, three}},
			expectResult:  three,
		},
		"larger than set": {
			requestedVal:  &three,
			capacityValue: three,
			requestPolicy: &resourceapi.CapacityRequestPolicy{Default: &one, ValidValues: []resource.Quantity{one, two}},
			expectResult:  three,
		},
		// overflow guard: min=1, step=1, requested is huge; rounding n overflows, caps at MaxInt64m
		"fractional step overflow cap rounds to MaxInt64 milli": {
			requestedVal:  &tooBigForMilli,
			capacityValue: tooBigForMilli,
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &one,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointOne,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			// requested can't be converted to milli safely, so roundUpRange returns
			// requestedVal unchanged; calculateConsumedCapacity uses that as-is.
			expectResult: tooBigForMilli,
		},
		// overflow guard via large but milli-representable requested value.
		// min=100m, step=100m, request=MaxInt64-1 milli:
		//   added = MaxInt64-1 - 100 = MaxInt64-101
		//   n     = (MaxInt64-101) / 100 = 92233720368547757  (added%100 == 6, so n++)
		//   n     = 92233720368547758
		//   guard = (MaxInt64-100)/100 = 92233720368547757  → n > guard → cap at MaxInt64m
		"fractional step overflow cap: large milli-representable request capped at MaxInt64m": {
			requestedVal: func() *resource.Quantity {
				q := resource.NewMilliQuantity(math.MaxInt64-1, resource.DecimalSI)
				return q
			}(),
			capacityValue: *resource.NewMilliQuantity(math.MaxInt64, resource.DecimalSI),
			requestPolicy: &resourceapi.CapacityRequestPolicy{
				Default: &pointOne,
				ValidRange: &resourceapi.CapacityRequestPolicyRange{
					Min:  &pointOne,
					Step: &pointOne,
				},
			},
			fractionalCapacityRange: true,
			expectResult:            *resource.NewMilliQuantity(math.MaxInt64, resource.DecimalSI),
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			capacity := resourceapi.DeviceCapacity{
				Value:         tc.capacityValue,
				RequestPolicy: tc.requestPolicy,
			}
			consumedCapacity := calculateConsumedCapacity(tc.requestedVal, capacity, tc.fractionalCapacityRange)
			g.Expect(consumedCapacity.Cmp(tc.expectResult)).To(BeZero())
		})
	}
}
