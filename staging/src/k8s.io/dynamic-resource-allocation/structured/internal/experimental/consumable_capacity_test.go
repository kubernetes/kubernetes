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
	"testing"

	. "github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
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
					Default:    ptr.To(two),
					ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one)},
				},
			},
			capacity1: { // no request but with default, expect default
				Value: two,
				RequestPolicy: &resourceapi.CapacityRequestPolicy{
					Default:    ptr.To(one),
					ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one)},
				},
			},
			"dummy": {
				Value: one, // no request and no policy (no default), expect capacity value
			},
		}
		consumedCapacity := GetConsumedCapacityFromRequest(requestedCapacity, consumableCapacity)
		g := NewWithT(t)
		g.Expect(consumedCapacity).To(HaveLen(3))
		for name, val := range consumedCapacity {
			g.Expect(string(name)).Should(BeElementOf([]string{capacity0, capacity1, "dummy"}))
			g.Expect(val.Cmp(one)).To(BeZero())
		}
	})

	t.Run("violate-capacity-sharing", testViolateCapacityRequestPolicy)

	t.Run("calculate-consumed-capacity", testCalculateConsumedCapacity)

}

func testViolateCapacityRequestPolicy(t *testing.T) {
	testcases := map[string]struct {
		requestedVal  resource.Quantity
		requestPolicy *resourceapi.CapacityRequestPolicy

		expectResult bool
	}{
		"no constraint": {one, nil, false},
		"less than maximum": {
			one,
			&resourceapi.CapacityRequestPolicy{
				Default:    ptr.To(one),
				ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one), Max: &two},
			},
			false,
		},
		"more than maximum": {
			two,
			&resourceapi.CapacityRequestPolicy{
				Default:    ptr.To(one),
				ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one), Max: &one},
			},
			true,
		},
		"in set": {
			one,
			&resourceapi.CapacityRequestPolicy{
				Default:     ptr.To(one),
				ValidValues: []resource.Quantity{one},
			},
			false,
		},
		"not in set": {
			two,
			&resourceapi.CapacityRequestPolicy{
				Default:     ptr.To(one),
				ValidValues: []resource.Quantity{one},
			},
			true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			violate := violatesPolicy(tc.requestedVal, tc.requestPolicy)
			g.Expect(violate).To(BeEquivalentTo(tc.expectResult))
		})
	}
}

func testCalculateConsumedCapacity(t *testing.T) {
	testcases := map[string]struct {
		requestedVal  *resource.Quantity
		capacityValue resource.Quantity
		requestPolicy *resourceapi.CapacityRequestPolicy

		expectResult resource.Quantity
	}{
		"empty": {nil, one, &resourceapi.CapacityRequestPolicy{}, one},
		"min in range": {
			nil,
			two,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one)}},
			one,
		},
		"default in set": {
			nil,
			two,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidValues: []resource.Quantity{one}},
			one,
		},
		"more than min in range": {
			&two,
			two,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one)}},
			two,
		},
		"less than min in range": {
			&one,
			two,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(two)}},
			two,
		},
		"with step (round up)": {
			&two,
			three,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one), Step: ptr.To(two.DeepCopy())}},
			three,
		},
		"with step (no remaining)": {
			&two,
			two,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidRange: &resourceapi.CapacityRequestPolicyRange{Min: ptr.To(one), Step: ptr.To(one.DeepCopy())}},
			two,
		},
		"valid value in set": {
			&two,
			three,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidValues: []resource.Quantity{one, two, three}},
			two,
		},
		"set (round up)": {
			&two,
			three,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidValues: []resource.Quantity{one, three}},
			three,
		},
		"larger than set": {
			&three,
			three,
			&resourceapi.CapacityRequestPolicy{Default: ptr.To(one), ValidValues: []resource.Quantity{one, two}},
			three,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			g := NewWithT(t)
			capacity := resourceapi.DeviceCapacity{
				Value:         tc.capacityValue,
				RequestPolicy: tc.requestPolicy,
			}
			consumedCapacity := calculateConsumedCapacity(tc.requestedVal, capacity)
			g.Expect(consumedCapacity.Cmp(tc.expectResult)).To(BeZero())
		})
	}
}
