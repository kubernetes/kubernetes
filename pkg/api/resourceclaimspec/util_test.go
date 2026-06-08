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

package resourceclaimspec

import (
	"testing"

	"github.com/stretchr/testify/assert"

	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

var testCapacity = map[resource.QualifiedName]apiresource.Quantity{
	resource.QualifiedName("test-capacity"): apiresource.MustParse("1"),
}

var obj = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					Exactly: &resource.ExactDeviceRequest{
						DeviceClassName: "class",
						AllocationMode:  resource.DeviceAllocationModeAll,
					},
				},
			},
		},
	},
}

var objWithPrioritizedList = &resource.ResourceClaim{
	ObjectMeta: metav1.ObjectMeta{
		Name:      "valid-claim",
		Namespace: "kube-system",
	},
	Spec: resource.ResourceClaimSpec{
		Devices: resource.DeviceClaim{
			Requests: []resource.DeviceRequest{
				{
					Name: "req-0",
					FirstAvailable: []resource.DeviceSubRequest{
						{
							Name:            "subreq-0",
							DeviceClassName: "class",
							AllocationMode:  resource.DeviceAllocationModeExactCount,
							Count:           1,
						},
					},
				},
			},
		},
	},
}

func modifySpecDeviceRequestWithCapacityRequests(resourceClaim *resource.ResourceClaim,
	capacity map[resource.QualifiedName]apiresource.Quantity, prioritizedListFeature bool) {
	if capacity != nil {
		if prioritizedListFeature {
			resourceClaim.Spec.Devices.Requests[0].FirstAvailable[0].Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		} else {
			resourceClaim.Spec.Devices.Requests[0].Exactly.Capacity = &resource.CapacityRequirements{
				Requests: capacity,
			}
		}
	}
}

func addDistinctAttribute(resourceClaim *resource.ResourceClaim) {
	distinctConstraint := resource.DeviceConstraint{
		Requests:          []string{"req-0"},
		DistinctAttribute: ptr.To(resource.FullyQualifiedName("driver-a/attr")),
	}
	resourceClaim.Spec.Devices.Constraints = append(resourceClaim.Spec.Devices.Constraints, distinctConstraint)
}

func TestDRAConsumableCapacityFeatureInUse(t *testing.T) {
	testcases := map[string]struct {
		obj    *resource.ResourceClaim
		expect bool
	}{
		"consumable-capacity-empty": {
			obj:    nil,
			expect: false,
		},
		"consumable-capacity-no-inuse": {
			obj:    obj,
			expect: false,
		},
		"consumable-capacity-with-inuse-fields": {
			obj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				return obj
			}(),
			expect: true,
		},
		"consumable-capacity--with-inuse-fields-with-distinct-attribute": {
			obj: func() *resource.ResourceClaim {
				obj := obj.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, false)
				addDistinctAttribute(obj)
				return obj
			}(),
			expect: true,
		},
		"consumable-capacity--with-inuse-fields-in-subrequests": {
			obj: func() *resource.ResourceClaim {
				obj := objWithPrioritizedList.DeepCopy()
				modifySpecDeviceRequestWithCapacityRequests(obj, testCapacity, true)
				return obj
			}(),
			expect: true,
		},
	}
	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			var spec *resource.ResourceClaimSpec
			if tc.obj != nil {
				spec = &tc.obj.Spec
			}
			assert.Equal(t, DRAConsumableCapacityFeatureInUse(spec), tc.expect)
		})
	}
}
