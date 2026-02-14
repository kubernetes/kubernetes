/*
Copyright 2026 The Kubernetes Authors.

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

package allocatortesting

import (
	"context"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/cel"
)

// TestLexicographicalAllocator runs tests which depend on lexicographical sorting.
func TestLexicographicalAllocator(t *testing.T,
	supportedFeatures Features,
	newAllocator func(
		ctx context.Context,
		features Features,
		allocateState AllocatedState,
		classLister DeviceClassLister,
		slices []*resourceapi.ResourceSlice,
		celCache *cel.Cache,
	) (Allocator, error)) {
	testcases := map[string]AllocatorTestCase{
		"lexicographical-sorting-pools": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// pool-3 before pool-2 before pool-1 in input.
				sliceWithOneDevice(slice1, node1, resourcePool(pool3, 1), driverA),
				sliceWithOneDevice(slice2, node1, resourcePool(pool2, 1), driverA),
				sliceWithOneDevice(slice3, node1, resourcePool(pool1, 1), driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool2, device1, false),
			)},
		},
		"lexicographical-sorting-pools-with-binding-conditions": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 1), driverA,
					device(device1, nil, nil).withBindingConditions([]string{"Ready"}, nil),
				),
				sliceWithDevices(slice2, node1, resourcePool(pool3, 1), driverA, device(device2, nil, nil)),
				sliceWithDevices(slice3, node1, resourcePool(pool2, 1), driverA, device(device3, nil, nil)),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool2, device3, false),
				deviceAllocationResult(req0, driverA, pool3, device2, false),
			)},
		},
		"lexicographical-sorting-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				// slice-3 before slice-2 before slice-1 in input.
				sliceWithDevices(slice3, node1, resourcePool(pool1, 3), driverA, device(device3, nil, nil)),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA, device(device2, nil, nil)),
				sliceWithDevices(slice1, node1, resourcePool(pool1, 3), driverA, device(device1, nil, nil)),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1, false),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
			)},
		},
		"lexicographical-sorting-slices-with-binding-conditions": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: unwrapResourceSlices(
				sliceWithDevices(slice1, node1, resourcePool(pool1, 3), driverA,
					device(device1, nil, nil).withBindingConditions([]string{"Ready"}, nil),
				),
				sliceWithDevices(slice3, node1, resourcePool(pool1, 3), driverA, device(device3, nil, nil)),
				sliceWithDevices(slice2, node1, resourcePool(pool1, 3), driverA, device(device2, nil, nil)),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device2, false),
				deviceAllocationResult(req0, driverA, pool1, device3, false),
			)},
		},
	}

	RunTestAllocator(t, supportedFeatures, newAllocator, testcases)
}
