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
	"flag"
	"fmt"
	"testing"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

const (
	region1 = "region-1"
	region2 = "region-2"
	node1   = "node-1"
	node2   = "node-2"
	classA  = "class-a"
	classB  = "class-b"
	driverA = "driver-a"
	driverB = "driver-b"
	pool1   = "pool-1"
	pool2   = "pool-2"
	pool3   = "pool-3"
	pool4   = "pool-4"
	req0    = "req-0"
	req1    = "req-1"
	req2    = "req-2"
	req3    = "req-3"
	claim0  = "claim-0"
	claim1  = "claim-1"
	slice1  = "slice-1"
	slice2  = "slice-2"
	device1 = "device-1"
	device2 = "device-2"
)

func init() {
	// Bump up the default verbosity for testing. Allocate uses very
	// high thresholds because it is used in the scheduler's per-node
	// filter operation.
	ktesting.DefaultConfig = ktesting.NewConfig(ktesting.Verbosity(7))
	ktesting.DefaultConfig.AddFlags(flag.CommandLine)
}

// Test objects generators

const (
	fieldNameKey     = "metadata.name"
	regionKey        = "region"
	planetKey        = "planet"
	planetValueEarth = "earth"
)

// generate a node object given a name and a region
func node(name, region string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				regionKey: region,
				planetKey: planetValueEarth,
			},
		},
	}
}

// generate a DeviceClass object with the given name and a driver CEL selector.
// driver name is assumed to be the same as the class name.
func class(name, driver string) *resourceapi.DeviceClass {
	return &resourceapi.DeviceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.DeviceClassSpec{
			Selectors: []resourceapi.DeviceSelector{
				{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.driver == "%s"`, driver),
					},
				},
			},
		},
	}
}

// generate a DeviceConfiguration object with the given driver and attribute.
func deviceConfiguration(driver, attribute string) resourceapi.DeviceConfiguration {
	return resourceapi.DeviceConfiguration{
		Opaque: &resourceapi.OpaqueDeviceConfiguration{
			Driver: driver,
			Parameters: runtime.RawExtension{
				Raw: []byte(fmt.Sprintf("{\"%s\":\"%s\"}", attribute, attribute+"Value")),
			},
		},
	}
}

// generate a DeviceClass object with the given name and attribute.
// attribute is used to generate device configuration parameters in a form of JSON {attribute: attributeValue}.
func classWithConfig(name, driver, attribute string) *resourceapi.DeviceClass {
	class := class(name, driver)
	class.Spec.Config = []resourceapi.DeviceClassConfiguration{
		{
			DeviceConfiguration: deviceConfiguration(driver, attribute),
		},
	}
	return class
}

// generate a DeviceClass object with the given name and the node selector
// that selects nodes with the region label set to either "west" or "east".
func classWithSuitableNodes(name, driver string) *resourceapi.DeviceClass {
	class := class(name, driver)
	class.Spec.SuitableNodes = &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      regionKey,
						Operator: v1.NodeSelectorOpIn,
						Values:   []string{region1, region2},
					},
				},
			},
		},
	}
	return class
}

// generate a ResourceClaim object with the given name and device requests.
func claimWithRequests(name string, constraints []resourceapi.DeviceConstraint, requests ...resourceapi.DeviceRequest) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests:    requests,
				Constraints: constraints,
			},
		},
	}
}

// generate a DeviceRequest object with the given name, class and selectors.
func request(name, class string, count int64, selectors ...resourceapi.DeviceSelector) resourceapi.DeviceRequest {
	return resourceapi.DeviceRequest{
		Name:            name,
		Count:           count,
		AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
		DeviceClassName: class,
		Selectors:       selectors,
	}
}

// generate a ResourceClaim object with the given name, request and class.
func claim(name, req, class string, constraints ...resourceapi.DeviceConstraint) *resourceapi.ResourceClaim {
	claim := claimWithRequests(name, constraints, request(req, class, 1))
	return claim
}

// generate a ResourceClaim object with the given name, request, class, and attribute.
// attribute is used to generate parameters in a form of JSON {attribute: attributeValue}.
func claimWithDeviceConfig(name, request, class, driver, attribute string) *resourceapi.ResourceClaim {
	claim := claim(name, request, class)
	claim.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{
		{
			DeviceConfiguration: deviceConfiguration(driver, attribute),
		},
	}
	return claim
}

// generate allocated ResourceClaim object
func allocatedClaim(name, request, class string, results ...resourceapi.DeviceRequestAllocationResult) *resourceapi.ResourceClaim {
	claim := claim(name, request, class)
	claim.Status.Allocation = &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: results,
		},
	}
	return claim
}

// generate a Device object with the given name, capacity and attributes.
func device(name string, capacity map[resourceapi.QualifiedName]resource.Quantity, attributes map[resourceapi.QualifiedName]resourceapi.DeviceAttribute) resourceapi.Device {
	return resourceapi.Device{
		Name: name,
		Basic: &resourceapi.BasicDevice{
			Attributes: attributes,
			Capacity:   capacity,
		},
	}
}

// generate a ResourceSlice object with the given name, node,
// driver and pool names, generation and a list of devices.
// The nodeSelection parameter may be a string (= node name),
// true (= all nodes), or a node selector (= specific nodes).
func slice(name string, nodeSelection any, pool, driver string, devices ...resourceapi.Device) *resourceapi.ResourceSlice {
	slice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: driver,
			Pool: resourceapi.ResourcePool{
				Name:               pool,
				ResourceSliceCount: 1,
				Generation:         1,
			},
			Devices: devices,
		},
	}

	switch nodeSelection := nodeSelection.(type) {
	case *v1.NodeSelector:
		slice.Spec.NodeSelector = nodeSelection
	case bool:
		if !nodeSelection {
			panic("nodeSelection == false is not valid")
		}
		slice.Spec.AllNodes = true
	case string:
		slice.Spec.NodeName = nodeSelection
	default:
		panic(fmt.Sprintf("unexpected nodeSelection type %T: %+v", nodeSelection, nodeSelection))
	}

	return slice
}

func deviceAllocationResult(request, driver, pool, device string) resourceapi.DeviceRequestAllocationResult {
	return resourceapi.DeviceRequestAllocationResult{
		Request: request,
		Driver:  driver,
		Pool:    pool,
		Device:  device,
	}
}

// nodeLabelSelector creates a node selector with a label match for "key" in "values".
func nodeLabelSelector(key string, values ...string) *v1.NodeSelector {
	requirements := []v1.NodeSelectorRequirement{{
		Key:      key,
		Operator: v1.NodeSelectorOpIn,
		Values:   values,
	}}
	selector := &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: requirements}}}
	return selector
}

// localNodeSelector returns a node selector for a specific node.
func localNodeSelector(nodeName string) *v1.NodeSelector {
	selector := nodeLabelSelector(fieldNameKey, nodeName)
	// Swap the requirements: we need to select by field, not label.
	selector.NodeSelectorTerms[0].MatchFields, selector.NodeSelectorTerms[0].MatchExpressions = selector.NodeSelectorTerms[0].MatchExpressions, selector.NodeSelectorTerms[0].MatchFields
	return selector
}

// allocationResult returns a matcher for one AllocationResult pointer with a list of
// embedded device allocation results. The order of those results doesn't matter.
func allocationResult(selector *v1.NodeSelector, results ...resourceapi.DeviceRequestAllocationResult) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchFields(0, gstruct.Fields{
		"Devices": gstruct.MatchFields(0, gstruct.Fields{
			"Results": gomega.ConsistOf(results), // Order is irrelevant.
			"Config":  gomega.BeNil(),
		}),
		"NodeSelector": matchNodeSelector(selector),
		"Controller":   gomega.BeEmpty(),
	}))
}

// matchNodeSelector returns a matcher for a node selector. The order
// of terms, requirements, and values is irrelevant.
func matchNodeSelector(selector *v1.NodeSelector) types.GomegaMatcher {
	if selector == nil {
		return gomega.BeNil()
	}
	return gomega.HaveField("NodeSelectorTerms", matchNodeSelectorTerms(selector.NodeSelectorTerms))
}

func matchNodeSelectorTerms(terms []v1.NodeSelectorTerm) types.GomegaMatcher {
	var matchTerms []types.GomegaMatcher
	for _, term := range terms {
		matchTerms = append(matchTerms, matchNodeSelectorTerm(term))
	}
	return gomega.ConsistOf(matchTerms)
}

func matchNodeSelectorTerm(term v1.NodeSelectorTerm) types.GomegaMatcher {
	return gstruct.MatchFields(0, gstruct.Fields{
		"MatchExpressions": matchNodeSelectorRequirements(term.MatchExpressions),
		"MatchFields":      matchNodeSelectorRequirements(term.MatchFields),
	})
}

func matchNodeSelectorRequirements(requirements []v1.NodeSelectorRequirement) types.GomegaMatcher {
	var matchRequirements []types.GomegaMatcher
	for _, requirement := range requirements {
		matchRequirements = append(matchRequirements, matchNodeSelectorRequirement(requirement))
	}
	return gomega.ConsistOf(matchRequirements)
}

func matchNodeSelectorRequirement(requirement v1.NodeSelectorRequirement) types.GomegaMatcher {
	return gstruct.MatchFields(0, gstruct.Fields{
		"Key":      gomega.Equal(requirement.Key),
		"Operator": gomega.Equal(requirement.Operator),
		"Values":   gomega.ConsistOf(requirement.Values),
	})
}

func allocationResultWithConfig(selector *v1.NodeSelector, driver string, source resourceapi.AllocationConfigSource, attribute string, results ...resourceapi.DeviceRequestAllocationResult) *resourceapi.AllocationResult {
	return &resourceapi.AllocationResult{
		Devices: resourceapi.DeviceAllocationResult{
			Results: results,
			Config: []resourceapi.DeviceAllocationConfiguration{
				{
					Source:              source,
					DeviceConfiguration: deviceConfiguration(driver, attribute),
				},
			},
		},
		NodeSelector: selector,
	}
}

// Helpers

// convert a list of objects to a slice
func objects[T any](objs ...T) []T {
	return objs
}

// generate a ResourceSlice object with the given parameters and one device "device-1"
func sliceWithOneDevice(name string, nodeSelection any, pool, driver string) *resourceapi.ResourceSlice {
	return slice(name, nodeSelection, pool, driver, device(device1, nil, nil))
}

func TestAllocator(t *testing.T) {
	nonExistentAttribute := resourceapi.FullyQualifiedName("NonExistentAttribute")
	boolAttribute := resourceapi.FullyQualifiedName("boolAttribute")
	stringAttribute := resourceapi.FullyQualifiedName("stringAttribute")
	versionAttribute := resourceapi.FullyQualifiedName("driverVersion")
	intAttribute := resourceapi.FullyQualifiedName("numa")

	testcases := map[string]struct {
		claimsToAllocate []*resourceapi.ResourceClaim
		allocatedClaims  []*resourceapi.ResourceClaim
		classes          []*resourceapi.DeviceClass
		slices           []*resourceapi.ResourceSlice
		node             *v1.Node

		expectResults []any
		expectError   types.GomegaMatcher // can be used to check for no error or match specific error types
	}{

		"empty": {},
		"simple": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
			)},
		},
		"other-node": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: objects(
				sliceWithOneDevice(slice1, node1, pool1, driverB),
				sliceWithOneDevice(slice2, node2, pool2, driverA),
			),
			node: node(node2, region2),

			expectResults: []any{allocationResult(
				localNodeSelector(node2),
				deviceAllocationResult(req0, driverA, pool2, device1),
			)},
		},
		"small-and-large": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				request(req0, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
					}}),
				request(req1, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
					}}),
			)),
			classes: objects(class(classA, driverA)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req1, driverA, pool1, device2),
			)},
		},
		"small-and-large-backtrack-requests": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				nil,
				request(req0, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
					}}),
				request(req1, classA, 1, resourceapi.DeviceSelector{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
					}}),
			)),
			classes: objects(class(classA, driverA)),
			// Reversing the order in which the devices are listed causes the "large" device to
			// be allocated for the "small" request, leaving the "large" request unsatisfied.
			// The initial decision needs to be undone before a solution is found.
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req1, driverA, pool1, device2),
			)},
		},
		"small-and-large-backtrack-claims": {
			claimsToAllocate: objects(
				claimWithRequests(
					claim0,
					nil,
					request(req0, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("1Gi")) >= 0`, driverA),
						}})),
				claimWithRequests(
					claim1,
					nil,
					request(req1, classA, 1, resourceapi.DeviceSelector{
						CEL: &resourceapi.CELDeviceSelector{
							Expression: fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("2Gi")) >= 0`, driverA),
						}}),
				)),
			classes: objects(class(classA, driverA)),
			// Reversing the order in which the devices are listed causes the "large" device to
			// be allocated for the "small" request, leaving the "large" request unsatisfied.
			// The initial decision needs to be undone before a solution is found.
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device2, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("2Gi"),
				}, nil),
				device(device1, map[resourceapi.QualifiedName]resource.Quantity{
					"memory": resource.MustParse("1Gi"),
				}, nil),
			)),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req1, driverA, pool1, device2)),
			},
		},
		"devices-split-across-different-slices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name:            req0,
				Count:           2,
				AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
				DeviceClassName: classA,
			})),
			classes: objects(class(classA, driverA)),
			slices: objects(
				sliceWithOneDevice(slice1, node1, pool1, driverA),
				sliceWithOneDevice(slice2, node1, pool2, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req0, driverA, pool2, device1),
			)},
		},
		"obsolete-slice": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices: objects(
				sliceWithOneDevice("slice-1-obsolete", node1, pool1, driverA),
				func() *resourceapi.ResourceSlice {
					slice := sliceWithOneDevice(slice1, node1, pool1, driverA)
					// This makes the other slice obsolete.
					slice.Spec.Pool.Generation++
					return slice
				}(),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
			)},
		},
		"no-slices": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           nil,
			node:             node(node1, region1),

			expectResults: nil,
		},
		"not-enough-suitable-devices": {
			claimsToAllocate: objects(claim(claim0, req0, classA), claim(claim0, req1, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),

			node: node(node1, region1),

			expectResults: nil,
		},
		"no-classes": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          nil,
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("could not retrieve device class class-a")),
		},
		"unknown-class": {
			claimsToAllocate: objects(claim(claim0, req0, "unknown-class")),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("could not retrieve device class unknown-class")),
		},
		"empty-class": {
			claimsToAllocate: objects(claim(claim0, req0, "")),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: missing device class name (unsupported request type?)")),
		},
		"no-claims-to-allocate": {
			claimsToAllocate: nil,
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: nil,
		},
		"all-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name:            req0,
				AllocationMode:  resourceapi.DeviceAllocationModeAll,
				Count:           1,
				DeviceClassName: classA,
			})),
			classes: objects(class(classA, driverA)),
			slices:  objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
			)},
		},
		"all-devices-of-the-incomplete-pool": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, resourceapi.DeviceRequest{
				Name:            req0,
				AllocationMode:  resourceapi.DeviceAllocationModeAll,
				Count:           1,
				DeviceClassName: classA,
			})),
			classes: objects(class(classA, driverA)),
			slices: objects(
				func() *resourceapi.ResourceSlice {
					slice := sliceWithOneDevice(slice1, node1, pool1, driverA)
					// This makes the pool incomplete, one other slice is missing.
					slice.Spec.Pool.ResourceSliceCount++
					return slice
				}(),
			),
			node: node(node1, region1),

			expectResults: nil,
			expectError:   gomega.MatchError(gomega.ContainSubstring("claim claim-0, request req-0: asks for all devices, but resource pool driver-a/pool-1 is currently being updated")),
		},
		"network-attached-device": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{allocationResult(
				nodeLabelSelector(regionKey, region1),
				deviceAllocationResult(req0, driverA, pool1, device1),
			)},
		},
		"unsuccessful-allocation-network-attached-device": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA)),
			// Wrong region, no devices available.
			node: node(node2, region2),

			expectResults: nil,
		},
		"many-network-attached-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 4))),
			classes:          objects(class(classA, driverA)),
			slices: objects(
				sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA),
				sliceWithOneDevice(slice1, true, pool2, driverA),
				sliceWithOneDevice(slice1, nodeLabelSelector(planetKey, planetValueEarth), pool3, driverA),
				sliceWithOneDevice(slice1, localNodeSelector(node1), pool4, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				// A union of the individual selectors.
				&v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{Key: regionKey, Operator: v1.NodeSelectorOpIn, Values: []string{region1}},
							{Key: planetKey, Operator: v1.NodeSelectorOpIn, Values: []string{planetValueEarth}},
						},
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: fieldNameKey, Operator: v1.NodeSelectorOpIn, Values: []string{node1}},
						},
					}},
				},
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req0, driverA, pool2, device1),
				deviceAllocationResult(req0, driverA, pool3, device1),
				deviceAllocationResult(req0, driverA, pool4, device1),
			)},
		},
		"local-and-network-attached-devices": {
			claimsToAllocate: objects(claimWithRequests(claim0, nil, request(req0, classA, 2))),
			classes:          objects(class(classA, driverA)),
			slices: objects(
				sliceWithOneDevice(slice1, nodeLabelSelector(regionKey, region1), pool1, driverA),
				sliceWithOneDevice(slice1, node1, pool2, driverA),
			),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				// Once there is any node-local device, the selector is for that node.
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req0, driverA, pool2, device1),
			)},
		},
		"several-different-drivers": {
			claimsToAllocate: objects(claim(claim0, req0, classA), claim(claim0, req0, classB)),
			classes:          objects(class(classA, driverA), class(classB, driverB)),
			slices: objects(
				slice(slice1, node1, pool1, driverA,
					device(device1, nil, nil),
					device(device2, nil, nil),
				),
				sliceWithOneDevice(slice1, node1, pool1, driverB),
			),
			node: node(node1, region1),

			expectResults: []any{
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverA, pool1, device1)),
				allocationResult(localNodeSelector(node1), deviceAllocationResult(req0, driverB, pool1, device1)),
			},
		},
		"already-allocated-devices": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			allocatedClaims: objects(
				allocatedClaim(claim0, req0, classA,
					deviceAllocationResult(req0, driverA, pool1, device1),
					deviceAllocationResult(req1, driverA, pool1, device2),
				),
			),
			classes: objects(class(classA, driverA)),
			slices:  objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
		},
		"with-constraint": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{
					{MatchAttribute: &intAttribute},
					{MatchAttribute: &versionAttribute},
					{MatchAttribute: &stringAttribute},
					{MatchAttribute: &boolAttribute},
				},
				request(req0, classA, 1),
				request(req1, classA, 1),
			),
			),
			classes: objects(class(classA, driverA)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("1.0.0")},
					"numa":            {IntValue: ptr.To(int64(1))},
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
					"boolAttribute":   {BoolValue: ptr.To(true)},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion":   {VersionValue: ptr.To("1.0.0")},
					"numa":            {IntValue: ptr.To(int64(1))},
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
					"boolAttribute":   {BoolValue: ptr.To(true)},
				}),
			)),
			node: node(node1, region1),

			expectResults: []any{allocationResult(
				localNodeSelector(node1),
				deviceAllocationResult(req0, driverA, pool1, device1),
				deviceAllocationResult(req1, driverA, pool1, device2),
			)},
		},
		"with-constraint-non-existent-attribute": {
			claimsToAllocate: objects(claim(claim0, req0, classA, resourceapi.DeviceConstraint{
				MatchAttribute: &nonExistentAttribute,
			})),
			classes: objects(class(classA, driverA)),
			slices:  objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:    node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-int-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &intAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(1))},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"numa": {IntValue: ptr.To(int64(2))},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-version-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &versionAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("1.0.0")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"driverVersion": {VersionValue: ptr.To("2.0.0")},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-string-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &stringAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue")},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"stringAttribute": {StringValue: ptr.To("stringAttributeValue2")},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-constraint-not-matching-bool-attribute": {
			claimsToAllocate: objects(claimWithRequests(
				claim0,
				[]resourceapi.DeviceConstraint{{MatchAttribute: &boolAttribute}},
				request(req0, classA, 3)),
			),
			classes: objects(class(classA, driverA), class(classB, driverB)),
			slices: objects(slice(slice1, node1, pool1, driverA,
				device(device1, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"boolAttribute": {BoolValue: ptr.To(true)},
				}),
				device(device2, nil, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"boolAttribute": {BoolValue: ptr.To(false)},
				}),
			)),
			node: node(node1, region1),

			expectResults: nil,
		},
		"with-class-device-config": {
			claimsToAllocate: objects(claim(claim0, req0, classA)),
			classes:          objects(classWithConfig(classA, driverA, "classAttribute")),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{
				allocationResultWithConfig(
					localNodeSelector(node1),
					driverA,
					resourceapi.AllocationConfigSourceClass,
					"classAttribute",
					deviceAllocationResult(req0, driverA, pool1, device1),
				),
			},
		},
		"claim-with-device-config": {
			claimsToAllocate: objects(claimWithDeviceConfig(claim0, req0, classA, driverA, "deviceAttribute")),
			classes:          objects(class(classA, driverA)),
			slices:           objects(sliceWithOneDevice(slice1, node1, pool1, driverA)),
			node:             node(node1, region1),

			expectResults: []any{
				allocationResultWithConfig(
					localNodeSelector(node1),
					driverA,
					resourceapi.AllocationConfigSourceClaim,
					"deviceAttribute",
					deviceAllocationResult(req0, driverA, pool1, device1),
				),
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			g := gomega.NewWithT(t)

			// Listing objects is deterministic and returns them in the same
			// order as in the test case. That makes the allocation result
			// also deterministic.
			var allocated, toAllocate claimLister
			var classLister informerLister[resourceapi.DeviceClass]
			var sliceLister informerLister[resourceapi.ResourceSlice]
			for _, claim := range tc.claimsToAllocate {
				toAllocate.claims = append(toAllocate.claims, claim.DeepCopy())
			}
			for _, claim := range tc.allocatedClaims {
				allocated.claims = append(allocated.claims, claim.DeepCopy())
			}
			for _, slice := range tc.slices {
				sliceLister.objs = append(sliceLister.objs, slice.DeepCopy())
			}
			for _, class := range tc.classes {
				classLister.objs = append(classLister.objs, class.DeepCopy())
			}

			allocator, err := NewAllocator(ctx, toAllocate.claims, allocated, classLister, sliceLister)
			g.Expect(err).ToNot(gomega.HaveOccurred())

			results, err := allocator.Allocate(ctx, tc.node)
			matchError := tc.expectError
			if matchError == nil {
				matchError = gomega.Not(gomega.HaveOccurred())
			}
			g.Expect(err).To(matchError)
			g.Expect(results).To(gomega.ConsistOf(tc.expectResults...))

			// Objects that the allocator had access to should not have been modified.
			g.Expect(toAllocate.claims).To(gomega.HaveExactElements(tc.claimsToAllocate))
			g.Expect(allocated.claims).To(gomega.HaveExactElements(tc.allocatedClaims))
			g.Expect(sliceLister.objs).To(gomega.ConsistOf(tc.slices))
			g.Expect(classLister.objs).To(gomega.ConsistOf(tc.classes))
		})
	}
}

type claimLister struct {
	claims []*resourceapi.ResourceClaim
	err    error
}

func (l claimLister) ListAllAllocated() ([]*resourceapi.ResourceClaim, error) {
	return l.claims, l.err
}

type informerLister[T any] struct {
	objs []*T
	err  error
}

func (l informerLister[T]) List(selector labels.Selector) (ret []*T, err error) {
	if selector.String() != labels.Everything().String() {
		return nil, errors.New("labels selector not implemented")
	}
	return l.objs, l.err
}

func (l informerLister[T]) Get(name string) (*T, error) {
	for _, obj := range l.objs {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		if accessor.GetName() == name {
			return obj, nil
		}
	}
	return nil, apierrors.NewNotFound(schema.GroupResource{}, "not found")
}
