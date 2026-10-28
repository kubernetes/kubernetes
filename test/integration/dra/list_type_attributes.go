/*
Copyright The Kubernetes Authors.

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

package dra

import (
	"fmt"

	"github.com/onsi/gomega"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// The attributes below are declared with an explicit domain so that a
// constraint can name them without having to know the generated driver name.
const listAttributeDomain = "dra.example.com"

var (
	// listDeviceNameAttribute mirrors the name of the device it is set on.
	// The CEL environment for device selectors has no device.name variable,
	// so this attribute is how a request pins itself to one specific device.
	listDeviceNameAttribute = resourceapi.QualifiedName(listAttributeDomain + "/deviceName")

	// listAttrValuesAttribute is what matchAttribute/distinctAttribute compare.
	listAttrValuesAttribute = resourceapi.QualifiedName(listAttributeDomain + "/attrValues")

	// listAttrValuesConstraintAttribute names listAttrValuesAttribute in a DeviceConstraint.
	listAttrValuesConstraintAttribute = resourceapi.FullyQualifiedName(listAttrValuesAttribute)
)

// listAttributeDevices is the device set published for every test case below. A
// scalar value is treated as a single-element set, therefore:
//
//	device-a {0,1} and device-b {1,2} intersect in {1}
//	device-a {0,1} and device-c {3,4} are disjoint
//	device-s {1}   and device-a {0,1} intersect in {1}
//	device-a {0,1}, device-b {1,2} and device-d {0,2} intersect pairwise,
//	but have no element in common
var listAttributeDevices = []struct {
	name       string
	attrValues resourceapi.DeviceAttribute
}{
	{"device-a", resourceapi.DeviceAttribute{IntValues: []int64{0, 1}}},
	{"device-b", resourceapi.DeviceAttribute{IntValues: []int64{1, 2}}},
	{"device-c", resourceapi.DeviceAttribute{IntValues: []int64{3, 4}}},
	{"device-d", resourceapi.DeviceAttribute{IntValues: []int64{0, 2}}},
	{"device-s", resourceapi.DeviceAttribute{IntValue: new(int64(1))}},
}

func listAttributeSlice(nodeName, driverName string) *resourceapi.ResourceSlice {
	slice := st.MakeResourceSlice(nodeName, driverName)
	for _, device := range listAttributeDevices {
		slice = slice.Device(device.name, map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			listDeviceNameAttribute: {StringValue: new(device.name)},
			listAttrValuesAttribute: device.attrValues,
		})
	}
	return slice.Obj()
}

// listAttributeClaim builds a claim with one request per device name. Each
// request is pinned to that device, therefore only the constraint decides
// whether the claim can be allocated.
func listAttributeClaim(class *resourceapi.DeviceClass, constraint resourceapi.DeviceConstraint, deviceNames ...string) *resourceapi.ResourceClaim {
	claim := st.MakeResourceClaim().Name("claim").Obj()
	requests := make([]resourceapi.DeviceRequest, 0, len(deviceNames))
	requestNames := make([]string, 0, len(deviceNames))
	for i, deviceName := range deviceNames {
		requestName := fmt.Sprintf("req-%d", i+1)
		requestNames = append(requestNames, requestName)
		requests = append(requests, resourceapi.DeviceRequest{
			Name: requestName,
			Exactly: &resourceapi.ExactDeviceRequest{
				DeviceClassName: class.Name,
				AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
				Count:           1,
				Selectors: []resourceapi.DeviceSelector{{
					CEL: &resourceapi.CELDeviceSelector{
						Expression: fmt.Sprintf(`device.attributes[%q].deviceName == %q`, listAttributeDomain, deviceName),
					},
				}},
			},
		})
	}
	constraint.Requests = requestNames
	claim.Spec.Devices.Requests = requests
	claim.Spec.Devices.Constraints = []resourceapi.DeviceConstraint{constraint}
	return claim
}

// testListTypeAttributes exercises the DRAListTypeAttributes feature end to end
// through a real apiserver + scheduler: a pod must get scheduled or stay pending
// depending on whether the list-typed attributes of the devices which its
// requests are pinned to satisfy the claim's constraint.
func testListTypeAttributes(tCtx ktesting.TContext) {
	// distinctAttribute is guarded by the DRAConsumableCapacity feature gate,
	// which the "all" configuration enables explicitly.
	matchAttrValues := resourceapi.DeviceConstraint{MatchAttribute: &listAttrValuesConstraintAttribute}
	distinctAttrValues := resourceapi.DeviceConstraint{DistinctAttribute: &listAttrValuesConstraintAttribute}

	tCtx.Run("MatchAttributeIntersectingLists", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, matchAttrValues, []string{"device-a", "device-b"}, true)
	})
	tCtx.Run("MatchAttributeDisjointLists", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, matchAttrValues, []string{"device-a", "device-c"}, false)
	})
	// matchAttribute narrows one intersection across all devices instead of
	// comparing them pairwise, so devices which only overlap pairwise must not
	// be allocated together.
	tCtx.Run("MatchAttributePairwiseOverlappingLists", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, matchAttrValues, []string{"device-a", "device-b", "device-d"}, false)
	})
	tCtx.Run("MatchAttributeScalarAndList", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, matchAttrValues, []string{"device-s", "device-a"}, true)
	})
	tCtx.Run("DistinctAttributeDisjointLists", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, distinctAttrValues, []string{"device-a", "device-c"}, true)
	})
	tCtx.Run("DistinctAttributeOverlappingLists", func(tCtx ktesting.TContext) {
		testListTypeAttributesConstraint(tCtx, distinctAttrValues, []string{"device-a", "device-b"}, false)
	})
}

// testListTypeAttributesConstraint publishes listAttributeDevices in one
// node-local pool and checks the outcome of a claim whose requests are pinned to
// the given devices. When schedulable is true the claim must end up allocated to
// exactly those devices, otherwise the pod must stay unschedulable.
func testListTypeAttributesConstraint(tCtx ktesting.TContext, constraint resourceapi.DeviceConstraint, deviceNames []string, schedulable bool) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	node := firstNodeName(tCtx)

	createSlice(tCtx, listAttributeSlice(node, driverName))
	startScheduler(tCtx)

	claim := createClaim(tCtx, namespace, "-list", class, listAttributeClaim(class, constraint, deviceNames...))
	pod := st.MakePod().Name(podName).Namespace(namespace).Container("c").Obj()
	createdPod := createPod(tCtx, namespace, "-list", pod, claim)

	if !schedulable {
		expectPodUnschedulable(tCtx, createdPod, "cannot allocate all claims")
		unallocated, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get claim")
		tCtx.Expect(unallocated.Status.Allocation).To(gomega.BeNil(), "claim must not be allocated")
		return
	}

	scheduled := waitForPodScheduled(tCtx, namespace, createdPod.Name)
	tCtx.Expect(scheduled.Spec.NodeName).To(gomega.Equal(node))

	allocated, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get allocated claim")
	tCtx.Expect(allocated.Status.Allocation).ToNot(gomega.BeNil(), "claim should be allocated")

	devices := make([]string, 0, len(allocated.Status.Allocation.Devices.Results))
	for _, result := range allocated.Status.Allocation.Devices.Results {
		devices = append(devices, result.Device)
	}
	tCtx.Expect(devices).To(gomega.ConsistOf(deviceNames))
}
