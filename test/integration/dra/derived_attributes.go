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
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

func testDerivedAttributes(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx, nil)

	// 1. Create two distinct device classes and drivers to match our allocator test.
	classA, driverA := createTestClass(tCtx, namespace+"-a")
	classB, driverB := createTestClass(tCtx, namespace+"-b")

	// 2. Create ResourceSlices on worker-0 for both drivers.
	// Driver A has two devices:
	// - device-a1 (numa: numa-0)
	// - device-a2 (numa: numa-1)
	sliceA := st.MakeResourceSlice("worker-0", driverA).
		Device("device-a1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"dra.example.com/numa": {StringValue: new("numa-0")},
		}).
		Device("device-a2", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"dra.example.com/numa": {StringValue: new("numa-1")},
		})
	createSlice(tCtx, sliceA.Obj())

	// Driver B has two devices:
	// - device-b1 (numaNode: numa-1)
	// - device-b2 (numaNode: numa-2)
	sliceB := st.MakeResourceSlice("worker-0", driverB).
		Device("device-b1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"dra.example.com/numaNode": {StringValue: new("numa-1")},
		}).
		Device("device-b2", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"dra.example.com/numaNode": {StringValue: new("numa-2")},
		})
	createSlice(tCtx, sliceB.Obj())

	// 3. Start the scheduler and claim controller.
	startScheduler(tCtx)
	startClaimController(tCtx)

	// 4. Construct the ResourceClaim with DerivedAttributes and a MatchAttribute constraint.
	// Both requests must align on the virtual "sharedNumaNode" grouping key.
	// Since device-a2 and device-b1 both resolve to "numa-1", the allocator should co-allocate them.
	claimName := "derived-attributes-claim"
	claimObj := st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Obj()

	virtualKey := resourceapi.FullyQualifiedName("derived/sharedNumaNode")

	claimObj.Spec.Devices = resourceapi.DeviceClaim{
		Constraints: []resourceapi.DeviceConstraint{
			{
				Requests:       []string{"req-a", "req-b"},
				MatchAttribute: &virtualKey,
			},
		},
		Requests: []resourceapi.DeviceRequest{
			{
				Name: "req-a",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: classA.Name,
					AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
					Count:           1,
					DerivedAttributes: []resourceapi.DeviceDerivedAttribute{
						{
							Name:       virtualKey,
							Expression: `device.attributes["dra.example.com"].numa`,
						},
					},
				},
			},
			{
				Name: "req-b",
				Exactly: &resourceapi.ExactDeviceRequest{
					DeviceClassName: classB.Name,
					AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
					Count:           1,
					DerivedAttributes: []resourceapi.DeviceDerivedAttribute{
						{
							Name:       virtualKey,
							Expression: `device.attributes["dra.example.com"].numaNode`,
						},
					},
				},
			},
		},
	}

	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claimObj, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim")

	// 5. Create a pod referencing this claim.
	pod := st.MakePod().Name("derived-attributes-pod").Namespace(namespace).
		Container("container").
		PodResourceClaims(v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claim.Name}).
		Obj()
	createPod(tCtx, namespace, "", pod, claim)

	// 6. Verify that the pod is successfully scheduled.
	waitForPodScheduled(tCtx, namespace, pod.Name)

	// 7. Verify that the claim is allocated to the correct devices (device-a2 and device-b1).
	allocatedClaim := waitForClaimAllocatedToDevice(tCtx, namespace, claim.Name, schedulingTimeout)
	require.NotNil(tCtx, allocatedClaim.Status.Allocation)

	results := allocatedClaim.Status.Allocation.Devices.Results
	require.Len(tCtx, results, 2)

	sortResults(results)

	require.Equal(tCtx, "req-a", results[0].Request)
	require.Equal(tCtx, "device-a2", results[0].Device)
	require.Equal(tCtx, "req-b", results[1].Request)
	require.Equal(tCtx, "device-b1", results[1].Device)
}

// sortResults sorts results by Request name so assertion is deterministic.
func sortResults(results []resourceapi.DeviceRequestAllocationResult) {
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[i].Request > results[j].Request {
				results[i], results[j] = results[j], results[i]
			}
		}
	}
}
