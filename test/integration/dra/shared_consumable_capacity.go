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
	"context"
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

const sharedConsumableCapacityKey = resourceapi.QualifiedName("dra.example.com/bandwidth")

func testSharedConsumableCapacity(tCtx ktesting.TContext, enabled bool) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)

	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")
	nodeName := nodes.Items[0].Name

	createSlice(tCtx, makeSharedConsumableCounterSlice(nodeName, driverName))
	createSlice(tCtx, makeSharedConsumableDeviceSlice(nodeName, driverName))

	startScheduler(tCtx)

	if !enabled {
		// When the feature gate is disabled, ValueFrom fields are dropped by
		// the strategy. The devices effectively have zero-value static counter
		// consumption, so the scheduler can still allocate them. Verify that
		// allocation works (the feature is transparent when disabled).
		claim1 := createClaim(tCtx, namespace, "-1", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
		pod1 := createPod(tCtx, namespace, "-1", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim1)
		waitForPodScheduled(tCtx, namespace, pod1.Name)
		return
	}

	claim1 := createClaim(tCtx, namespace, "-1", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod1 := createPod(tCtx, namespace, "-1", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim1)
	waitForPodScheduled(tCtx, namespace, pod1.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)

	claim2 := createClaim(tCtx, namespace, "-2", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod2 := createPod(tCtx, namespace, "-2", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim2)
	waitForPodScheduled(tCtx, namespace, pod2.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim2.Name, schedulingTimeout)

	claim3 := createClaim(tCtx, namespace, "-3", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod3 := createPod(tCtx, namespace, "-3", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim3)
	assertPodPending(tCtx, pod3)

	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pod1.Name)
	clearClaimAndDelete(tCtx, namespace, claim1.Name)

	waitForPodScheduled(tCtx, namespace, pod3.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim3.Name, schedulingTimeout)
}

// clearClaimAndDelete removes the finalizer and allocation from a claim, then
// deletes it. In integration tests there is no kubelet or controller to do this
// automatically.
func clearClaimAndDelete(tCtx ktesting.TContext, namespace, claimName string) {
	tCtx.Helper()

	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim %s for cleanup", claimName)

	claim.Finalizers = nil
	claim.Status.Allocation = nil
	claim.Status.ReservedFor = nil
	claim, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).Update(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "clear claim %s finalizers and allocation", claimName)

	claim.Status.Allocation = nil
	claim.Status.ReservedFor = nil
	_, err = tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "clear claim %s status", claimName)

	err = tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete(tCtx, claimName, metav1.DeleteOptions{})
	tCtx.ExpectNoError(err, "delete claim %s", claimName)

	waitForNotFound(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).Get, claimName)
}

// assertPodPending checks that the pod is currently pending. It uses
// Consistently to verify the pod stays pending for a short period.
func assertPodPending(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	tCtx.Consistently(func(ctx context.Context) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(
		gomega.HaveField("Status.Phase", gomega.Equal(v1.PodPending)),
		"Pod %s should remain pending.", pod.Name,
	)
}

// makeSharedConsumableClaim creates a ResourceClaim that requests one device and
// consumes shared counter capacity through capacity.requests.
func makeSharedConsumableClaim(name string, quantity resource.Quantity) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Requests: []resourceapi.DeviceRequest{
					{
						Name: "req-0",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: "placeholder",
							AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
							Count:           1,
							Capacity: &resourceapi.CapacityRequirements{
								Requests: map[resourceapi.QualifiedName]resource.Quantity{
									sharedConsumableCapacityKey: quantity,
								},
							},
						},
					},
				},
			},
		},
	}
}

// makeSharedConsumableCounterSlice creates the shared counter slice for the test pool.
func makeSharedConsumableCounterSlice(nodeName, driverName string) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "shared-consumable-counters",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Driver:   driverName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				Generation:         1,
				ResourceSliceCount: 2,
			},
			SharedCounters: []resourceapi.CounterSet{
				{
					Name: "shared-bandwidth",
					Counters: map[string]resourceapi.Counter{
						"bandwidth": func() resourceapi.Counter {
							defaultVal := resource.MustParse("1")
							minVal := resource.MustParse("1")
							maxVal := resource.MustParse("2")
							stepVal := resource.MustParse("1")
							return resourceapi.Counter{
								Value: resource.MustParse("2"),
								RequestPolicy: &resourceapi.CapacityRequestPolicy{
									Default: &defaultVal,
									ValidRange: &resourceapi.CapacityRequestPolicyRange{
										Min:  &minVal,
										Max:  &maxVal,
										Step: &stepVal,
									},
								},
							}
						}(),
					},
				},
			},
		},
	}
}

// makeSharedConsumableDeviceSlice creates two devices that both consume from one shared counter pool.
func makeSharedConsumableDeviceSlice(nodeName, driverName string) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: "shared-consumable-devices",
		},
		Spec: resourceapi.ResourceSliceSpec{
			NodeName: &nodeName,
			Driver:   driverName,
			Pool: resourceapi.ResourcePool{
				Name:               nodeName,
				Generation:         1,
				ResourceSliceCount: 2,
			},
			Devices: []resourceapi.Device{
				{
					Name: "vf-0",
					ConsumesCounters: []resourceapi.DeviceCounterConsumption{
						{
							CounterSet: "shared-bandwidth",
							Counters: map[string]resourceapi.Counter{
								"bandwidth": {
									ValueFrom: &resourceapi.CounterValueFrom{
										CapacityKey: sharedConsumableCapacityKey,
									},
								},
							},
						},
					},
				},
				{
					Name: "vf-1",
					ConsumesCounters: []resourceapi.DeviceCounterConsumption{
						{
							CounterSet: "shared-bandwidth",
							Counters: map[string]resourceapi.Counter{
								"bandwidth": {
									ValueFrom: &resourceapi.CounterValueFrom{
										CapacityKey: sharedConsumableCapacityKey,
									},
								},
							},
						},
					},
				},
			},
		},
	}
}
