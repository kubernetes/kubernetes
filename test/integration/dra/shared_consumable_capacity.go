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
	"k8s.io/client-go/ktesting"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

const sharedConsumableCapacityName = resourceapi.QualifiedName("bandwidth")

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
									sharedConsumableCapacityName: quantity,
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
					Counters: map[string]resourceapi.SharedCounter{
						"bandwidth": func() resourceapi.SharedCounter {
							defaultVal := resource.MustParse("1")
							minVal := resource.MustParse("1")
							maxVal := resource.MustParse("2")
							stepVal := resource.MustParse("1")
							return resourceapi.SharedCounter{
								Value: mustParseQuantityPtr("2"),
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
							Counters: map[string]resourceapi.ConsumeCounter{
								"bandwidth": {
									ValueFrom: &resourceapi.CounterValueFrom{
										CapacityName: sharedConsumableCapacityName,
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
							Counters: map[string]resourceapi.ConsumeCounter{
								"bandwidth": {
									ValueFrom: &resourceapi.CounterValueFrom{
										CapacityName: sharedConsumableCapacityName,
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

func testSharedStaticCounterRelease(tCtx ktesting.TContext) {
	namespace := createTestNamespace(tCtx, nil)
	class, driverName := createTestClass(tCtx, namespace)
	nodes, err := tCtx.Client().CoreV1().Nodes().List(tCtx, metav1.ListOptions{})
	tCtx.ExpectNoError(err, "list nodes")
	nodeName := nodes.Items[0].Name

	counterSlice := makeSharedConsumableCounterSlice(nodeName, driverName)
	counterSlice.Name = namespace + "-static-counters"
	counterSlice.Spec.SharedCounters = []resourceapi.CounterSet{{
		Name:     "shared-memory",
		Counters: map[string]resourceapi.SharedCounter{"memory": {Value: mustParseQuantityPtr("10Gi")}},
	}}
	consumption := []resourceapi.DeviceCounterConsumption{{
		CounterSet: "shared-memory",
		Counters:   map[string]resourceapi.ConsumeCounter{"memory": {Value: mustParseQuantityPtr("6Gi")}},
	}}
	deviceSlice := makeSharedConsumableDeviceSlice(nodeName, driverName)
	deviceSlice.Name = namespace + "-static-devices"
	deviceSlice.Spec.Devices = []resourceapi.Device{
		{Name: "shared-device", AllowMultipleAllocations: new(true), ConsumesCounters: consumption},
		{Name: "other-device", ConsumesCounters: consumption},
	}
	createSlice(tCtx, counterSlice)
	createSlice(tCtx, deviceSlice)
	startScheduler(tCtx)

	var claims []*resourceapi.ResourceClaim
	var pods []*v1.Pod
	for _, suffix := range []string{"-1", "-2", "-3"} {
		newClaim := st.MakeResourceClaim().Name("claim").Request(class.Name).Obj()
		expression := "device.allowMultipleAllocations"
		if suffix == "-3" {
			expression = "!device.allowMultipleAllocations"
		}
		newClaim.Spec.Devices.Requests[0].Exactly.Selectors = []resourceapi.DeviceSelector{{CEL: &resourceapi.CELDeviceSelector{Expression: expression}}}
		newClaim = createClaim(tCtx, namespace, suffix, class, newClaim)
		pod := createPod(tCtx, namespace, suffix, st.MakePod().Name(podName).Container("my-container").Obj(), newClaim)
		if suffix != "-3" {
			waitForPodScheduled(tCtx, namespace, pod.Name)
			newClaim = waitForClaimAllocatedToDevice(tCtx, namespace, newClaim.Name, schedulingTimeout)
			tCtx.Expect(newClaim.Status.Allocation.Devices.Results).To(gomega.HaveLen(1))
			result := newClaim.Status.Allocation.Devices.Results[0]
			tCtx.Expect(result.Device).To(gomega.Equal("shared-device"))
			tCtx.Expect(result.ShareID).NotTo(gomega.BeNil())
			tCtx.Expect(result.ConsumedCounters).NotTo(gomega.BeNil())
			tCtx.Expect(result.ConsumedCounters.PerDevice).To(gomega.Equal([]resourceapi.CounterSetConsumption{{CounterSet: "shared-memory", Counters: map[string]resource.Quantity{"memory": resource.MustParse("6Gi")}}}))
		}
		claims = append(claims, newClaim)
		pods = append(pods, pod)
	}
	assertPodPending(tCtx, pods[2])

	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pods[0].Name)
	clearClaimAndDelete(tCtx, namespace, claims[0].Name)
	tCtx.Consistently(func(ctx context.Context) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(ctx, pods[2].Name, metav1.GetOptions{})
	}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveField("Spec.NodeName", gomega.BeEmpty()), "a surviving share must retain the static charge")
	survivor, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claims[1].Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get surviving share")
	tCtx.Expect(survivor.Status.Allocation).To(gomega.Equal(claims[1].Status.Allocation))

	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pods[1].Name)
	clearClaimAndDelete(tCtx, namespace, claims[1].Name)
	waitForPodScheduled(tCtx, namespace, pods[2].Name)
	allocated := waitForClaimAllocatedToDevice(tCtx, namespace, claims[2].Name, schedulingTimeout)
	tCtx.Expect(allocated.Status.Allocation.Devices.Results[0].Device).To(gomega.Equal("other-device"))
}
