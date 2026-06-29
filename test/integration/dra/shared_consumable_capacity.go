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
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
	"k8s.io/utils/ptr"
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

	claim1 := createClaim(tCtx, namespace, "-1", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod1 := createPod(tCtx, namespace, "-1", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim1)

	if !enabled {
		assertPodPendingEventually(tCtx, pod1, schedulingTimeout)
		return
	}

	waitForPodScheduled(tCtx, namespace, pod1.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim1.Name, schedulingTimeout)

	claim2 := createClaim(tCtx, namespace, "-2", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod2 := createPod(tCtx, namespace, "-2", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim2)
	waitForPodScheduled(tCtx, namespace, pod2.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim2.Name, schedulingTimeout)

	claim3 := createClaim(tCtx, namespace, "-3", class, makeSharedConsumableClaim("claim", resource.MustParse("1")))
	pod3 := createPod(tCtx, namespace, "-3", st.MakePod().Name(podName).Namespace(namespace).Container("my-container").Obj(), claim3)
	assertPodPendingEventually(tCtx, pod3, schedulingTimeout)

	deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pod1.Name)
	deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete, tCtx.Client().ResourceV1().ResourceClaims(namespace).Get, claim1.Name)

	waitForPodScheduled(tCtx, namespace, pod3.Name)
	waitForClaimAllocatedToDevice(tCtx, namespace, claim3.Name, schedulingTimeout)
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
							AllocationMode: resourceapi.DeviceAllocationModeExactCount,
							Count:          1,
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
			NodeName: ptr.To(nodeName),
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
						"bandwidth": {
							Value: resource.MustParse("2"),
							RequestPolicy: &resourceapi.CapacityRequestPolicy{
								Default: ptr.To(resource.MustParse("1")),
								ValidRange: &resourceapi.CapacityRequestPolicyRange{
									Min:  ptr.To(resource.MustParse("1")),
									Max:  ptr.To(resource.MustParse("2")),
									Step: ptr.To(resource.MustParse("1")),
								},
							},
						},
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
			NodeName: ptr.To(nodeName),
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

// assertPodPendingEventually waits until the pod remains pending and unscheduled.
func assertPodPendingEventually(tCtx ktesting.TContext, pod *v1.Pod, timeout time.Duration) {
	tCtx.Helper()
	tCtx.AssertEventually(tCtx.Client().CoreV1().Pods(pod.Namespace).Get).
		WithArguments(pod.Name, metav1.GetOptions{}).
		WithTimeout(timeout).
		WithPolling(time.Second).
		Should(gomega.And(
			gomega.HaveField("Status.Phase", gomega.Equal(v1.PodPending)),
			gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":   gomega.Equal(v1.PodScheduled),
				"Status": gomega.Equal(v1.ConditionFalse),
			}))),
		), "Pod %s should remain pending.", pod.Name)
}
