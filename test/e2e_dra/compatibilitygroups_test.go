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

package e2edra

import (
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

// compatibilityGroupsSharedCounters is one counter set with room for four devices
// (each consumes one unit): enough for the gate-cycle's long-lived two-device
// claim plus a fresh single-device claim allocated after re-enable, with headroom.
var compatibilityGroupsSharedCounters = []resourceapi.CounterSet{{
	Name:     "gpu",
	Counters: map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("4")}},
}}

// compatibilityGroupsDeviceList holds four devices declaring the same group, so
// grouped claims co-allocate on the shared counter set with the group check.
var compatibilityGroupsDeviceList = []resourceapi.Device{
	compatibilityGroupsDevice("device-0"),
	compatibilityGroupsDevice("device-1"),
	compatibilityGroupsDevice("device-2"),
	compatibilityGroupsDevice("device-3"),
}

func compatibilityGroupsDevice(name string) resourceapi.Device {
	return resourceapi.Device{
		Name: name,
		ConsumesCounters: []resourceapi.DeviceCounterConsumption{{
			CounterSet:          "gpu",
			Counters:            map[string]resourceapi.Counter{"mem": {Value: resource.MustParse("1")}},
			CompatibilityGroups: []string{"mig"},
		}},
	}
}

func compatibilityGroupsDriverResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	nodename := nodes.NodeNames[0]
	return map[string]resourceslice.DriverResources{
		nodename: {
			Pools: map[string]resourceslice.Pool{
				"compatibility-groups-pool": {
					Slices: []resourceslice.Slice{
						// Devices and SharedCounters must be exclusive per design.
						{SharedCounters: compatibilityGroupsSharedCounters},
						{Devices: compatibilityGroupsDeviceList},
					},
				},
			},
		},
	}
}

// compatibilityGroupsCreatePod creates a pod whose claim requests count devices
// of the grouped device class; count>1 forces the allocator to co-allocate them
// on the shared counter set and run the compatibility-group intersection.
func compatibilityGroupsCreatePod(tCtx ktesting.TContext, b *drautils.Builder, count int64) (*v1.Pod, string) {
	claim := b.ExternalClaim()
	claim.Spec.Devices.Requests = []resourceapi.DeviceRequest{{
		Name: "req-0",
		Exactly: &resourceapi.ExactDeviceRequest{
			DeviceClassName: b.ClassName(),
			Count:           count,
			AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
		},
	}}
	pod := b.PodExternal(claim.Name)
	b.Create(tCtx, claim, pod)
	return pod, claim.Name
}

func verifyCompatibilityGroupsPodAllocated(tCtx ktesting.TContext, pod *v1.Pod, claimName string) {
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(pod.Namespace).Get(tCtx, claimName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get ResourceClaim %s", claimName)
	tCtx.Expect(claim.Status.Allocation).ToNot(gomega.BeNil(), "claim %s must be allocated", claimName)
}

// compatibilityGroupsGateCycle implements the DRADeviceCompatibilityGroups
// ON→OFF→ON cycle test (KEP integration test #4: upgrade → downgrade → upgrade):
//
//   - Phase 0 (gate ON): a claim co-allocates two devices sharing a group; the
//     compatibility check runs and the pod is scheduled.
//   - Phase 1 (gate OFF, downgrade): the existing allocation remains valid.
//   - Phase 2 (gate ON again, re-upgrade): re-enabling enforcement does not
//     re-evaluate or evict the existing allocation, and a fresh grouped
//     allocation works.
func compatibilityGroupsGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	var survivor *v1.Pod
	var survivorClaim string
	tCtx.Run("grouped-pod-running", func(tCtx ktesting.TContext) {
		// Two devices sharing a group, co-allocated on one counter set.
		survivor, survivorClaim = compatibilityGroupsCreatePod(tCtx, b, 2)
		verifyCompatibilityGroupsPodAllocated(tCtx, survivor, survivorClaim)
	})

	return func(tCtx ktesting.TContext) gateOnAgainFunc {
		tCtx.Run("allocation-survives-downgrade", func(tCtx ktesting.TContext) {
			verifyCompatibilityGroupsPodAllocated(tCtx, survivor, survivorClaim)
		})

		return func(tCtx ktesting.TContext) {
			tCtx.Run("allocation-survives-reenable", func(tCtx ktesting.TContext) {
				verifyCompatibilityGroupsPodAllocated(tCtx, survivor, survivorClaim)
			})
			tCtx.Run("fresh-grouped-pod-allocates", func(tCtx ktesting.TContext) {
				// A fresh grouped device allocates end-to-end after re-enable.
				fresh, freshClaim := compatibilityGroupsCreatePod(tCtx, b, 1)
				verifyCompatibilityGroupsPodAllocated(tCtx, fresh, freshClaim)
			})
		}
	}
}
