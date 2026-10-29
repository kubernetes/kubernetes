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
	"fmt"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/ktesting"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/utils/ptr"
)

var sharingAffinityExtractors = []resourceapi.SharingAffinityExtractor{{
	Name:       "subnet",
	APIGroup:   "networking.example.com",
	Kind:       "NICConfig",
	Expression: `{"subnet": string(config.subnetID)}`,
}}

var sharingAffinityDeviceList = []resourceapi.Device{
	sharingAffinityDevice("nic-0", 4),
}

func sharingAffinityDevice(name string, maxUnits int64) resourceapi.Device {
	oneUnit := resource.MustParse("1")
	return resourceapi.Device{
		Name:                     name,
		AllowMultipleAllocations: ptr.To(true),
		Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
			"slots": {
				Value: *resource.NewQuantity(maxUnits, resource.DecimalSI),
				RequestPolicy: &resourceapi.CapacityRequestPolicy{
					Default: &oneUnit,
					ValidRange: &resourceapi.CapacityRequestPolicyRange{
						Min:  &oneUnit,
						Step: &oneUnit,
					},
				},
			},
		},
	}
}

func sharingAffinityDriverResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	nodename := nodes.NodeNames[0]
	return map[string]resourceslice.DriverResources{
		nodename: {
			Pools: map[string]resourceslice.Pool{
				"sharing-affinity-pool": {
					Slices: []resourceslice.Slice{
						// SharingAffinity and Devices must be in separate slices per design.
						{SharingAffinity: sharingAffinityExtractors},
						{Devices: sharingAffinityDeviceList},
					},
				},
			},
		},
	}
}

func sharingAffinityCreatePod(tCtx ktesting.TContext, b *drautils.Builder, subnetID string) (*v1.Pod, string) {
	claim := b.ExternalClaim()
	claim.Spec.Devices.Config = []resourceapi.DeviceClaimConfiguration{{
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{
				Driver: b.Driver.Name,
				Parameters: runtime.RawExtension{
					Raw: []byte(fmt.Sprintf(`{"apiVersion":"networking.example.com/v1","kind":"NICConfig","subnetID":%q}`, subnetID)),
				},
			},
		},
	}}
	pod := b.PodExternal(claim.Name)
	b.Create(tCtx, claim, pod)
	return pod, claim.Name
}

func verifySharingAffinityPodAllocated(tCtx ktesting.TContext, pod *v1.Pod, claimName string) {
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(pod.Namespace).Get(tCtx, claimName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get ResourceClaim %s", claimName)
	tCtx.Expect(claim.Status.Allocation).ToNot(gomega.BeNil(), "claim %s must be allocated", claimName)
}

// sharingAffinityGateCycle implements the DRASharingAffinity ON→OFF→ON cycle test:
//
//   - Phase 0 (gate ON): a claim with matching NICConfig allocates on the shared
//     device with sharing affinity extracted and locked.
//   - Phase 1 (gate OFF, downgrade): the existing allocation remains valid.
//   - Phase 2 (gate ON again, re-upgrade): re-enabling enforcement preserves the
//     existing allocation, reconstructs the device's affinity lock from the
//     active claim, and co-locates a fresh matching claim on the same device.
func sharingAffinityGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	var survivor *v1.Pod
	var survivorClaim string
	tCtx.Run("affinity-pod-running", func(tCtx ktesting.TContext) {
		survivor, survivorClaim = sharingAffinityCreatePod(tCtx, b, "subnet-a")
		verifySharingAffinityPodAllocated(tCtx, survivor, survivorClaim)
	})

	return func(tCtx ktesting.TContext) gateOnAgainFunc {
		tCtx.Run("allocation-survives-downgrade", func(tCtx ktesting.TContext) {
			verifySharingAffinityPodAllocated(tCtx, survivor, survivorClaim)
		})

		return func(tCtx ktesting.TContext) {
			tCtx.Run("allocation-survives-reenable", func(tCtx ktesting.TContext) {
				verifySharingAffinityPodAllocated(tCtx, survivor, survivorClaim)
			})
			tCtx.Run("fresh-compatible-pod-allocates", func(tCtx ktesting.TContext) {
				fresh, freshClaim := sharingAffinityCreatePod(tCtx, b, "subnet-a")
				verifySharingAffinityPodAllocated(tCtx, fresh, freshClaim)
			})
		}
	}
}
