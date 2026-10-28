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

	"github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

func stringAttribute(v string) resourceapi.DeviceAttribute {
	return resourceapi.DeviceAttribute{StringValue: &v}
}

// Devices from two drivers publish different attribute names for the same NUMA
// node. derivedAttributes normalizes them so matchAttribute can co-allocate.
var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), framework.WithFeatureGate(features.DRADerivedAttributes), func() {
	f := framework.NewDefaultFramework("dra-derived-attr")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	nodes := drautils.NewNodes(f, 1, 1)

	gpu := drautils.NewDriver(f, nodes, drautils.DriverResources(-1, map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"gpu-0": {"dra.example.com/numa": stringAttribute("numa-0")},
		"gpu-1": {"dra.example.com/numa": stringAttribute("numa-1")},
	}))
	gpu.NameSuffix = "-gpu"
	nic := drautils.NewDriver(f, nodes, drautils.DriverResources(-1, map[string]map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
		"nic-0": {"dra.example.com/numaNode": stringAttribute("numa-1")},
		"nic-1": {"dra.example.com/numaNode": stringAttribute("numa-2")},
	}))
	nic.NameSuffix = "-nic"

	gpuBuilder := drautils.NewBuilder(f, gpu)
	nicBuilder := drautils.NewBuilder(f, nic)

	f.It("co-allocates a GPU and NIC on the same derived NUMA key", func(ctx context.Context) {
		tCtx := f.TContext(ctx)
		virtualKey := resourceapi.FullyQualifiedName("derived/sharedNumaNode")
		claim := &resourceapi.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{
				Name: "derived-attributes-claim",
			},
			Spec: resourceapi.ResourceClaimSpec{
				Devices: resourceapi.DeviceClaim{
					Constraints: []resourceapi.DeviceConstraint{{
						Requests:       []string{"gpu", "nic"},
						MatchAttribute: &virtualKey,
					}},
					Requests: []resourceapi.DeviceRequest{
						{
							Name: "gpu",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: gpuBuilder.ClassName(),
								AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
								Count:           1,
								DerivedAttributes: []resourceapi.DeviceDerivedAttribute{{
									Name:       virtualKey,
									Expression: `device.attributes["dra.example.com"].numa`,
								}},
							},
						},
						{
							Name: "nic",
							Exactly: &resourceapi.ExactDeviceRequest{
								DeviceClassName: nicBuilder.ClassName(),
								AllocationMode:  resourceapi.DeviceAllocationModeExactCount,
								Count:           1,
								DerivedAttributes: []resourceapi.DeviceDerivedAttribute{{
									Name:       virtualKey,
									Expression: `device.attributes["dra.example.com"].numaNode`,
								}},
							},
						},
					},
				},
			},
		}
		gpuBuilder.Create(tCtx, claim)
		pod := gpuBuilder.PodExternal(claim.Name)
		gpuBuilder.Create(tCtx, pod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))

		allocated, err := gpuBuilder.ClientV1(tCtx).ResourceClaims(f.Namespace.Name).Get(tCtx, claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(allocated.Status.Allocation).NotTo(gomega.BeNil())
		results := map[string]string{}
		for _, result := range allocated.Status.Allocation.Devices.Results {
			results[result.Request] = result.Device
		}
		gomega.Expect(results).To(gomega.Equal(map[string]string{
			"gpu": "gpu-1",
			"nic": "nic-0",
		}))
	})
})
