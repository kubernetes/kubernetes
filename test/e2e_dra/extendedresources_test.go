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
	"slices"
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

const (
	resourceTypeExplicit = "explicit"
	resourceTypeImplicit = "implicit"
)

func extendedResourcesDriverResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	return drautils.DriverResourcesNow(nodes, 8)
}

// extendedResourceUpgradeDowngrade tests the DRAExtendedResources feature during upgrade/downgrade scenarios
// for a specific resource type (explicit or implicit).
// This test verifies:
// 1. DeviceClasses can be created with the specified extendedResourceName type
// 2. Pods requesting extended resources are scheduled and allocated devices before upgrade
// 3. Existing pods continue running correctly after upgrade
// 4. New pods with extended resources work correctly after upgrade
// 5. All pods continue running correctly after downgrade
// 6. ResourceClaims are garbage collected when pods are deleted
func extendedResourceUpgradeDowngrade(resourceType string) initialTestFunc {
	return func(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
		return testExtendedResource(tCtx, b, resourceType)
	}
}

// testExtendedResource tests a single extended resource type through the upgrade/downgrade cycle.
// It creates the DeviceClass, creates pods before upgrade, verifies them after upgrade,
// creates new pods after upgrade, and verifies and cleans up all pods after downgrade.
func testExtendedResource(tCtx ktesting.TContext, b *drautils.Builder, resourceType string) upgradedTestFunc {
	namespace := tCtx.Namespace()

	// Create DeviceClass and get the resource name
	resourceName := createDeviceClassForExtendedResource(tCtx, b, resourceType)

	// Before upgrade: create and verify pod
	podBefore := createPodWithExtendedResource(tCtx, b, fmt.Sprintf("%s-before-upgrade", resourceType), resourceName)
	verifyPodRunningWithClaim(tCtx, b, podBefore, resourceName)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade: verify existing pod still runs correctly
		verifyPodRunningWithClaim(tCtx, b, podBefore, resourceName)

		// Create and verify another pod with the same extended resource
		podAfter := createPodWithExtendedResource(tCtx, b, fmt.Sprintf("%s-after-upgrade", resourceType), resourceName)
		verifyPodRunningWithClaim(tCtx, b, podAfter, resourceName)

		return func(tCtx ktesting.TContext) {
			// After downgrade: verify all pods still run correctly with their claims
			pods := []*v1.Pod{podBefore, podAfter}
			for _, pod := range pods {
				verifyPodRunningWithClaim(tCtx, b, pod, resourceName)
			}

			// Collect claim names before deleting pods so GC can be verified.
			claimNames := make([]string, 0, len(pods))
			for _, pod := range pods {
				claimNames = append(claimNames, claimNameFromPod(tCtx, pod))
			}

			// Clean up pods
			tCtx.Logf("Cleaning up %s resource pods after downgrade", resourceType)
			for _, pod := range pods {
				tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), pod.Name, namespace))
			}
			tCtx.Logf("Successfully cleaned up %s resource pods", resourceType)

			eventuallyClaimsGone(tCtx, claimNames, 3*time.Minute)
		}
	}
}

// extendedResourceGateCycle implements the DRAExtendedResource feature gate ON→OFF→ON cycle test:
//
//   - Phase 0 (gate ON): create a survivor pod and a to-delete pod with extended resource
//     claims and verify they run and are allocated.
//   - Phase 1 (gate OFF): assert survivor claims are preserved; verify that a new
//     DeviceClass.ExtendedResourceName field is stripped by the apiserver; verify that a
//     probe pod stays Pending without a special claim; delete to-delete pod and verify
//     its claim is garbage collected.
//   - Phase 2 (gate ON again): wait for kubelet's reconcile loop to unprepare the claim
//     deleted in phase 1; verify the survivor still runs; allocate a fresh pod end-to-end;
//     clean up.
func extendedResourceGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	var podSurvivor, podToDelete *v1.Pod
	var resourceName v1.ResourceName
	tCtx.Run("pods-with-extended-resource-running", func(tCtx ktesting.TContext) {
		resourceName = createDeviceClassForExtendedResource(tCtx, b, resourceTypeExplicit)
		podSurvivor = createPodWithExtendedResource(tCtx, b, "survivor", resourceName)
		verifyPodRunningWithClaim(tCtx, b, podSurvivor, resourceName)
		podToDelete = createPodWithExtendedResource(tCtx, b, "to-delete", resourceName)
		verifyPodRunningWithClaim(tCtx, b, podToDelete, resourceName)
	})

	return func(tCtx ktesting.TContext) gateOnAgainFunc {
		tCtx.Run("survivor-pods-still-running", func(tCtx ktesting.TContext) {
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), podSurvivor))
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), podToDelete))
		})

		// Verify that each pod's special ResourceClaim still exists and is allocated.
		tCtx.Run("survivor-claim-preserved", func(tCtx ktesting.TContext) {
			for _, pod := range []*v1.Pod{podSurvivor, podToDelete} {
				claimName := claimNameFromPod(tCtx, pod)
				claim, err := tCtx.Client().ResourceV1().ResourceClaims(pod.Namespace).Get(tCtx, claimName, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get ResourceClaim %s", claimName)
				tCtx.Expect(claim.Status.Allocation).ToNot(gomega.BeNil(),
					"ResourceClaim %s must still be allocated", claimName)
			}
		})

		tCtx.Run("new-deviceclass-field-stripped", func(tCtx ktesting.TContext) {
			dc := b.ClassWithExtendedResource(string(resourceName))
			dc.Name += "-post-off"
			created := b.Create(tCtx, dc)[0].(*resourceapi.DeviceClass)
			tCtx.Expect(created.Spec.ExtendedResourceName).To(gomega.BeNil(),
				"DeviceClass.Spec.ExtendedResourceName must be stripped when gate is off")
		})

		// check that probe stays Pending and never gets a special claim.
		tCtx.Run("probe-pod-pending-no-special-claim", func(tCtx ktesting.TContext) {
			probePod := createPodWithExtendedResource(tCtx, b, "probe-off", resourceName)
			tCtx.WithStep("probe pod stays Pending").
				Consistently(func(tCtx ktesting.TContext) v1.PodPhase {
					p, err := tCtx.Client().CoreV1().Pods(probePod.Namespace).Get(tCtx, probePod.Name, metav1.GetOptions{})
					tCtx.ExpectNoError(err)
					tCtx.Expect(p.Status.ExtendedResourceClaimStatus).To(gomega.BeNil(),
						"probe pod must not get a special claim when gate is off")
					return p.Status.Phase
				}).WithTimeout(30 * time.Second).Should(gomega.Equal(v1.PodPending))

			// Cleanup probe pod.
			tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), probePod.Name, probePod.Namespace))
		})

		var toDeleteClaimName string
		tCtx.Run("delete-pod-triggers-gc-with-gate-off", func(tCtx ktesting.TContext) {
			// KEP-required: deleting a pod with gate OFF must let the GC (not
			// the apiserver) remove the claim, and node accounting must release
			// the device.
			toDeleteClaimName = claimNameFromPod(tCtx, podToDelete)
			tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), podToDelete.Name, podToDelete.Namespace))
			eventuallyClaimsGone(tCtx, []string{toDeleteClaimName}, 3*time.Minute)
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), podSurvivor))
		})

		return func(tCtx ktesting.TContext) {
			// Wait for kubelet's reconcile loop (period: 60s) to call NodeUnprepareResources
			// for to-delete, which was deleted while the gate was OFF. The reconcile loop
			// only calls unprepare when the gate is ON, so this must happen after the flip.
			tCtx.Run("wait-for-claim-unprepared", func(tCtx ktesting.TContext) {
				for host, plugin := range b.Driver.Nodes {
					tCtx.WithStep(fmt.Sprintf("wait for claim %s on %s to be unprepared", toDeleteClaimName, host)).
						Eventually(func(ktesting.TContext) []app.ClaimID {
							return plugin.GetPreparedResources()
						}).WithTimeout(3*time.Minute).ShouldNot(gomega.ContainElement(gomega.HaveField("Name", gomega.HavePrefix(toDeleteClaimName))),
						// HavePrefix because app.ClaimID.Name is "<claimName>/<device>", not the bare claim name.
						"claim %s on host %s should be unprepared after gate re-enable", toDeleteClaimName, host)
				}
			})

			tCtx.Run("survivor-pod-survives-reenable", func(tCtx ktesting.TContext) {
				verifyPodRunningWithClaim(tCtx, b, podSurvivor, resourceName)
			})

			var freshPod *v1.Pod
			tCtx.Run("fresh-pod-allocates-end-to-end", func(tCtx ktesting.TContext) {
				freshPod = createPodWithExtendedResource(tCtx, b, "fresh", resourceName)
				verifyPodRunningWithClaim(tCtx, b, freshPod, resourceName)
			})

			claimNames := []string{
				claimNameFromPod(tCtx, podSurvivor),
				claimNameFromPod(tCtx, freshPod),
			}

			tCtx.Run("cleanup-pods", func(tCtx ktesting.TContext) {
				tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), podSurvivor.Name, podSurvivor.Namespace))
				tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), freshPod.Name, freshPod.Namespace))
			})

			tCtx.Run("claims-GC", func(tCtx ktesting.TContext) {
				eventuallyClaimsGone(tCtx, claimNames, 3*time.Minute)
			})
		}
	}
}

// createDeviceClassForExtendedResource gets or creates a DeviceClass for testing extended resources.
// For explicit resource type, it creates a DeviceClass with a unique extendedResourceName and returns that name.
// For implicit resource type, it returns the resource name derived from the existing DeviceClass created by the builder.
func createDeviceClassForExtendedResource(tCtx ktesting.TContext, b *drautils.Builder, resourceType string) v1.ResourceName {
	switch resourceType {
	case resourceTypeExplicit:
		resourceName := v1.ResourceName(b.ExtendedResourceName(resourceTypeExplicit))
		deviceClass := b.ClassWithExtendedResource(string(resourceName))
		// Create the DeviceClass
		createdObjs := b.Create(tCtx, deviceClass)
		// Verify ExtendedResourceName is present in the spec
		deviceClass = createdObjs[0].(*resourceapi.DeviceClass)
		tCtx.Expect(deviceClass.Spec.ExtendedResourceName).ToNot(gomega.BeNil(), "DeviceClass.Spec.ExtendedResourceName should be set for explicit resource name")
		tCtx.Logf("Created DeviceClass %q with extendedResourceName: %s", deviceClass.Name, resourceName)
		return resourceName
	case resourceTypeImplicit:
		// The device class was already created by the builder, we only need the name.
		return v1.ResourceName(fmt.Sprintf("%s%s", resourceapi.ResourceDeviceClassPrefix, b.ClassName()))
	default:
		tCtx.Fatalf("invalid resource type %q", resourceType)
		return ""
	}
}

// claimNameFromPod re-fetches the pod and returns the ResourceClaim name from
// pod.Status.ExtendedResourceClaimStatus.
func claimNameFromPod(tCtx ktesting.TContext, pod *v1.Pod) string {
	tCtx.Helper()
	p, err := tCtx.Client().CoreV1().Pods(pod.Namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get pod %s", pod.Name)
	tCtx.Expect(p.Status.ExtendedResourceClaimStatus).ToNot(gomega.BeNil(),
		"pod %s must have ExtendedResourceClaimStatus", pod.Name)
	return p.Status.ExtendedResourceClaimStatus.ResourceClaimName
}

// createPodWithExtendedResource creates a pod with the given name requesting an extended resource.
func createPodWithExtendedResource(tCtx ktesting.TContext, b *drautils.Builder, podName string, resourceName v1.ResourceName) *v1.Pod {
	pod := b.Pod()
	pod.Name = podName
	resources := v1.ResourceList{resourceName: resource.MustParse("1")}
	pod.Spec.Containers[0].Resources.Requests = resources
	pod.Spec.Containers[0].Resources.Limits = resources
	pod = b.Create(tCtx, pod)[0].(*v1.Pod)
	tCtx.Logf("Created pod %q requesting extended resource %q", pod.Name, resourceName)
	return pod
}

// verifyPodRunningWithClaim verifies a pod is running and has a valid ResourceClaim.
func verifyPodRunningWithClaim(tCtx ktesting.TContext, b *drautils.Builder, pod *v1.Pod, resourceName v1.ResourceName) {
	namespace := tCtx.Namespace()
	// Use Eventually to retry the full run+exec check. During cluster upgrades and
	// downgrades, nodes are temporarily restarted, which can cause transient exec
	// failures ("unable to upgrade connection: pod does not exist") even though the
	// pod is reported as Running by the API server.
	tCtx.Eventually(func(tCtx ktesting.TContext) {
		b.TestPod(tCtx, pod, "container_0_request_0", "true")
	}).WithTimeout(framework.PodStartTimeout).Should(gomega.Succeed(), "pod %s should be running with expected env", pod.Name)
	tCtx.Logf("Pod %q is running", pod.Name)

	// Get updated pod to check status
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get pod to check status")

	// Verify pod.status.extendedResourceClaimStatus contains the mapping
	tCtx.Expect(pod.Status.ExtendedResourceClaimStatus).ToNot(gomega.BeNil(), "pod.status.extendedResourceClaimStatus should not be nil for pod %s", pod.Name)

	// Get the claim name from pod status
	claimName := pod.Status.ExtendedResourceClaimStatus.ResourceClaimName
	tCtx.Logf("Verified extendedResourceClaimStatus.ResourceClaimName: %s for pod %q", claimName, pod.Name)

	// Fetch the ResourceClaim using the name from pod status
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "getting ResourceClaim %s", claimName)

	// Verify the claim is allocated
	tCtx.Expect(claim.Status.Allocation).ToNot(gomega.BeNil(), "ResourceClaim %s should be allocated", claim.Name)
	tCtx.Logf("ResourceClaim %q is allocated for pod %q", claim.Name, pod.Name)

	// Verify request mappings contain the extended resource
	foundMapping := slices.ContainsFunc(pod.Status.ExtendedResourceClaimStatus.RequestMappings, func(reqMapping v1.ContainerExtendedResourceRequest) bool {
		if reqMapping.ResourceName == string(resourceName) {
			tCtx.Logf("Found request mapping for pod %q: container=%s, resource=%s, request=%s",
				pod.Name, reqMapping.ContainerName, reqMapping.ResourceName, reqMapping.RequestName)
			return true
		}
		return false
	})
	tCtx.Expect(foundMapping).To(gomega.BeTrueBecause("extendedResourceClaimStatus.RequestMappings should contain %s for pod %s", resourceName, pod.Name))
}
