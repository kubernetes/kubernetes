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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/resourceslice"
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
			podNames := []string{podBefore.Name, podAfter.Name}
			claimNames := make([]string, 0, len(podNames))

			for _, podName := range podNames {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get %s pod %s", resourceType, podName)
				verifyPodRunningWithClaim(tCtx, b, pod, resourceName)

				// Collect claim name before cleanup
				if pod.Status.ExtendedResourceClaimStatus != nil {
					claimNames = append(claimNames, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName)
				}
			}

			// Clean up pods
			tCtx.Logf("Cleaning up %s resource pods after downgrade", resourceType)
			for _, podName := range podNames {
				tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), podName, namespace), "delete %s pod %s", resourceType, podName)
			}
			tCtx.Logf("Successfully cleaned up %s resource pods", resourceType)

			// Verify that ResourceClaims for our pods are cleaned up by garbage collector
			remainingClaims := claimNames
			tCtx.Eventually(func(tCtx ktesting.TContext) int {
				stillExisting := make([]string, 0, len(remainingClaims))
				for _, claimName := range remainingClaims {
					_, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
					if err == nil {
						stillExisting = append(stillExisting, claimName)
					} else if !apierrors.IsNotFound(err) {
						// If we get a real API error (not NotFound), fail immediately
						tCtx.ExpectNoError(err, "unexpected error checking ResourceClaim %s", claimName)
					}
					// If IsNotFound, the claim has been deleted (expected behavior)
				}
				remainingClaims = stillExisting
				return len(remainingClaims)
			}).WithTimeout(3*time.Minute).Should(gomega.Equal(0), "ResourceClaims for %s pods should be garbage collected", resourceType)

			tCtx.Logf("Verified ResourceClaims for %s resource were garbage collected", resourceType)
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
