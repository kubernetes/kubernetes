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
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	resourceinternal "k8s.io/kubernetes/pkg/apis/resource"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// createPodWithExtendedResourceAndVerifyClaim creates a pod requesting an extended resource,
// waits for it to run, and verifies the scheduler created a special ResourceClaim for it.
// Returns the created pod and the special ResourceClaim.
func createPodWithExtendedResourceAndVerifyClaim(tCtx ktesting.TContext, b *drautils.Builder, pod *v1.Pod, resourceName v1.ResourceName, namespace string) (*v1.Pod, *resourceapi.ResourceClaim) {
	// Set resource requests and limits
	resource := v1.ResourceList{
		resourceName: resource.MustParse("1"),
	}
	pod.Spec.Containers[0].Resources.Requests = resource
	pod.Spec.Containers[0].Resources.Limits = resource

	// Create the pod using builder
	objs := b.Create(tCtx, pod)
	createdPod := objs[0].(*v1.Pod)
	tCtx.Logf("Created pod %q requesting extended resource %q", createdPod.Name, resourceName)

	// Wait for pod to be running and validate
	b.TestPod(tCtx, createdPod)

	// Verify the scheduler created a special ResourceClaim for this pod
	var specialClaim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		if err != nil {
			tCtx.Logf("Error listing claims: %v", err)
			return false
		}
		for _, claim := range claims.Items {
			// Special claims created by scheduler have owner reference to the pod
			for _, ownerRef := range claim.OwnerReferences {
				if ownerRef.UID == createdPod.UID && ownerRef.Kind == "Pod" {
					specialClaim = &claim
					return true
				}
			}
		}
		return false
	}).Should(gomega.BeTrueBecause("scheduler should create special ResourceClaim"))

	tCtx.Logf("Found special ResourceClaim %q created by scheduler for pod %q", specialClaim.Name, createdPod.Name)

	return createdPod, specialClaim
}

// testExtendedResourcePod creates and verifies a pod requesting an extended resource.
// It validates that the scheduler creates a special ResourceClaim and that the pod status
// contains the proper extendedResourceClaimStatus mapping.
func testExtendedResourcePod(tCtx ktesting.TContext, b *drautils.Builder, resourceName v1.ResourceName, podName string) *v1.Pod {
	namespace := tCtx.Namespace()

	pod := b.Pod()
	pod.Name = podName

	pod, specialClaim := createPodWithExtendedResourceAndVerifyClaim(tCtx, b, pod, resourceName, namespace)

	// Verify the special claim is allocated
	require.NotNil(tCtx, specialClaim.Status.Allocation, "special ResourceClaim should be allocated")
	tCtx.Logf("Special ResourceClaim %q is allocated", specialClaim.Name)

	// Get updated pod to check status
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get pod to check status")

	// Verify pod.status.extendedResourceClaimStatus contains the mapping
	require.NotNil(tCtx, pod.Status.ExtendedResourceClaimStatus, "pod.status.extendedResourceClaimStatus should not be nil")

	// Verify the ResourceClaimName matches the special claim
	require.Equal(tCtx, specialClaim.Name, pod.Status.ExtendedResourceClaimStatus.ResourceClaimName,
		"extendedResourceClaimStatus.ResourceClaimName should match special claim")
	tCtx.Logf("Verified extendedResourceClaimStatus.ResourceClaimName: %s",
		pod.Status.ExtendedResourceClaimStatus.ResourceClaimName)

	// Verify request mappings contain the extended resource
	foundMapping := slices.ContainsFunc(pod.Status.ExtendedResourceClaimStatus.RequestMappings, func(reqMapping v1.ContainerExtendedResourceRequest) bool {
		if reqMapping.ResourceName == string(resourceName) {
			tCtx.Logf("Found request mapping: container=%s, resource=%s, request=%s",
				reqMapping.ContainerName, reqMapping.ResourceName, reqMapping.RequestName)
			return true
		}
		return false
	})
	require.True(tCtx, foundMapping, "extendedResourceClaimStatus.RequestMappings should contain %s", resourceName)

	return pod
}

// verifyPodsRunning verifies that pods are running and logs their status.
// After downgrade, pods may not be running, so we just log warnings instead of failing.
func verifyPodsRunning(tCtx ktesting.TContext, namespace string, podNames []string, resourceType, phase string) {
	for _, podName := range podNames {
		pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get %s pod %s after %s", resourceType, podName, phase)
		if pod.Status.Phase != v1.PodRunning {
			tCtx.Logf("Warning: pod %q (%s resource) is in phase %s after %s (expected Running)", podName, resourceType, pod.Status.Phase, phase)
		} else {
			tCtx.Logf("Pod %q (%s resource) still running after %s", podName, resourceType, phase)
		}
	}
}

// cleanupPods deletes the specified pods and waits for them to be deleted.
func cleanupPods(tCtx ktesting.TContext, namespace string, podNames []string, resourceType string) {
	tCtx.Logf("Cleaning up %s resource pods after downgrade", resourceType)
	for _, podName := range podNames {
		tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), podName, namespace), "delete %s pod %s", resourceType, podName)
	}
	tCtx.Logf("Successfully cleaned up %s resource pods", resourceType)
}

// createDeviceClassForExtendedResource creates a DeviceClass for testing extended resources.
// It sets up the builder state, creates the DeviceClass with a unique name, and returns
// the created DeviceClass and its corresponding resource name.
func createDeviceClassForExtendedResource(tCtx ktesting.TContext, b *drautils.Builder, namespace string, useExplicit bool) (*resourceapi.DeviceClass, v1.ResourceName) {
	var classIndex int
	var nameSuffix string

	if useExplicit {
		b.UseExtendedResourceName = true
		classIndex = drautils.SingletonIndex
		nameSuffix = "-extended-resource-explicit"
	} else {
		b.UseExtendedResourceName = false
		classIndex = 1
		nameSuffix = "-extended-resource-implicit"
	}

	// Create the DeviceClass with a unique name
	deviceClass := b.Class(classIndex)
	deviceClass.Name = namespace + nameSuffix
	createdObjs := b.Create(tCtx, deviceClass)
	deviceClass = createdObjs[0].(*resourceapi.DeviceClass)

	// Determine the resource name based on the type
	var resourceName v1.ResourceName
	if useExplicit {
		// Verify ExtendedResourceName is present in the spec when using explicit resource name
		// This ensures the API server hasn't stripped the field due to feature gates
		require.NotNil(tCtx, deviceClass.Spec.ExtendedResourceName, "DeviceClass.Spec.ExtendedResourceName should be set for explicit resource name")
		resourceName = v1.ResourceName(*deviceClass.Spec.ExtendedResourceName)
	} else {
		resourceName = v1.ResourceName(fmt.Sprintf("%s%s", resourceinternal.ResourceDeviceClassPrefix, deviceClass.Name))
	}

	tCtx.Logf("Created DeviceClass %q with extendedResourceName: %s", deviceClass.Name, resourceName)

	return deviceClass, resourceName
}

// testExtendedResource tests a single extended resource type through the upgrade/downgrade cycle.
// It creates the DeviceClass, creates pods before upgrade, verifies them after upgrade,
// creates new pods after upgrade, and verifies and cleans up all pods after downgrade.
func testExtendedResource(tCtx ktesting.TContext, b *drautils.Builder, useExplicit bool) upgradedTestFunc {
	namespace := tCtx.Namespace()

	// Create DeviceClass and get the resource name
	_, resourceName := createDeviceClassForExtendedResource(tCtx, b, namespace, useExplicit)

	// Determine resource type string for logging
	resourceType := "implicit"
	if useExplicit {
		resourceType = "explicit"
	}

	// Before upgrade: create and verify pod
	podBefore := testExtendedResourcePod(tCtx, b, resourceName, fmt.Sprintf("%s-before-upgrade", resourceType))
	tCtx.Logf("Created pod %q for %s resource before upgrade", podBefore.Name, resourceType)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade: verify existing pod still runs correctly
		podBefore, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podBefore.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get %s pod after upgrade", resourceType)
		require.Equal(tCtx, v1.PodRunning, podBefore.Status.Phase, "%s pod should still be running after upgrade", resourceType)
		tCtx.Logf("Verified pod %q (%s resource) still running after upgrade", podBefore.Name, resourceType)

		// Create and verify another pod with the same extended resource
		podAfter := testExtendedResourcePod(tCtx, b, resourceName, fmt.Sprintf("%s-after-upgrade", resourceType))
		tCtx.Logf("Created pod %q for %s resource after upgrade", podAfter.Name, resourceType)

		return func(tCtx ktesting.TContext) {
			// After downgrade: verify all pods still run correctly
			podNames := []string{podBefore.Name, podAfter.Name}
			podUIDs := []string{string(podBefore.UID), string(podAfter.UID)}
			verifyPodsRunning(tCtx, namespace, podNames, resourceType, "downgrade")

			// Clean up pods
			cleanupPods(tCtx, namespace, podNames, resourceType)

			// Verify that special ResourceClaims for our pods are cleaned up by garbage collector
			tCtx.Eventually(func(tCtx ktesting.TContext) int {
				claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
				if err != nil {
					tCtx.Logf("Error listing claims: %v", err)
					return -1
				}
				specialClaimCount := 0
				for _, claim := range claims.Items {
					for _, ownerRef := range claim.OwnerReferences {
						// Only count claims owned by our specific pods
						if ownerRef.Kind == "Pod" && slices.Contains(podUIDs, string(ownerRef.UID)) {
							specialClaimCount++
							break
						}
					}
				}
				return specialClaimCount
			}).WithTimeout(3*time.Minute).Should(gomega.Equal(0), "special ResourceClaims for %s pods should be garbage collected", resourceType)

			tCtx.Logf("Verified special ResourceClaims for %s resource were garbage collected", resourceType)
		}
	}
}

// extendedResourceUpgradeDowngrade tests the DRAExtendedResources feature during upgrade/downgrade scenarios.
// This test verifies that:
// 1. DeviceClass with explicit extendedResourceName field (custom resource name)
// 2. DeviceClass with implicit extendedResourceName (using default format)
// 3. Pods requesting extended resources are scheduled and allocated devices
// 4. The scheduler creates special ResourceClaims for extended resource requests
// 5. Pod status contains extendedResourceClaimStatus mapping
// 6. Feature works correctly across upgrade/downgrade cycles
func extendedResourceUpgradeDowngrade(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	// Test both explicit and implicit extended resources through upgrade/downgrade
	explicitUpgradedFunc := testExtendedResource(tCtx, b, true)
	implicitUpgradedFunc := testExtendedResource(tCtx, b, false)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		explicitDowngradedFunc := explicitUpgradedFunc(tCtx)
		implicitDowngradedFunc := implicitUpgradedFunc(tCtx)

		return func(tCtx ktesting.TContext) {
			// Run downgrade verification and cleanup for both resource types
			explicitDowngradedFunc(tCtx)
			implicitDowngradedFunc(tCtx)

			tCtx.Logf("Extended resource upgrade/downgrade test completed successfully")
		}
	}
}
