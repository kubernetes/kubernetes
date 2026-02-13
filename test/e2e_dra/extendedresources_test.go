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
	"time"

	"github.com/onsi/gomega"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// createPodWithExtendedResourceAndVerifyClaim creates a pod requesting an extended resource,
// waits for it to run, and verifies the scheduler created a special ResourceClaim for it.
// Returns the created pod and the special ResourceClaim.
func createPodWithExtendedResourceAndVerifyClaim(tCtx ktesting.TContext, pod *v1.Pod, resourceName v1.ResourceName, namespace string) (*v1.Pod, *resourceapi.ResourceClaim) {
	// Set resource requests and limits
	pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{
		resourceName: resource.MustParse("1"),
	}
	pod.Spec.Containers[0].Resources.Limits = v1.ResourceList{
		resourceName: resource.MustParse("1"),
	}

	// Create the pod
	createdPod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod requesting extended resource %s", resourceName)
	tCtx.Logf("Created pod %q requesting extended resource %q", createdPod.Name, resourceName)

	// Wait for pod to be scheduled and running
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), createdPod), "wait for pod to run")
	tCtx.Logf("Pod %q is running", createdPod.Name)

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

	pod, specialClaim := createPodWithExtendedResourceAndVerifyClaim(tCtx, pod, resourceName, namespace)

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
	foundMapping := false
	for _, reqMapping := range pod.Status.ExtendedResourceClaimStatus.RequestMappings {
		if reqMapping.ResourceName == string(resourceName) {
			foundMapping = true
			tCtx.Logf("Found request mapping: container=%s, resource=%s, request=%s",
				reqMapping.ContainerName, reqMapping.ResourceName, reqMapping.RequestName)
			break
		}
	}
	require.True(tCtx, foundMapping, "extendedResourceClaimStatus.RequestMappings should contain %s", resourceName)

	return pod
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
	namespace := tCtx.Namespace()

	// Test 1: Explicit extended resource name
	// DeviceClass with extendedResourceName field explicitly set to a custom value
	b.UseExtendedResourceName = true
	deviceClass1 := b.Class(drautils.SingletonIndex)
	createdObjs := b.Create(tCtx, deviceClass1)
	deviceClass1 = createdObjs[0].(*resourceapi.DeviceClass)
	resourceName1 := v1.ResourceName(*deviceClass1.Spec.ExtendedResourceName)
	tCtx.Logf("Created DeviceClass %q with explicit extendedResourceName: %s", deviceClass1.Name, resourceName1)

	pod1 := testExtendedResourcePod(tCtx, b, resourceName1, "extended-resource-pod-1")

	// Test 2: Implicit extended resource name
	// This tests the implicit naming: deviceclass.resource.kubernetes.io/<class-name>
	// using the DeviceClass created by builder.setUp()
	resourceName2 := v1.ResourceName(fmt.Sprintf("deviceclass.resource.kubernetes.io/%s", b.ClassName()))
	tCtx.Logf("Using implicit extended resource name: %s", resourceName2)

	pod2 := testExtendedResourcePod(tCtx, b, resourceName2, "extended-resource-pod-2")

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade: feature is still enabled
		// Verify existing pods still run correctly
		pod1, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod1.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get pod1 after upgrade")
		require.Equal(tCtx, v1.PodRunning, pod1.Status.Phase, "pod1 should still be running after upgrade")
		tCtx.Logf("Verified pod %q (explicit resource) still running after upgrade", pod1.Name)

		pod2, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod2.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get pod2 after upgrade")
		require.Equal(tCtx, v1.PodRunning, pod2.Status.Phase, "pod2 should still be running after upgrade")
		tCtx.Logf("Verified pod %q (implicit resource) still running after upgrade", pod2.Name)

		// Create another pod with explicit extended resource request to verify scheduler still works
		pod3 := testExtendedResourcePod(tCtx, b, resourceName1, "extended-resource-pod-3")

		return func(tCtx ktesting.TContext) {
			// After downgrade: feature should be disabled or degraded
			// Existing pods should continue running (CDI devices remain prepared)
			podNames := []string{pod1.Name, pod2.Name, pod3.Name}
			for _, podName := range podNames {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
				tCtx.ExpectNoError(err, "get pod %s after downgrade", podName)
				if pod.Status.Phase != v1.PodRunning {
					tCtx.Logf("Warning: pod %q is in phase %s after downgrade (expected Running)", podName, pod.Status.Phase)
				} else {
					tCtx.Logf("Pod %q still running after downgrade", podName)
				}
			}

			// Clean up all pods
			tCtx.Logf("Cleaning up pods after downgrade")
			for _, podName := range podNames {
				tCtx.ExpectNoError(e2epod.DeletePodWithWaitByName(tCtx, tCtx.Client(), podName, namespace), "delete pod %s", podName)
			}
			tCtx.Logf("Successfully cleaned up all pods")

			// Verify that special ResourceClaims are cleaned up by garbage collector
			tCtx.Eventually(func(tCtx ktesting.TContext) int {
				claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
				if err != nil {
					tCtx.Logf("Error listing claims: %v", err)
					return -1
				}
				specialClaimCount := 0
				for _, claim := range claims.Items {
					for _, ownerRef := range claim.OwnerReferences {
						if ownerRef.Kind == "Pod" {
							specialClaimCount++
							break
						}
					}
				}
				return specialClaimCount
			}).WithTimeout(3*time.Minute).Should(gomega.Equal(0), "special ResourceClaims should be garbage collected")

			tCtx.Logf("Verified special ResourceClaims were garbage collected")
			tCtx.Logf("Test completed successfully")
		}
	}
}
