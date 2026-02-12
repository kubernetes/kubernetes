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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// extendedResourcesUpgradeDowngrade tests the DRAExtendedResources feature during upgrade/downgrade scenarios.
// This test verifies that:
// 1. DeviceClass with explicit extendedResourceName field (custom resource name)
// 2. DeviceClass with implicit extendedResourceName (using default format)
// 3. Pods requesting extended resources are scheduled and allocated devices
// 4. The scheduler creates special ResourceClaims for extended resource requests
// 5. Pod status contains extendedResourceClaimStatus mapping
// 6. Feature works correctly across upgrade/downgrade cycles
func extendedResourcesUpgradeDowngrade(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
	namespace := tCtx.Namespace()

	// Test 1: Explicit extended resource name
	// DeviceClass with extendedResourceName field explicitly set to a custom value
	b.UseExtendedResourceName = true
	deviceClass1 := b.Class(drautils.SingletonIndex)
	deviceClass1, err := tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, deviceClass1, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create DeviceClass with explicit extendedResourceName")

	// Check if the ExtendedResourceName field is supported (may be nil in older versions)
	if deviceClass1.Spec.ExtendedResourceName == nil {
		tCtx.Logf("ExtendedResourceName feature not supported in this version, skipping extended resources test")
		// Return no-op functions for upgrade and downgrade phases
		return func(tCtx ktesting.TContext) downgradedTestFunc {
			return func(tCtx ktesting.TContext) {
				// Cleanup the DeviceClass we created
				tCtx.Eventually(func(tCtx ktesting.TContext) error {
					return tCtx.Client().ResourceV1().DeviceClasses().Delete(tCtx, deviceClass1.Name, metav1.DeleteOptions{})
				}).Should(gomega.Succeed(), "delete DeviceClass")
			}
		}
	}

	tCtx.Logf("Created DeviceClass %q with explicit extendedResourceName: %s", deviceClass1.Name, *deviceClass1.Spec.ExtendedResourceName)

	// Create a pod that requests the extended resource via container.resources.requests
	// The scheduler should automatically create a ResourceClaim for this pod
	pod1 := b.Pod()
	pod1.Name = "extended-resource-pod-1"
	resourceName1 := v1.ResourceName(*deviceClass1.Spec.ExtendedResourceName)
	pod1.Spec.Containers[0].Resources.Requests = v1.ResourceList{
		resourceName1: resource.MustParse("1"),
	}
	pod1.Spec.Containers[0].Resources.Limits = v1.ResourceList{
		resourceName1: resource.MustParse("1"),
	}

	pod1, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod1, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod requesting explicit extended resource")
	tCtx.Logf("Created pod %q requesting explicit extended resource %q", pod1.Name, resourceName1)

	// Wait for pod to be scheduled and running
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod1), "wait for pod to run")

	// Verify the scheduler created a special ResourceClaim for this pod
	var specialClaim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		// The special claim should be named after the pod and extended resource
		claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		if err != nil {
			tCtx.Logf("Error listing claims: %v", err)
			return false
		}
		for _, claim := range claims.Items {
			// Special claims created by scheduler have owner reference to the pod
			for _, ownerRef := range claim.OwnerReferences {
				if ownerRef.UID == pod1.UID && ownerRef.Kind == "Pod" {
					specialClaim = &claim
					return true
				}
			}
		}
		return false
	}).Should(gomega.BeTrueBecause("scheduler should create special ResourceClaim"))

	tCtx.Logf("Found special ResourceClaim %q created by scheduler for pod %q", specialClaim.Name, pod1.Name)

	// Verify the special claim is allocated
	require.NotNil(tCtx, specialClaim.Status.Allocation, "special ResourceClaim should be allocated")
	tCtx.Logf("Special ResourceClaim %q is allocated", specialClaim.Name)

	// Get updated pod to check status
	pod1, err = tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod1.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get pod to check status")

	// Verify pod.status.extendedResourceClaimStatus contains the mapping
	require.NotNil(tCtx, pod1.Status.ExtendedResourceClaimStatus, "pod.status.extendedResourceClaimStatus should not be nil")

	// Verify the ResourceClaimName matches the special claim
	require.Equal(tCtx, specialClaim.Name, pod1.Status.ExtendedResourceClaimStatus.ResourceClaimName,
		"extendedResourceClaimStatus.ResourceClaimName should match special claim")
	tCtx.Logf("Verified extendedResourceClaimStatus.ResourceClaimName: %s",
		pod1.Status.ExtendedResourceClaimStatus.ResourceClaimName)

	// Verify request mappings contain the extended resource
	foundMapping := false
	for _, reqMapping := range pod1.Status.ExtendedResourceClaimStatus.RequestMappings {
		if reqMapping.ResourceName == string(resourceName1) {
			foundMapping = true
			tCtx.Logf("Found request mapping: container=%s, resource=%s, request=%s",
				reqMapping.ContainerName, reqMapping.ResourceName, reqMapping.RequestName)
			break
		}
	}
	require.True(tCtx, foundMapping, "extendedResourceClaimStatus.RequestMappings should contain %s", resourceName1)

	// Test 2: Implicit extended resource name
	// Create DeviceClass WITHOUT setting extendedResourceName field
	// This tests the implicit naming: deviceclass.resource.kubernetes.io/<class-name>
	// Use index 1 to avoid conflict with the default DeviceClass created by builder.setUp() which uses index 0
	b.UseExtendedResourceName = false
	deviceClass2 := b.Class(1)
	deviceClass2, err = tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, deviceClass2, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create DeviceClass for implicit extended resource")
	tCtx.Logf("Created DeviceClass %q for implicit extended resource testing", deviceClass2.Name)

	// The implicit extended resource name format
	resourceName2 := v1.ResourceName(fmt.Sprintf("deviceclass.resource.kubernetes.io/%s", deviceClass2.Name))
	tCtx.Logf("Using implicit extended resource name: %s", resourceName2)

	// Create a pod that requests the extended resource
	pod3 := b.Pod()
	pod3.Name = "extended-resource-pod-2"
	pod3.Spec.Containers[0].Resources.Requests = v1.ResourceList{
		resourceName2: resource.MustParse("1"),
	}
	pod3.Spec.Containers[0].Resources.Limits = v1.ResourceList{
		resourceName2: resource.MustParse("1"),
	}

	pod3, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod3, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod requesting implicit extended resource")
	tCtx.Logf("Created pod %q requesting implicit extended resource %q", pod3.Name, resourceName2)

	// Wait for pod to be scheduled and running
	tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod3), "wait for pod to run")
	tCtx.Logf("Pod %q with implicit extended resource is running", pod3.Name)

	// Verify scheduler created special ResourceClaim for implicit resource
	var claim3 *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) bool {
		claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
		if err != nil {
			tCtx.Logf("Error listing claims: %v", err)
			return false
		}
		for _, claim := range claims.Items {
			for _, ownerRef := range claim.OwnerReferences {
				if ownerRef.UID == pod3.UID && ownerRef.Kind == "Pod" {
					claim3 = &claim
					return true
				}
			}
		}
		return false
	}).Should(gomega.BeTrueBecause("scheduler should create special ResourceClaim for implicit resource"))

	tCtx.Logf("Found special ResourceClaim %q for implicit extended resource", claim3.Name)

	return func(tCtx ktesting.TContext) downgradedTestFunc {
		// After upgrade: feature is still enabled
		// Verify existing pods still run correctly
		pod1, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod1.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get pod1 after upgrade")
		require.Equal(tCtx, v1.PodRunning, pod1.Status.Phase, "pod1 should still be running after upgrade")
		tCtx.Logf("Verified pod %q (explicit resource) still running after upgrade", pod1.Name)

		pod3, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod3.Name, metav1.GetOptions{})
		tCtx.ExpectNoError(err, "get pod3 after upgrade")
		require.Equal(tCtx, v1.PodRunning, pod3.Status.Phase, "pod3 should still be running after upgrade")
		tCtx.Logf("Verified pod %q (implicit resource) still running after upgrade", pod3.Name)

		// Create another pod with explicit extended resource request to verify scheduler still works
		b.UseExtendedResourceName = true
		pod2 := b.Pod()
		pod2.Name = "extended-resource-pod-3"
		pod2.Spec.Containers[0].Resources.Requests = v1.ResourceList{
			resourceName1: resource.MustParse("1"),
		}
		pod2.Spec.Containers[0].Resources.Limits = v1.ResourceList{
			resourceName1: resource.MustParse("1"),
		}

		pod2, err = tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod2, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create pod with explicit resource after upgrade")
		tCtx.Logf("Created pod %q with explicit extended resource after upgrade", pod2.Name)

		// Wait for second pod to be scheduled and running
		tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod2), "wait for second pod to run")
		tCtx.Logf("Pod %q is running after upgrade", pod2.Name)

		// Verify scheduler created a special claim for the second pod
		var claim2 *resourceapi.ResourceClaim
		tCtx.Eventually(func(tCtx ktesting.TContext) bool {
			claims, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).List(tCtx, metav1.ListOptions{})
			if err != nil {
				tCtx.Logf("Error listing claims: %v", err)
				return false
			}
			for _, claim := range claims.Items {
				for _, ownerRef := range claim.OwnerReferences {
					if ownerRef.UID == pod2.UID && ownerRef.Kind == "Pod" {
						claim2 = &claim
						return true
					}
				}
			}
			return false
		}).Should(gomega.BeTrueBecause("scheduler should create special ResourceClaim"))

		tCtx.Logf("Found special ResourceClaim %q for pod %q", claim2.Name, pod2.Name)

		return func(tCtx ktesting.TContext) {
			// After downgrade: feature should be disabled or degraded
			// Existing pods should continue running (CDI devices remain prepared)
			pod1, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod1.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod1 after downgrade")
			if pod1.Status.Phase != v1.PodRunning {
				tCtx.Logf("Warning: pod %q is in phase %s after downgrade (expected Running)", pod1.Name, pod1.Status.Phase)
			} else {
				tCtx.Logf("Pod %q still running after downgrade", pod1.Name)
			}

			pod2, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod2.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod2 after downgrade")
			if pod2.Status.Phase != v1.PodRunning {
				tCtx.Logf("Warning: pod %q is in phase %s after downgrade (expected Running)", pod2.Name, pod2.Status.Phase)
			} else {
				tCtx.Logf("Pod %q still running after downgrade", pod2.Name)
			}

			pod3, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod3.Name, metav1.GetOptions{})
			tCtx.ExpectNoError(err, "get pod3 after downgrade")
			if pod3.Status.Phase != v1.PodRunning {
				tCtx.Logf("Warning: pod %q is in phase %s after downgrade (expected Running)", pod3.Name, pod3.Status.Phase)
			} else {
				tCtx.Logf("Pod %q still running after downgrade", pod3.Name)
			}

			// Clean up all pods
			tCtx.Logf("Cleaning up pods after downgrade")
			tCtx.Eventually(func(tCtx ktesting.TContext) error {
				return tCtx.Client().CoreV1().Pods(namespace).Delete(tCtx, pod1.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete pod1 after downgrade")
			tCtx.Eventually(func(tCtx ktesting.TContext) error {
				return tCtx.Client().CoreV1().Pods(namespace).Delete(tCtx, pod2.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete pod2 after downgrade")
			tCtx.Eventually(func(tCtx ktesting.TContext) error {
				return tCtx.Client().CoreV1().Pods(namespace).Delete(tCtx, pod3.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete pod3 after downgrade")

			// Wait for pods to be fully deleted
			tCtx.Eventually(func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod1.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return nil
				}
				tCtx.ExpectNoError(err, "get pod1 during cleanup")
				return pod
			}).WithTimeout(3*time.Minute).Should(gomega.BeNil(), "pod1 should be deleted")

			tCtx.Eventually(func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod2.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return nil
				}
				tCtx.ExpectNoError(err, "get pod2 during cleanup")
				return pod
			}).WithTimeout(3*time.Minute).Should(gomega.BeNil(), "pod2 should be deleted")

			tCtx.Eventually(func(tCtx ktesting.TContext) *v1.Pod {
				pod, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod3.Name, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return nil
				}
				tCtx.ExpectNoError(err, "get pod3 during cleanup")
				return pod
			}).WithTimeout(3*time.Minute).Should(gomega.BeNil(), "pod3 should be deleted")

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

			// Clean up both DeviceClasses (explicit and implicit)
			tCtx.Eventually(func(tCtx ktesting.TContext) error {
				return tCtx.Client().ResourceV1().DeviceClasses().Delete(tCtx, deviceClass1.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete DeviceClass with explicit extendedResourceName")

			tCtx.Eventually(func(tCtx ktesting.TContext) error {
				return tCtx.Client().ResourceV1().DeviceClasses().Delete(tCtx, deviceClass2.Name, metav1.DeleteOptions{})
			}).Should(gomega.Succeed(), "delete DeviceClass with implicit extendedResourceName")

			tCtx.Logf("Test completed successfully")
		}
	}
}
