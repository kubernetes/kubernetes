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
	"strings"
	"time"

	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/ktesting"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var skipAllNodeOperations = []resourceapi.SkipNodeOperation{resourceapi.SkipNodeOperationAll}

// optionalNodeOperationsResources publishes one device through the kubelet plugin.
// The plugin stays registered with kubelet, so unexpected NodePrepareResources
// and NodeUnprepareResources calls are observable.
func optionalNodeOperationsResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	return drautils.DriverResourcesWithSkipNodeOperationsNow(nodes, 1, skipAllNodeOperations...)
}

// optionalNodeOperationsClusterResources publishes a NodeSelector-based ResourceSlice not owned
// by the Node, so kubelet restarts do not wipe it and the apiserver preserves its existing
// skipNodeOperations field while the feature gate is disabled.
func optionalNodeOperationsClusterResources(nodes *drautils.Nodes) map[string]resourceslice.DriverResources {
	resources := drautils.NetworkResources(4, false)(nodes)
	drautils.SkipNodeOperations(skipAllNodeOperations...)(resources)
	return resources
}

// optionalNodeOperationsUpgradeDowngrade covers KEP-5945 scenario 3.2: a claim
// allocated with skipNodeOperations keeps working after a downgrade to a release
// where the feature is off by default.
func optionalNodeOperationsUpgradeDowngrade(nodes *drautils.Nodes) initialTestFunc {
	return func(tCtx ktesting.TContext, b *drautils.Builder) upgradedTestFunc {
		// Releases before 1.38 drop skipNodeOperations because the feature is off
		// by default there, so there is nothing to check yet.
		// TODO: once the previous release is 1.38 or newer, also create a pod here
		// to cover the feature staying enabled across upgrade and downgrade.
		return func(tCtx ktesting.TContext) downgradedTestFunc {
			// Re-publish so skipNodeOperations is restored: if the previous release dropped the
			// field, the resourceslice controller latched onto the stripped spec.
			b.Driver.PublishResources(tCtx, optionalNodeOperationsResources(nodes))
			expectSlicesSkipNodeOperations(tCtx, b, skipAllNodeOperations)

			claim, pod := createPodWithExternalClaim(tCtx, b)
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
			expectAllocationSkipNodeOperations(tCtx, claim, skipAllNodeOperations)
			expectNoNodeOperations(tCtx, b.Driver)

			return func(tCtx ktesting.TContext) {
				// Kubelet from the previous release still skips unprepare
				// because the decision is stored in its checkpoint.
				tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
				deletePodAndWait(tCtx, pod)
				expectNoNodeOperations(tCtx, b.Driver)
				deleteClaims(tCtx, claim)
			}
		}
	}
}

// optionalNodeOperationsGateCycle covers KEP-5945 scenario 3.2 for disabling the
// feature gate and scenario 3.3 for enabling it again:
//
//   - Phase 0 (gate ON): pods use devices which skip node operations.
//   - Phase 1 (gate OFF): running pods keep running and terminate without
//     NodeUnprepareResources. The ResourceSlice keeps skipNodeOperations, so the
//     scheduler must not allocate its devices.
//   - Phase 2 (gate ON again): the pending pod gets scheduled without waiting for
//     the 5-minute unschedulable pod flush, and new pods work.
func optionalNodeOperationsGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	var claimSurvivor, claimToDelete *resourceapi.ResourceClaim
	var podSurvivor, podToDelete *v1.Pod
	tCtx.Run("pods-skip-node-operations", func(tCtx ktesting.TContext) {
		claimSurvivor, podSurvivor = createPodWithExternalClaim(tCtx, b)
		claimToDelete, podToDelete = createPodWithExternalClaim(tCtx, b)
		for _, pod := range []*v1.Pod{podSurvivor, podToDelete} {
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
		}
		expectAllocationSkipNodeOperations(tCtx, claimSurvivor, skipAllNodeOperations)
		expectAllocationSkipNodeOperations(tCtx, claimToDelete, skipAllNodeOperations)
		expectNoNodeOperations(tCtx, b.Driver)
	})

	return func(tCtx ktesting.TContext) gateOnAgainFunc {
		tCtx.Run("running-pods-survive", func(tCtx ktesting.TContext) {
			for _, pod := range []*v1.Pod{podSurvivor, podToDelete} {
				tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
			}
		})

		tCtx.Run("slice-keeps-skip-node-operations", func(tCtx ktesting.TContext) {
			expectSlicesSkipNodeOperations(tCtx, b, skipAllNodeOperations)
		})

		var claimPending *resourceapi.ResourceClaim
		var podPending *v1.Pod
		tCtx.Run("new-pod-stays-pending", func(tCtx ktesting.TContext) {
			claimPending, podPending = createPodWithExternalClaim(tCtx, b)
			tCtx.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(tCtx, tCtx.Client(), podPending.Name, podPending.Namespace))
			tCtx.WithStep("pod stays unscheduled").
				Consistently(func(tCtx ktesting.TContext) string {
					pod, err := tCtx.Client().CoreV1().Pods(podPending.Namespace).Get(tCtx, podPending.Name, metav1.GetOptions{})
					tCtx.ExpectNoError(err, "get pod")
					return pod.Spec.NodeName
				}).WithTimeout(30 * time.Second).WithPolling(time.Second).Should(gomega.BeEmpty())
		})

		tCtx.Run("delete-pod-without-unprepare", func(tCtx ktesting.TContext) {
			deletePodAndWait(tCtx, podToDelete)
			expectNoNodeOperations(tCtx, b.Driver)
			deleteClaims(tCtx, claimToDelete)
		})

		return func(tCtx ktesting.TContext) {
			tCtx.Run("pending-pod-gets-scheduled", func(tCtx ktesting.TContext) {
				// Timeout < 5m unschedulable pod flush to verify event-driven requeueing.
				tCtx.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(tCtx, tCtx.Client(), podPending.Name, podPending.Namespace, 2*time.Minute))
				expectAllocationSkipNodeOperations(tCtx, claimPending, skipAllNodeOperations)
				expectNoNodeOperations(tCtx, b.Driver)
			})

			var claimFresh *resourceapi.ResourceClaim
			var podFresh *v1.Pod
			tCtx.Run("fresh-pod-skips-node-operations", func(tCtx ktesting.TContext) {
				claimFresh, podFresh = createPodWithExternalClaim(tCtx, b)
				tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), podFresh))
				expectAllocationSkipNodeOperations(tCtx, claimFresh, skipAllNodeOperations)
				expectNoNodeOperations(tCtx, b.Driver)
			})

			tCtx.Run("cleanup", func(tCtx ktesting.TContext) {
				for _, pod := range []*v1.Pod{podSurvivor, podPending, podFresh} {
					deletePodAndWait(tCtx, pod)
				}
				expectNoNodeOperations(tCtx, b.Driver)
				deleteClaims(tCtx, claimSurvivor, claimPending, claimFresh)
			})
		}
	}
}

func createPodWithExternalClaim(tCtx ktesting.TContext, b *drautils.Builder) (*resourceapi.ResourceClaim, *v1.Pod) {
	tCtx.Helper()
	claim := b.ExternalClaim()
	pod := b.PodExternal(claim.Name)
	b.Create(tCtx, claim, pod)
	return claim, pod
}

func expectSlicesSkipNodeOperations(tCtx ktesting.TContext, b *drautils.Builder, want []resourceapi.SkipNodeOperation) {
	tCtx.Helper()
	tCtx.WithStep("wait for skipNodeOperations in ResourceSlices").
		Eventually(b.Driver.NewGetSlices()).
		WithTimeout(time.Minute).
		Should(gomega.HaveField("Items", gomega.HaveEach(gomega.HaveField("Spec.SkipNodeOperations", gomega.Equal(want)))))
}

func expectAllocationSkipNodeOperations(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim, want []resourceapi.SkipNodeOperation) {
	tCtx.Helper()
	allocated, err := tCtx.Client().ResourceV1().ResourceClaims(tCtx.Namespace()).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get ResourceClaim %s", claim.Name)
	tCtx.Expect(allocated.Status.Allocation).NotTo(gomega.BeNil(), "ResourceClaim %s must be allocated", claim.Name)
	tCtx.Expect(allocated.Status.Allocation.Devices.Results).To(gomega.HaveEach(gomega.HaveField("SkipNodeOperations", gomega.Equal(want))), "allocation result of ResourceClaim %s", claim.Name)
}

// expectNoNodeOperations fails if kubelet called NodePrepareResources or
// NodeUnprepareResources on any of the driver's kubelet plugins. The devices
// in these tests skip all node operations, so neither call is ever expected.
func expectNoNodeOperations(tCtx ktesting.TContext, driver *drautils.Driver) {
	tCtx.Helper()
	tCtx.Expect(driver.Nodes).NotTo(gomega.BeEmpty(), "kubelet plugins of driver %s", driver.Name)
	for nodeName, plugin := range driver.Nodes {
		for _, call := range plugin.GetGRPCCalls() {
			if strings.HasSuffix(call.FullMethod, "/NodePrepareResources") || strings.HasSuffix(call.FullMethod, "/NodeUnprepareResources") {
				tCtx.Errorf("unexpected %s call on node %s with request %+v", call.FullMethod, nodeName, call.Request)
			}
		}
	}
}

// deletePodAndWait retries because the client may still hold a connection to a restarted apiserver.
func deletePodAndWait(tCtx ktesting.TContext, pod *v1.Pod) {
	tCtx.Helper()
	tCtx.Eventually(func(tCtx ktesting.TContext) error {
		err := tCtx.Client().CoreV1().Pods(pod.Namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}).Should(gomega.Succeed(), "delete pod %s", pod.Name)
	tCtx.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(tCtx, tCtx.Client(), pod.Name, pod.Namespace, 3*time.Minute), "pod %s should terminate", pod.Name)
}

// deleteClaims retries because the client may still hold a connection to a restarted apiserver.
func deleteClaims(tCtx ktesting.TContext, claims ...*resourceapi.ResourceClaim) {
	tCtx.Helper()
	for _, claim := range claims {
		tCtx.Eventually(func(tCtx ktesting.TContext) error {
			err := tCtx.Client().ResourceV1().ResourceClaims(tCtx.Namespace()).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				return nil
			}
			return err
		}).Should(gomega.Succeed(), "delete ResourceClaim %s", claim.Name)
	}
}
