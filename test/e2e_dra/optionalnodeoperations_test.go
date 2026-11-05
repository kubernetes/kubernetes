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

// optionalNodeOperationsClusterResources publishes a ResourceSlice with a node selector.
// Kubelet doesn't own it, so it doesn't get wiped when kubelet restarts and keeps
// skipNodeOperations while the feature is disabled.
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
		return func(tCtx ktesting.TContext) downgradedTestFunc {
			// The resourceslice controller keeps publishing what the previous release
			// stored, so publish again now that the apiserver keeps the field.
			b.Driver.PublishResources(tCtx, optionalNodeOperationsResources(nodes))
			expectSlicesSkipNodeOperations(tCtx, b, skipAllNodeOperations)

			plugin := b.Driver.Nodes[nodes.NodeNames[0]]
			numCalls := len(plugin.GetGRPCCalls())
			claim, pod := createPodWithExternalClaim(tCtx, b)
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
			expectAllocationSkipNodeOperations(tCtx, claim, skipAllNodeOperations)
			expectNoNodeOperation(tCtx, plugin, numCalls, "NodePrepareResources")

			return func(tCtx ktesting.TContext) {
				// Kubelet from the previous release still skips unprepare
				// because the decision is stored in its checkpoint.
				tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
				numCalls := len(plugin.GetGRPCCalls())
				deletePodAndWait(tCtx, pod)
				expectNoNodeOperation(tCtx, plugin, numCalls, "NodeUnprepareResources")
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
//     the periodic retry of unschedulable pods, and new pods work.
func optionalNodeOperationsGateCycle(tCtx ktesting.TContext, b *drautils.Builder) gateOffFunc {
	nodeNames := b.Driver.Nodenames()
	tCtx.Expect(nodeNames).To(gomega.HaveLen(1), "nodes with a kubelet plugin")
	plugin := b.Driver.Nodes[nodeNames[0]]

	var claimSurvivor, claimToDelete *resourceapi.ResourceClaim
	var podSurvivor, podToDelete *v1.Pod
	tCtx.Run("pods-skip-node-operations", func(tCtx ktesting.TContext) {
		numCalls := len(plugin.GetGRPCCalls())
		claimSurvivor, podSurvivor = createPodWithExternalClaim(tCtx, b)
		claimToDelete, podToDelete = createPodWithExternalClaim(tCtx, b)
		for _, pod := range []*v1.Pod{podSurvivor, podToDelete} {
			tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), pod))
		}
		expectAllocationSkipNodeOperations(tCtx, claimSurvivor, skipAllNodeOperations)
		expectAllocationSkipNodeOperations(tCtx, claimToDelete, skipAllNodeOperations)
		expectNoNodeOperation(tCtx, plugin, numCalls, "NodePrepareResources")
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
		var numCallsBeforePending int
		tCtx.Run("new-pod-stays-pending", func(tCtx ktesting.TContext) {
			numCallsBeforePending = len(plugin.GetGRPCCalls())
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
			numCalls := len(plugin.GetGRPCCalls())
			deletePodAndWait(tCtx, podToDelete)
			expectNoNodeOperation(tCtx, plugin, numCalls, "NodeUnprepareResources")
			deleteClaims(tCtx, claimToDelete)
		})

		return func(tCtx ktesting.TContext) {
			tCtx.Run("pending-pod-gets-scheduled", func(tCtx ktesting.TContext) {
				// Much shorter than the periodic retry of unschedulable pods (five minutes).
				tCtx.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(tCtx, tCtx.Client(), podPending.Name, podPending.Namespace, 2*time.Minute))
				expectAllocationSkipNodeOperations(tCtx, claimPending, skipAllNodeOperations)
				expectNoNodeOperation(tCtx, plugin, numCallsBeforePending, "NodePrepareResources")
			})

			var claimFresh *resourceapi.ResourceClaim
			var podFresh *v1.Pod
			tCtx.Run("fresh-pod-skips-node-operations", func(tCtx ktesting.TContext) {
				numCalls := len(plugin.GetGRPCCalls())
				claimFresh, podFresh = createPodWithExternalClaim(tCtx, b)
				tCtx.ExpectNoError(e2epod.WaitForPodRunningInNamespace(tCtx, tCtx.Client(), podFresh))
				expectAllocationSkipNodeOperations(tCtx, claimFresh, skipAllNodeOperations)
				expectNoNodeOperation(tCtx, plugin, numCalls, "NodePrepareResources")
			})

			tCtx.Run("cleanup", func(tCtx ktesting.TContext) {
				numCalls := len(plugin.GetGRPCCalls())
				for _, pod := range []*v1.Pod{podSurvivor, podPending, podFresh} {
					deletePodAndWait(tCtx, pod)
				}
				expectNoNodeOperation(tCtx, plugin, numCalls, "NodeUnprepareResources")
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

// expectSlicesSkipNodeOperations waits until all ResourceSlices of the driver have the given skipNodeOperations.
func expectSlicesSkipNodeOperations(tCtx ktesting.TContext, b *drautils.Builder, want []resourceapi.SkipNodeOperation) {
	tCtx.Helper()
	tCtx.WithStep("wait for skipNodeOperations in ResourceSlices").
		Eventually(b.Driver.NewGetSlices()).
		WithTimeout(time.Minute).
		Should(gomega.HaveField("Items", gomega.HaveEach(gomega.HaveField("Spec.SkipNodeOperations", gomega.Equal(want)))))
}

// expectAllocationSkipNodeOperations checks what the scheduler copied into the allocation result.
func expectAllocationSkipNodeOperations(tCtx ktesting.TContext, claim *resourceapi.ResourceClaim, want []resourceapi.SkipNodeOperation) {
	tCtx.Helper()
	allocated, err := tCtx.Client().ResourceV1().ResourceClaims(tCtx.Namespace()).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get ResourceClaim %s", claim.Name)
	tCtx.Expect(allocated.Status.Allocation).NotTo(gomega.BeNil(), "ResourceClaim %s must be allocated", claim.Name)
	tCtx.Expect(allocated.Status.Allocation.Devices.Results).To(gomega.HaveEach(gomega.HaveField("SkipNodeOperations", gomega.Equal(want))), "allocation result of ResourceClaim %s", claim.Name)
}

// expectNoNodeOperation fails if kubelet called the given gRPC method after the first n recorded calls.
func expectNoNodeOperation(tCtx ktesting.TContext, plugin drautils.KubeletPlugin, n int, method string) {
	tCtx.Helper()
	for _, call := range plugin.GetGRPCCalls()[n:] {
		if strings.HasSuffix(call.FullMethod, "/"+method) {
			tCtx.Errorf("unexpected %s call with request %+v", call.FullMethod, call.Request)
		}
	}
}

// deletePodAndWait retries because the client may still have a connection to an apiserver which got restarted.
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

// deleteClaims retries because the client may still have a connection to an apiserver which got restarted.
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
