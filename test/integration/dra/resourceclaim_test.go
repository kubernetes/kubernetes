/*
Copyright 2025 The Kubernetes Authors.

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
	"sync"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// testShareResourceClaim is the former "sharing a claim sequentially" test which was removed from E2E testing
// in https://github.com/kubernetes/kubernetes/pull/133214. It creates a single ResourceClaim and then schedules
// more pods using that ResourceClaim than supported at the same time. Scheduling so many pods in an E2E test
// was problematic, but in an integration test it's fine because we control the environment.
func testShareResourceClaimSequentially(tCtx ktesting.TContext) {
	// Some E2E helpers still use Gomega directly.
	// TODO: rewrite those helpers to use TContext.
	// In the meantime, make Gomega work by running sequentially.
	gomega.RegisterTestingT(tCtx)
	tCtx.Cleanup(func() {
		gomega.RegisterFailHandler(nil)
	})

	tCtx = tCtx.WithNamespace(createTestNamespace(tCtx, nil))
	startScheduler(tCtx)
	startClaimController(tCtx)

	nodes := drautils.NewNodesNow(tCtx, 1, 8)
	driver := drautils.NewDriverInstance(tCtx)
	driver.WithRealNodes = false
	driver.WithKubelet = false
	driver.Run(tCtx, "/no/kubelet/root", nodes, drautils.NetworkResources(1, false)(nodes))
	b := drautils.NewBuilderNow(tCtx, driver)

	var objects []klog.KMetadata
	objects = append(objects, b.ExternalClaim())

	// This test used to test usage of the claim by one pod
	// at a time. After removing the "not sharable"
	// feature and bumping up the maximum number of
	// consumers this is became a stress test.
	numMaxPods := resourceapi.ResourceClaimReservedForMaxSize
	tCtx.Logf("Creating %d pods sharing the same claim", numMaxPods)
	pods := make([]*v1.Pod, numMaxPods)
	for i := range numMaxPods {
		pod := b.PodExternal()
		pods[i] = pod
		objects = append(objects, pod)
	}
	b.Create(tCtx, objects...)

	podStartTimeout := 5 * time.Minute * time.Duration(numMaxPods)
	ensureDuration := time.Minute // Don't check for too long, even if it is less precise.
	podIsScheduled := gomega.HaveField("Spec.NodeName", gomega.Not(gomega.BeEmpty()))
	podIsPending := gomega.And(
		gomega.HaveField("Status.Phase", gomega.Equal(v1.PodPending)),
		gomega.HaveField("Status.Conditions", gomega.ContainElement(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Type":   gomega.Equal(v1.PodScheduled),
			"Status": gomega.Equal(v1.ConditionFalse),
			// With the current scheduler code, creating too many pods is treated as an error,
			// so this would have to be "SchedulerError". Maybe it shouldn't be an error?
			// "Reason": gomega.Equal(v1.PodReasonUnschedulable),
		}))),
	)
	assertPodScheduledEventually := func(tCtx ktesting.TContext, pod *v1.Pod) {
		tCtx.Helper()
		tCtx.AssertEventually(tCtx.Client().CoreV1().Pods(pod.Namespace).Get).
			WithArguments(pod.Name, metav1.GetOptions{}).
			WithTimeout(podStartTimeout).
			WithPolling(10*time.Second).
			Should(podIsScheduled, "Pod %s should get scheduled.", pod.Name)
	}

	assertPodPendingEventually := func(tCtx ktesting.TContext, pod *v1.Pod) {
		tCtx.Helper()
		tCtx.AssertEventually(tCtx.Client().CoreV1().Pods(pod.Namespace).Get).
			WithArguments(pod.Name, metav1.GetOptions{}).
			WithTimeout(ensureDuration).
			WithPolling(10*time.Second).
			Should(podIsPending, "Pod %s should remain pending.", pod.Name)
	}

	// We don't know the order. All that matters is that all of them get scheduled eventually.
	tCtx.Logf("Waiting for %d pods to be scheduled", numMaxPods)
	var wg sync.WaitGroup
	for i := range numMaxPods {
		wg.Go(func() {
			assertPodScheduledEventually(tCtx, pods[i])
		})
	}
	wg.Wait()
	if tCtx.Failed() {
		return
	}

	// TODO (?): check metrics about pod scheduling, speed up pod scheduling.
	// Currently pods go into backoff, so the initial 256 pods get scheduled much
	// more slowly that they could be.

	numMorePods := 10
	tCtx.Logf("Creating %d additional pods for the same claim", numMorePods)
	morePods := make([]*v1.Pod, numMorePods)
	objects = nil
	for i := range numMorePods {
		pod := b.PodExternal()
		morePods[i] = pod
		objects = append(objects, pod)
	}
	b.Create(tCtx, objects...)

	// None of the additional pods can run because of the ReservedFor limit.
	tCtx.Logf("Check for %s that the additional pods don't get scheduled", ensureDuration)
	for i := range numMorePods {
		wg.Go(func() {
			assertPodPendingEventually(tCtx, morePods[i])
		})
	}
	wg.Wait()
	if tCtx.Failed() {
		return
	}

	// We need to force-delete (no kubelet!) each scheduled pod,
	// otherwise the new ones cannot use the claim.
	tCtx.Logf("Deleting the initial %d pods", numMaxPods)
	for i := range numMaxPods {
		wg.Go(func() {
			if !tCtx.AssertNoError(tCtx.Client().CoreV1().Pods(pods[i].Namespace).Delete(tCtx, pods[i].Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))})) {
				return
			}
			// Should be almost (completely?) instantaneous.
			tCtx.AssertNoError(e2epod.WaitForPodNotFoundInNamespace(tCtx, tCtx.Client(), pods[i].Name, pods[i].Namespace, time.Second))
		})
	}
	wg.Wait()
	if tCtx.Failed() {
		return
	}

	// Now those should also get scheduled - eventually...
	tCtx.Logf("Waiting for the additional %d pods to be scheduled", numMorePods)
	for i := range numMorePods {
		wg.Go(func() {
			assertPodScheduledEventually(tCtx, morePods[i])
		})
	}
	wg.Wait()
}
