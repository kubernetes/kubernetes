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

package node

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	dynamicContainersTimeout = 2 * time.Minute
	// The kubelet keeps the statuses (and runtime containers) of at most this many removed containers.
	maxRemovedContainerStatuses = 10
	// Matches kubecontainer.ContainerReasonUnallocated.
	unallocatedReason = "Unallocated"
)

var _ = SIGDescribe("DynamicContainers", framework.WithFeatureGate(features.DynamicContainers), func() {
	f := framework.NewDefaultFramework("dynamic-containers")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("dynamic containers are not supported on windows")
		}
	})

	ginkgo.It("should start added containers and stop removed containers", func(ctx context.Context) {
		pod := e2epod.NewPodClient(f).CreateSync(ctx, makeDynamicContainersPod("dynamic-lifecycle", makeSleepContainer("main", nil)))

		ginkgo.By("adding a container")
		updateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			p.Spec.Containers = append(p.Spec.Containers, makeSleepContainer("dyn-0", nil))
		})
		waitForContainerStatuses(ctx, f, pod.Name, "dyn-0 to be running", func(_ *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			return isRunning(statuses["dyn-0"])
		})

		ginkgo.By("reordering the containers")
		p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		runningIDs := map[string]string{}
		for _, s := range p.Status.ContainerStatuses {
			runningIDs[s.Name] = s.ContainerID
		}
		updateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			slices.Reverse(p.Spec.Containers)
		})
		gomega.Consistently(ctx, func(ctx context.Context) error {
			p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			for name, s := range containerStatusesByName(p) {
				if !isRunning(s) || s.ContainerID != runningIDs[name] {
					return fmt.Errorf("container %q should keep running unchanged after a reorder: %+v", name, s)
				}
			}
			return nil
		}).WithTimeout(10 * time.Second).WithPolling(2 * time.Second).Should(gomega.Succeed())

		ginkgo.By("removing the container")
		updateDynamicContainers(ctx, f, pod.Name, removeContainers("dyn-0"))
		waitForContainerStatuses(ctx, f, pod.Name, "dyn-0 to be terminated", func(_ *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			return isTerminated(statuses["dyn-0"]) && isRunning(statuses["main"])
		})

		ginkgo.By("reading the logs of the removed container")
		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "dyn-0")
		framework.ExpectNoError(err, "failed to get the logs of removed container dyn-0")
		gomega.Expect(logs).To(gomega.ContainSubstring(sleepContainerStartedLog))

		ginkgo.By(fmt.Sprintf("adding and removing more than %d containers", maxRemovedContainerStatuses))
		var names []string
		for i := 1; i <= maxRemovedContainerStatuses+1; i++ {
			names = append(names, fmt.Sprintf("dyn-%d", i))
		}
		updateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			for _, name := range names {
				p.Spec.Containers = append(p.Spec.Containers, makeSleepContainer(name, nil))
			}
		})
		waitForContainerStatuses(ctx, f, pod.Name, "added containers to be running", func(_ *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			for _, name := range names {
				if !isRunning(statuses[name]) {
					return false
				}
			}
			return true
		})
		updateDynamicContainers(ctx, f, pod.Name, removeContainers(names...))

		ginkgo.By(fmt.Sprintf("verifying that only the %d most recently removed container statuses are kept", maxRemovedContainerStatuses))
		waitForContainerStatuses(ctx, f, pod.Name, "removed container statuses to be capped", func(p *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			removed := removedContainerStatuses(p)
			if len(removed) != maxRemovedContainerStatuses {
				return false
			}
			for _, s := range removed {
				if !isTerminated(s) {
					return false
				}
			}
			// dyn-0 was removed first, so it's the first to be dropped.
			_, found := statuses["dyn-0"]
			return !found && isRunning(statuses["main"])
		})
		gomega.Consistently(ctx, func(ctx context.Context) (int, error) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return 0, err
			}
			return len(removedContainerStatuses(p)), nil
		}).WithTimeout(10*time.Second).WithPolling(2*time.Second).Should(gomega.Equal(maxRemovedContainerStatuses), "removed container statuses should stay capped")

		ginkgo.By("re-adding a container whose status was dropped")
		updateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			p.Spec.Containers = append(p.Spec.Containers, makeSleepContainer("dyn-0", nil))
		})
		waitForContainerStatuses(ctx, f, pod.Name, "re-added dyn-0 to be running", func(_ *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			return isRunning(statuses["dyn-0"]) && statuses["dyn-0"].RestartCount == 0
		})
	})

	// Serial because the filler pod takes up all the free CPU of the node.
	ginkgo.It("should apply container changes atomically with a pending resize", framework.WithSerial(), func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)
		pod := podClient.CreateSync(ctx, makeDynamicContainersPod("dynamic-deferred",
			makeSleepContainer("resized", cpuRequest("100m")),
			makeSleepContainer("removed", cpuRequest("100m")),
		))
		originalIDs := map[string]string{}
		for _, s := range pod.Status.ContainerStatuses {
			originalIDs[s.Name] = s.ContainerID
		}
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get node %q of pod %q", pod.Spec.NodeName, pod.Name)
		allocatableCPU, availableCPU, err := e2enode.GetNodeAllocatableAndAvailableQuantities(ctx, f.ClientSet, node, v1.ResourceCPU)
		framework.ExpectNoError(err, "failed to get the CPU available on node %q", node.Name)
		framework.Logf("Node %q: allocatable CPU %dm, available CPU %dm", node.Name, allocatableCPU.MilliValue(), availableCPU.MilliValue())

		// The filler leaves fillerSlackCPU free, which the resize below exceeds, and
		// must leave enough room to fit the resize once it is deleted.
		fillerSlackCPU := resource.MustParse("50m")
		resizeIncreaseCPU := resource.MustParse("200m")
		minAvailableCPU := resource.MustParse("300m")
		if availableCPU.Cmp(minAvailableCPU) < 0 {
			e2eskipper.Skipf("node %q has %dm CPU available, the test needs at least %dm", node.Name, availableCPU.MilliValue(), minAvailableCPU.MilliValue())
		}

		ginkgo.By("verifying that admission rejects a /dynamic update beyond node allocatable")
		tooMuchCPU := allocatableCPU
		tooMuchCPU.Add(resource.MustParse("1"))
		err = tryUpdateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			p.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = tooMuchCPU
		})
		gomega.Expect(err).To(gomega.HaveOccurred(), "expected PodResizeValidator to reject a /dynamic update beyond node allocatable")
		var statusErr *apierrors.StatusError
		gomega.Expect(errors.As(err, &statusErr)).To(gomega.BeTrueBecause("expected a StatusError, got: %v", err))
		gomega.Expect(statusErr.ErrStatus.Reason).To(gomega.Equal(metav1.StatusReasonForbidden))
		gomega.Expect(statusErr.Error()).To(gomega.ContainSubstring("node didn't have enough allocatable resources"))
		gomega.Expect(statusErr.ErrStatus.Details).NotTo(gomega.BeNil())
		gomega.Expect(statusErr.ErrStatus.Details.Causes).To(gomega.ContainElement(
			gomega.HaveField("Type", metav1.CauseType("NodeCapacity"))))
		p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(p.Generation).To(gomega.Equal(pod.Generation), "a rejected update must not change the pod")

		ginkgo.By("filling the node so that a resize of the pod is deferred")
		fillerCPU := availableCPU
		fillerCPU.Sub(fillerSlackCPU)
		filler := makeDynamicContainersPod("dynamic-deferred-filler", makeSleepContainer("filler", cpuRequest(fillerCPU.String())))
		e2epod.SetNodeAffinity(&filler.Spec, node.Name)
		filler = podClient.CreateSync(ctx, filler)
		gomega.Expect(filler.Spec.NodeName).To(gomega.Equal(node.Name), "the filler pod must run on the node of the resized pod")

		ginkgo.By("adding and removing containers, updating an image, and resizing beyond the free capacity in one update")
		resizedCPU := resource.MustParse("100m")
		resizedCPU.Add(resizeIncreaseCPU)
		updateDynamicContainers(ctx, f, pod.Name, func(p *v1.Pod) {
			removeContainers("removed")(p)
			p.Spec.Containers[0].Image = updatedImage
			p.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resizedCPU
			p.Spec.Containers = append(p.Spec.Containers, makeSleepContainer("added", cpuRequest("100m")))
		})
		waitForContainerStatuses(ctx, f, pod.Name, "the update to be deferred", func(p *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			return isResizeDeferred(p) && isUnallocated(statuses["added"])
		})

		expectNothingActuated := func(ctx context.Context) error {
			p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			statuses := containerStatusesByName(p)
			if !isUnallocated(statuses["added"]) {
				return fmt.Errorf("added container should be unallocated: %+v", statuses["added"])
			}
			for _, name := range []string{"resized", "removed"} {
				if !isRunning(statuses[name]) || statuses[name].ContainerID != originalIDs[name] {
					return fmt.Errorf("container %q should still be running unchanged: %+v", name, statuses[name])
				}
			}
			if !isResizeDeferred(p) {
				return fmt.Errorf("expected a deferred resize: %+v", p.Status.Conditions)
			}
			// Unallocated containers don't count against pod readiness.
			if !podutil.IsPodReady(p) {
				return fmt.Errorf("pod should stay ready while a container is unallocated: %+v", p.Status.Conditions)
			}
			return nil
		}
		ginkgo.By("verifying that none of the changes are actuated")
		gomega.Consistently(ctx, expectNothingActuated).WithTimeout(15 * time.Second).WithPolling(3 * time.Second).Should(gomega.Succeed())

		// Ephemeral containers aren't gated on allocation, so they start even while
		// the rest of the pod's changes wait.
		ginkgo.By("adding an ephemeral container while the update is deferred")
		ec := &v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    "debugger",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"sh", "-c", "trap exit TERM; while true; do sleep 1; done"},
			},
		}
		err = podClient.AddEphemeralContainerSync(ctx, pod, ec, dynamicContainersTimeout)
		framework.ExpectNoError(err, "failed to add an ephemeral container to pod %q", e2epod.FormatPod(pod))
		framework.ExpectNoError(expectNothingActuated(ctx), "the deferred update must stay pending after an ephemeral container is added")

		ginkgo.By("deleting the filler pod")
		framework.ExpectNoError(e2epod.DeletePodWithWait(ctx, f.ClientSet, filler))

		ginkgo.By("verifying that all the pending changes are actuated together")
		waitForContainerStatuses(ctx, f, pod.Name, "the changes to be actuated", func(p *v1.Pod, statuses map[string]v1.ContainerStatus) bool {
			resized := statuses["resized"]
			return !isResizePending(p) &&
				isRunning(statuses["added"]) &&
				isTerminated(statuses["removed"]) &&
				isRunning(resized) && resized.ContainerID != originalIDs["resized"] &&
				resized.Resources != nil && resized.Resources.Requests.Cpu().Cmp(resizedCPU) == 0
		})
		p, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(p.Status.EphemeralContainerStatuses).To(gomega.ContainElement(gomega.And(
			gomega.HaveField("Name", ec.Name),
			gomega.HaveField("State.Running", gomega.Not(gomega.BeNil())),
		)), "the ephemeral container should keep running")
	})
})

// An image other than the one of makeSleepContainer that can run the same command,
// so the kubelet sees an image update.
var updatedImage = imageutils.GetE2EImage(imageutils.Agnhost)

func makeDynamicContainersPod(name string, containers ...v1.Container) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyAlways,
			TerminationGracePeriodSeconds: ptr.To[int64](1),
			Containers:                    containers,
		},
	}
}

// sleepContainerStartedLog is logged by makeSleepContainer containers when they start.
const sleepContainerStartedLog = "started"

func makeSleepContainer(name string, resources *v1.ResourceRequirements) v1.Container {
	c := v1.Container{
		Name:    name,
		Image:   imageutils.GetE2EImage(imageutils.BusyBox),
		Command: []string{"sh", "-c", "echo " + sleepContainerStartedLog + "; trap exit TERM; while true; do sleep 1; done"},
	}
	if resources != nil {
		c.Resources = *resources
	}
	return c
}

func cpuRequest(q string) *v1.ResourceRequirements {
	return &v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(q)}}
}

func removeContainers(names ...string) func(*v1.Pod) {
	return func(p *v1.Pod) {
		p.Spec.Containers = slices.DeleteFunc(p.Spec.Containers, func(c v1.Container) bool {
			return slices.Contains(names, c.Name)
		})
	}
}

func tryUpdateDynamicContainers(ctx context.Context, f *framework.Framework, podName string, mutate func(*v1.Pod)) error {
	pods := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		pod, err := pods.Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		mutate(pod)
		_, err = pods.UpdateDynamic(ctx, podName, pod, metav1.UpdateOptions{})
		return err
	})
}

func updateDynamicContainers(ctx context.Context, f *framework.Framework, podName string, mutate func(*v1.Pod)) {
	err := tryUpdateDynamicContainers(ctx, f, podName, mutate)
	framework.ExpectNoError(err, "failed to update the containers of pod %q through pods/dynamic", podName)
}

func waitForContainerStatuses(ctx context.Context, f *framework.Framework, podName, desc string, condition func(*v1.Pod, map[string]v1.ContainerStatus) bool) {
	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, podName, desc, dynamicContainersTimeout, func(pod *v1.Pod) (bool, error) {
		if pod.Status.Phase != v1.PodRunning {
			return false, fmt.Errorf("pod is %s, expected it to stay %s", pod.Status.Phase, v1.PodRunning)
		}
		return condition(pod, containerStatusesByName(pod)), nil
	})
	framework.ExpectNoError(err)
}

func containerStatusesByName(pod *v1.Pod) map[string]v1.ContainerStatus {
	statuses := make(map[string]v1.ContainerStatus, len(pod.Status.ContainerStatuses))
	for _, s := range pod.Status.ContainerStatuses {
		statuses[s.Name] = s
	}
	return statuses
}

func removedContainerStatuses(pod *v1.Pod) []v1.ContainerStatus {
	inSpec := sets.New[string]()
	for _, c := range pod.Spec.Containers {
		inSpec.Insert(c.Name)
	}
	var removed []v1.ContainerStatus
	for _, s := range pod.Status.ContainerStatuses {
		if !inSpec.Has(s.Name) {
			removed = append(removed, s)
		}
	}
	return removed
}

func isRunning(s v1.ContainerStatus) bool {
	return s.State.Running != nil
}

func isTerminated(s v1.ContainerStatus) bool {
	return s.State.Terminated != nil
}

func isUnallocated(s v1.ContainerStatus) bool {
	return s.State.Waiting != nil && s.State.Waiting.Reason == unallocatedReason
}

func resizePendingCondition(pod *v1.Pod) *v1.PodCondition {
	for i, c := range pod.Status.Conditions {
		if c.Type == v1.PodResizePending && c.Status == v1.ConditionTrue {
			return &pod.Status.Conditions[i]
		}
	}
	return nil
}

func isResizePending(pod *v1.Pod) bool {
	return resizePendingCondition(pod) != nil
}

// isResizeDeferred distinguishes a resize that waits for capacity from an infeasible one.
func isResizeDeferred(pod *v1.Pod) bool {
	c := resizePendingCondition(pod)
	return c != nil && c.Reason == v1.PodReasonDeferred
}
