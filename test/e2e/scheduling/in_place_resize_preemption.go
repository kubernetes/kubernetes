/*
Copyright 2026 The Kubernetes Authors.

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

package scheduling

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	helpers "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

func getNodeFreeCPU(ctx context.Context, cs clientset.Interface, targetNodeObj *v1.Node) int64 {
	allocatableCPU := targetNodeObj.Status.Allocatable[v1.ResourceCPU]
	totalAllocatableMilli := allocatableCPU.MilliValue()

	podList, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{FieldSelector: "spec.nodeName=" + targetNodeObj.Name})
	framework.ExpectNoError(err, "failed to list pods assigned to node %s", targetNodeObj.Name)

	usedMilliCPU := int64(0)
	for _, p := range podList.Items {
		if p.Status.Phase != v1.PodSucceeded && p.Status.Phase != v1.PodFailed {
			for _, c := range p.Spec.Containers {
				if req, ok := c.Resources.Requests[v1.ResourceCPU]; ok {
					usedMilliCPU += req.MilliValue()
				}
			}
		}
	}

	freeMilliCPU := totalAllocatableMilli - usedMilliCPU
	framework.Logf("Node %s has total allocatable CPU: %dm, used: %dm, free: %dm", targetNodeObj.Name, totalAllocatableMilli, usedMilliCPU, freeMilliCPU)
	if freeMilliCPU < 200 {
		freeMilliCPU = 200
	}
	return freeMilliCPU
}

func getNodeFreeMemory(ctx context.Context, cs clientset.Interface, targetNodeObj *v1.Node) int64 {
	allocatableMem := targetNodeObj.Status.Allocatable[v1.ResourceMemory]
	totalAllocatableBytes := allocatableMem.Value()

	podList, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{FieldSelector: "spec.nodeName=" + targetNodeObj.Name})
	framework.ExpectNoError(err, "failed to list pods assigned to node %s", targetNodeObj.Name)

	usedBytesMem := int64(0)
	for _, p := range podList.Items {
		if p.Status.Phase != v1.PodSucceeded && p.Status.Phase != v1.PodFailed {
			for _, c := range p.Spec.Containers {
				if req, ok := c.Resources.Requests[v1.ResourceMemory]; ok {
					usedBytesMem += req.Value()
				}
			}
		}
	}

	freeBytesMem := totalAllocatableBytes - usedBytesMem
	framework.Logf("Node %s has total allocatable Memory: %d bytes, used: %d bytes, free: %d bytes", targetNodeObj.Name, totalAllocatableBytes, usedBytesMem, freeBytesMem)
	if freeBytesMem < 200*1024*1024 {
		freeBytesMem = 200 * 1024 * 1024
	}
	return freeBytesMem
}

func makeNodeAffinity(nodeName string) *v1.Affinity {
	return &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchFields: []v1.NodeSelectorRequirement{
							{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{nodeName}},
						},
					},
				},
			},
		},
	}
}

func waitForPodDeferred(ctx context.Context, f *framework.Framework, testPod *v1.Pod) {
	framework.ExpectNoError(e2epod.WaitForPodCondition(ctx, f.ClientSet, testPod.Namespace, testPod.Name, "display pod resize status as deferred", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
		return helpers.IsPodResizeDeferred(pod), nil
	}))
}

var _ = SIGDescribe("InPlaceResizePreemption", framework.WithSerial(), framework.WithFeatureGate(features.InPlacePodVerticalScaling), framework.WithFeatureGate(features.InPlacePodVerticalScalingSchedulerPreemption), func() {
	var cs clientset.Interface
	var ns string
	f := framework.NewDefaultFramework("in-place-resize-preemption")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	lowPriority, highPriority := int32(100), int32(1000)
	lowPriorityClassName := f.BaseName + "-low-priority"
	highPriorityClassName := f.BaseName + "-high-priority"
	priorityPairs := []priorityPair{
		{name: lowPriorityClassName, value: lowPriority},
		{name: highPriorityClassName, value: highPriority},
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = f.ClientSet
		ns = f.Namespace.Name

		for _, pair := range priorityPairs {
			_, err := cs.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: pair.name},
				Value:      pair.value,
			}, metav1.CreateOptions{})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
			}
		}
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		for _, pair := range priorityPairs {
			_ = cs.SchedulingV1().PriorityClasses().Delete(ctx, pair.name, *metav1.NewDeleteOptions(0))
		}
	})

	ginkgo.It("validates in-place pod resize triggers scheduler preemption of lower-priority pods and completes resize actuation on node", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)

		ginkgo.By("Selecting a ready schedulable node")
		targetNodeObj, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		targetNode := targetNodeObj.Name

		freeMilliCPU := getNodeFreeCPU(ctx, cs, targetNodeObj)

		// Resource budget:
		// victim = 50% of free CPU
		// initialHigh = 30% of free CPU (sum = 80% <= 100% -> fits initially)
		// resizedHigh = 70% of free CPU (sum = 120% > 100% -> triggers preemption, victim is evicted, 70% fits)
		victimCPU := freeMilliCPU * 50 / 100
		if victimCPU < 50 {
			victimCPU = 50
		}
		initialHighCPU := freeMilliCPU * 30 / 100
		if initialHighCPU < 30 {
			initialHighCPU = 30
		}
		resizedHighCPU := freeMilliCPU * 70 / 100
		if resizedHighCPU < 70 {
			resizedHighCPU = 70
		}

		victimCPUStr := fmt.Sprintf("%dm", victimCPU)
		initialHighCPUStr := fmt.Sprintf("%dm", initialHighCPU)
		resizedHighCPUStr := fmt.Sprintf("%dm", resizedHighCPU)

		ginkgo.By(fmt.Sprintf("Creating low-priority victim pod on node %s with CPU request %s", targetNode, victimCPUStr))
		zeroGracePeriod := int64(0)
		victimPodConfig := pausePodConfig{
			Name:                          "victim-pod",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(victimCPUStr),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(victimCPUStr),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		}
		victimPod := createPausePod(ctx, f, victimPodConfig)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod), "victim pod failed to run")

		ginkgo.By(fmt.Sprintf("Creating high-priority pod on node %s with initial CPU request %s", targetNode, initialHighCPUStr))
		originalContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: initialHighCPUStr,
					CPULim: initialHighCPUStr,
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		tStamp := fmt.Sprintf("%d", time.Now().UnixNano())
		highPodSpec := podresize.MakePodWithResizableContainers(ns, "preemptor-resize-pod", tStamp, originalContainers, nil)
		highPodSpec.Spec.PriorityClassName = highPriorityClassName
		highPodSpec.Spec.Affinity = makeNodeAffinity(targetNode)

		highPod := podClient.Create(ctx, highPodSpec)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, highPod), "high priority pod failed to run")

		ginkgo.By("Verifying high priority pod initial state before resize")
		highPod, err = cs.CoreV1().Pods(ns).Get(ctx, highPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get high priority pod")
		gomega.Expect(highPod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		gomega.Expect(highPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))

		ginkgo.By(fmt.Sprintf("Patching high-priority pod to expand CPU from %s to %s, exceeding node capacity", initialHighCPUStr, resizedHighCPUStr))
		expectedContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: resizedHighCPUStr,
					CPULim: resizedHighCPUStr,
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patch := podresize.MakeResizePatch(originalContainers, expectedContainers, nil, nil)
		_, patchErr := cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch pod for resize")

		ginkgo.By("Verifying that scheduler preempts the low-priority victim pod to free node capacity")
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, victimPod.Name, metav1.GetOptions{})
			if err != nil {
				return apierrors.IsNotFound(err)
			}
			return p.DeletionTimestamp != nil
		}).WithTimeout(45 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "Victim pod should be preempted by scheduler")

		ginkgo.By("Waiting for resize actuation to complete on node without container restart")
		expected := podresize.UpdateExpectedContainerRestarts(ctx, highPod, expectedContainers)
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, highPod, expected)

		ginkgo.By("Verifying pod container restart count is 0 and allocated resources match resized target")
		gomega.Expect(resizedPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)), "container should not restart during in-place resize actuation")
		gomega.Expect(resizedPod.Status.ContainerStatuses[0].AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(resizedHighCPUStr)))
	})

	ginkgo.It("validates memory request expansion triggers scheduler preemption of lower-priority pods and completes memory cgroup actuation without pod restart", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)

		ginkgo.By("Selecting a ready schedulable node")
		targetNodeObj, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		targetNode := targetNodeObj.Name

		freeBytesMem := getNodeFreeMemory(ctx, cs, targetNodeObj)
		freeMemMiB := freeBytesMem / (1024 * 1024)

		victimMemMiB := freeMemMiB * 50 / 100
		if victimMemMiB < 50 {
			victimMemMiB = 50
		}
		initialHighMemMiB := freeMemMiB * 30 / 100
		if initialHighMemMiB < 30 {
			initialHighMemMiB = 30
		}
		resizedHighMemMiB := freeMemMiB * 70 / 100
		if resizedHighMemMiB < 70 {
			resizedHighMemMiB = 70
		}

		victimMemStr := fmt.Sprintf("%dMi", victimMemMiB)
		initialHighMemStr := fmt.Sprintf("%dMi", initialHighMemMiB)
		resizedHighMemStr := fmt.Sprintf("%dMi", resizedHighMemMiB)

		ginkgo.By(fmt.Sprintf("Creating low-priority victim pod on node %s with memory request %s", targetNode, victimMemStr))
		zeroGracePeriod := int64(0)
		victimPodConfig := pausePodConfig{
			Name:                          "victim-pod-mem",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("50m"),
					v1.ResourceMemory: resource.MustParse(victimMemStr),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("50m"),
					v1.ResourceMemory: resource.MustParse(victimMemStr),
				},
			},
		}
		victimPod := createPausePod(ctx, f, victimPodConfig)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod), "victim pod failed to run")

		ginkgo.By(fmt.Sprintf("Creating high-priority pod on node %s with initial memory request %s", targetNode, initialHighMemStr))
		originalContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: "50m",
					CPULim: "50m",
					MemReq: initialHighMemStr,
					MemLim: initialHighMemStr,
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		tStamp := fmt.Sprintf("%d", time.Now().UnixNano())
		highPodSpec := podresize.MakePodWithResizableContainers(ns, "preemptor-resize-mem-pod", tStamp, originalContainers, nil)
		highPodSpec.Spec.PriorityClassName = highPriorityClassName
		highPodSpec.Spec.Affinity = makeNodeAffinity(targetNode)

		highPod := podClient.Create(ctx, highPodSpec)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, highPod), "high priority pod failed to run")

		ginkgo.By("Verifying high priority pod initial state before resize")
		highPod, err = cs.CoreV1().Pods(ns).Get(ctx, highPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get high priority pod")
		gomega.Expect(highPod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		gomega.Expect(highPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))

		ginkgo.By(fmt.Sprintf("Patching high-priority pod to expand memory from %s to %s, exceeding node memory capacity", initialHighMemStr, resizedHighMemStr))
		expectedContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: "50m",
					CPULim: "50m",
					MemReq: resizedHighMemStr,
					MemLim: resizedHighMemStr,
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patch := podresize.MakeResizePatch(originalContainers, expectedContainers, nil, nil)
		_, patchErr := cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch pod for memory resize")

		ginkgo.By("Verifying that scheduler preempts the low-priority victim pod to free node memory capacity")
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, victimPod.Name, metav1.GetOptions{})
			if err != nil {
				return apierrors.IsNotFound(err)
			}
			return p.DeletionTimestamp != nil
		}).WithTimeout(45 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "Victim pod should be preempted by scheduler for memory resize")

		ginkgo.By("Waiting for memory resize actuation to complete on node without container restart")
		expected := podresize.UpdateExpectedContainerRestarts(ctx, highPod, expectedContainers)
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, highPod, expected)

		ginkgo.By("Verifying pod container restart count is 0 and allocated memory matches resized target")
		gomega.Expect(resizedPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)), "container should not restart during in-place memory resize actuation")
		gomega.Expect(resizedPod.Status.ContainerStatuses[0].AllocatedResources[v1.ResourceMemory]).To(gomega.Equal(resource.MustParse(resizedHighMemStr)))
	})

	ginkgo.It("validates multi-container pod in-place resize preemption correctly accounts for multi-container delta and updates container cgroup", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)

		ginkgo.By("Selecting a ready schedulable node")
		targetNodeObj, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		targetNode := targetNodeObj.Name

		freeMilliCPU := getNodeFreeCPU(ctx, cs, targetNodeObj)

		// Resource budget:
		// victimPod1 = 30% of free CPU
		// victimPod2 = 30% of free CPU
		// highPod container c1 = 20% of free CPU
		// highPod container c2 = 20% of free CPU
		// Total initial usage = 30% + 30% + 20% + 20% = 100%
		// Resize only c1 to 45% (delta = +25%).
		// Total after evicting only 1 victim = 45% + 20% + 30% = 95% <= 100%
		// Scheduler must only evict one victim pod, leaving the second intact.
		victim1CPU := freeMilliCPU * 30 / 100
		if victim1CPU < 30 {
			victim1CPU = 30
		}
		victim2CPU := freeMilliCPU * 30 / 100
		if victim2CPU < 30 {
			victim2CPU = 30
		}
		c1InitialCPU := freeMilliCPU * 20 / 100
		if c1InitialCPU < 20 {
			c1InitialCPU = 20
		}
		c2InitialCPU := freeMilliCPU * 20 / 100
		if c2InitialCPU < 20 {
			c2InitialCPU = 20
		}
		c1ResizedCPU := freeMilliCPU * 45 / 100
		if c1ResizedCPU < 45 {
			c1ResizedCPU = 45
		}

		victim1CPUStr := fmt.Sprintf("%dm", victim1CPU)
		victim2CPUStr := fmt.Sprintf("%dm", victim2CPU)
		c1InitialCPUStr := fmt.Sprintf("%dm", c1InitialCPU)
		c2InitialCPUStr := fmt.Sprintf("%dm", c2InitialCPU)
		c1ResizedCPUStr := fmt.Sprintf("%dm", c1ResizedCPU)

		zeroGracePeriod := int64(0)
		ginkgo.By("Creating 2 low-priority victim pods")
		victim1PodConfig := pausePodConfig{
			Name:                          "multi-ctr-victim-1",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(victim1CPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(victim1CPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
			},
		}
		victimPod1 := createPausePod(ctx, f, victim1PodConfig)

		victim2PodConfig := pausePodConfig{
			Name:                          "multi-ctr-victim-2",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(victim2CPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(victim2CPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
			},
		}
		victimPod2 := createPausePod(ctx, f, victim2PodConfig)

		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod1))
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod2))

		ginkgo.By("Creating multi-container high-priority pod with containers c1 and c2")
		originalContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: c1InitialCPUStr,
					CPULim: c1InitialCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
			{
				Name: "c2",
				Resources: &cgroups.ContainerResources{
					CPUReq: c2InitialCPUStr,
					CPULim: c2InitialCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		tStamp := fmt.Sprintf("%d", time.Now().UnixNano())
		highPodSpec := podresize.MakePodWithResizableContainers(ns, "multi-ctr-preemptor-pod", tStamp, originalContainers, nil)
		highPodSpec.Spec.PriorityClassName = highPriorityClassName
		highPodSpec.Spec.Affinity = makeNodeAffinity(targetNode)

		highPod := podClient.Create(ctx, highPodSpec)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, highPod), "multi-container high priority pod failed to run")

		highPod, err = cs.CoreV1().Pods(ns).Get(ctx, highPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(highPod.Status.ContainerStatuses).To(gomega.HaveLen(2))
		gomega.Expect(highPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
		gomega.Expect(highPod.Status.ContainerStatuses[1].RestartCount).To(gomega.Equal(int32(0)))

		ginkgo.By(fmt.Sprintf("Patching multi-container pod to expand only container c1 from %s to %s", c1InitialCPUStr, c1ResizedCPUStr))
		expectedContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: c1ResizedCPUStr,
					CPULim: c1ResizedCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
			{
				Name: "c2",
				Resources: &cgroups.ContainerResources{
					CPUReq: c2InitialCPUStr,
					CPULim: c2InitialCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patch := podresize.MakeResizePatch(originalContainers, expectedContainers, nil, nil)
		_, patchErr := cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch multi-container pod for resize")

		ginkgo.By("Verifying that scheduler preempts exactly one victim pod while preserving the other")
		var survivingPodName string
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			p1, err1 := cs.CoreV1().Pods(ns).Get(ctx, victimPod1.Name, metav1.GetOptions{})
			p1Preempted := (err1 != nil && apierrors.IsNotFound(err1)) || (err1 == nil && p1.DeletionTimestamp != nil)

			p2, err2 := cs.CoreV1().Pods(ns).Get(ctx, victimPod2.Name, metav1.GetOptions{})
			p2Preempted := (err2 != nil && apierrors.IsNotFound(err2)) || (err2 == nil && p2.DeletionTimestamp != nil)

			if p1Preempted && !p2Preempted {
				survivingPodName = victimPod2.Name
				return true
			} else if p2Preempted && !p1Preempted {
				survivingPodName = victimPod1.Name
				return true
			}
			return false
		}).WithTimeout(45 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "Expected exactly one victim pod to be preempted")

		gomega.Consistently(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, survivingPodName, metav1.GetOptions{})
			return err == nil && p.DeletionTimestamp == nil && p.Status.Phase == v1.PodRunning
		}).WithTimeout(5 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "Surviving victim pod should not be preempted")

		ginkgo.By("Waiting for multi-container resize actuation to complete on node")
		expected := podresize.UpdateExpectedContainerRestarts(ctx, highPod, expectedContainers)
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, highPod, expected)

		ginkgo.By("Verifying container restart counts and allocated resources")
		for _, cs := range resizedPod.Status.ContainerStatuses {
			gomega.Expect(cs.RestartCount).To(gomega.Equal(int32(0)), fmt.Sprintf("container %s should not restart during resize", cs.Name))
			if cs.Name == "c1" {
				gomega.Expect(cs.AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(c1ResizedCPUStr)))
			} else if cs.Name == "c2" {
				gomega.Expect(cs.AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(c2InitialCPUStr)))
			}
		}
	})

	ginkgo.It("validates node preemption policy DisableResizePreemption prevents scheduler preemption for deferred resize until capacity is freed voluntarily", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)

		ginkgo.By("Selecting a ready schedulable node")
		targetNodeObj, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		targetNode := targetNodeObj.Name

		ginkgo.By(fmt.Sprintf("Patching node %s to disable resize preemption by scheduler", targetNode))
		patchDisable := []byte(`{"spec": {"podPreemptionPolicy": {"disableResizePreemption": ["scheduler"]}}}`)
		_, err = cs.CoreV1().Nodes().Patch(ctx, targetNode, types.StrategicMergePatchType, patchDisable, metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch node to disable resize preemption")

		defer func() {
			patchReset := []byte(`{"spec": {"podPreemptionPolicy": null}}`)
			_, _ = cs.CoreV1().Nodes().Patch(ctx, targetNode, types.StrategicMergePatchType, patchReset, metav1.PatchOptions{})
		}()

		freeMilliCPU := getNodeFreeCPU(ctx, cs, targetNodeObj)

		victimCPU := freeMilliCPU * 50 / 100
		if victimCPU < 50 {
			victimCPU = 50
		}
		initialHighCPU := freeMilliCPU * 30 / 100
		if initialHighCPU < 30 {
			initialHighCPU = 30
		}
		resizedHighCPU := freeMilliCPU * 70 / 100
		if resizedHighCPU < 70 {
			resizedHighCPU = 70
		}

		victimCPUStr := fmt.Sprintf("%dm", victimCPU)
		initialHighCPUStr := fmt.Sprintf("%dm", initialHighCPU)
		resizedHighCPUStr := fmt.Sprintf("%dm", resizedHighCPU)

		ginkgo.By(fmt.Sprintf("Creating low-priority victim pod on node %s with CPU request %s", targetNode, victimCPUStr))
		zeroGracePeriod := int64(0)
		victimPodConfig := pausePodConfig{
			Name:                          "policy-victim-pod",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimCPUStr), v1.ResourceMemory: resource.MustParse("100Mi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimCPUStr), v1.ResourceMemory: resource.MustParse("100Mi")},
			},
		}
		victimPod := createPausePod(ctx, f, victimPodConfig)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod), "victim pod failed to run")

		ginkgo.By(fmt.Sprintf("Creating high-priority pod on node %s with initial CPU request %s", targetNode, initialHighCPUStr))
		originalContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: initialHighCPUStr,
					CPULim: initialHighCPUStr,
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		tStamp := fmt.Sprintf("%d", time.Now().UnixNano())
		highPodSpec := podresize.MakePodWithResizableContainers(ns, "policy-preemptor-pod", tStamp, originalContainers, nil)
		highPodSpec.Spec.PriorityClassName = highPriorityClassName
		highPodSpec.Spec.Affinity = makeNodeAffinity(targetNode)

		highPod := podClient.Create(ctx, highPodSpec)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, highPod), "high priority pod failed to run")

		ginkgo.By(fmt.Sprintf("Patching high-priority pod to expand CPU from %s to %s (exceeds node capacity)", initialHighCPUStr, resizedHighCPUStr))
		expectedContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: resizedHighCPUStr,
					CPULim: resizedHighCPUStr,
					MemReq: "100Mi",
					MemLim: "100Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patch := podresize.MakeResizePatch(originalContainers, expectedContainers, nil, nil)
		_, patchErr := cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch pod for resize")

		ginkgo.By("Verifying high-priority pod remains in Deferred resize state because node disables resize preemption")
		waitForPodDeferred(ctx, f, highPod)

		ginkgo.By("Verifying victim pod is NOT preempted while resize preemption is disabled on node")
		gomega.Consistently(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, victimPod.Name, metav1.GetOptions{})
			return err == nil && p.DeletionTimestamp == nil && p.Status.Phase == v1.PodRunning
		}).WithTimeout(10 * time.Second).WithPolling(1 * time.Second).Should(gomega.BeTrue(), "Victim pod must remain running while resize preemption is disabled")

		ginkgo.By("Voluntarily deleting victim pod to free node capacity")
		err = cs.CoreV1().Pods(ns).Delete(ctx, victimPod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, cs, victimPod.Name, ns, 30*time.Second))

		ginkgo.By("Verifying high-priority pod resize is actuated after capacity is freed voluntarily")
		expected := podresize.UpdateExpectedContainerRestarts(ctx, highPod, expectedContainers)
		resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, highPod, expected)

		gomega.Expect(resizedPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)), "container should not restart during in-place resize actuation")
		gomega.Expect(resizedPod.Status.ContainerStatuses[0].AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(resizedHighCPUStr)))
	})

	ginkgo.It("validates PodDisruptionBudget compliance and prioritization during in-place resize preemption", func(ctx context.Context) {
		podClient := e2epod.NewPodClient(f)

		ginkgo.By("Selecting a ready schedulable node")
		targetNodeObj, err := e2enode.GetRandomReadySchedulableNode(ctx, cs)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		targetNode := targetNodeObj.Name

		freeMilliCPU := getNodeFreeCPU(ctx, cs, targetNodeObj)

		// Resource budget:
		// pdbVictim: 25% of free CPU
		// nonPdbVictim: 25% of free CPU
		// initialHigh: 30% of free CPU (initial sum = 80% <= 100%)
		// step1High: 55% of free CPU (sum = 105% > 100% -> evicting 1 victim (25%) leaves 55+25=80% <= 100%)
		// step2High: 85% of free CPU (sum = 110% > 100% -> evicts remaining PDB victim)
		victimPdbCPU := freeMilliCPU * 25 / 100
		if victimPdbCPU < 40 {
			victimPdbCPU = 40
		}
		victimNonPdbCPU := freeMilliCPU * 25 / 100
		if victimNonPdbCPU < 40 {
			victimNonPdbCPU = 40
		}
		initialHighCPU := freeMilliCPU * 30 / 100
		if initialHighCPU < 40 {
			initialHighCPU = 40
		}
		step1HighCPU := freeMilliCPU * 55 / 100
		if step1HighCPU < 70 {
			step1HighCPU = 70
		}
		step2HighCPU := freeMilliCPU * 85 / 100
		if step2HighCPU < 100 {
			step2HighCPU = 100
		}

		victimPdbCPUStr := fmt.Sprintf("%dm", victimPdbCPU)
		victimNonPdbCPUStr := fmt.Sprintf("%dm", victimNonPdbCPU)
		initialHighCPUStr := fmt.Sprintf("%dm", initialHighCPU)
		step1HighCPUStr := fmt.Sprintf("%dm", step1HighCPU)
		step2HighCPUStr := fmt.Sprintf("%dm", step2HighCPU)

		zeroGracePeriod := int64(0)
		ginkgo.By("Creating low-priority PDB-protected victim pod")
		pdbVictimConfig := pausePodConfig{
			Name:                          "pdb-victim-pod",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Labels:                        map[string]string{"app": "pdb-protected-victim"},
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimPdbCPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimPdbCPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
			},
		}
		pdbVictimPod := createPausePod(ctx, f, pdbVictimConfig)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pdbVictimPod))

		ginkgo.By("Creating low-priority non-PDB victim pod")
		nonPdbVictimConfig := pausePodConfig{
			Name:                          "non-pdb-victim-pod",
			Namespace:                     ns,
			PriorityClassName:             lowPriorityClassName,
			Labels:                        map[string]string{"app": "non-pdb-victim"},
			Affinity:                      makeNodeAffinity(targetNode),
			TerminationGracePeriodSeconds: &zeroGracePeriod,
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimNonPdbCPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
				Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse(victimNonPdbCPUStr), v1.ResourceMemory: resource.MustParse("50Mi")},
			},
		}
		nonPdbVictimPod := createPausePod(ctx, f, nonPdbVictimConfig)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, nonPdbVictimPod))

		ginkgo.By("Creating PodDisruptionBudget protecting pdb-protected-victim pod")
		minAvail := intstr.FromInt32(1)
		pdb := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "victim-pdb",
				Namespace: ns,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				MinAvailable: &minAvail,
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": "pdb-protected-victim"},
				},
			},
		}
		_, err = cs.PolicyV1().PodDisruptionBudgets(ns).Create(ctx, pdb, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for PDB status to reflect 1 healthy pod and 0 disruptions allowed")
		err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
			curPDB, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(ctx, pdb.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return curPDB.Status.CurrentHealthy == 1 && curPDB.Status.DisruptionsAllowed == 0, nil
		})
		framework.ExpectNoError(err, "PDB failed to stabilize with 0 disruptions allowed")

		ginkgo.By("Creating high-priority pod on node")
		originalContainers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: initialHighCPUStr,
					CPULim: initialHighCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		tStamp := fmt.Sprintf("%d", time.Now().UnixNano())
		highPodSpec := podresize.MakePodWithResizableContainers(ns, "pdb-preemptor-pod", tStamp, originalContainers, nil)
		highPodSpec.Spec.PriorityClassName = highPriorityClassName
		highPodSpec.Spec.Affinity = makeNodeAffinity(targetNode)

		highPod := podClient.Create(ctx, highPodSpec)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, highPod))

		ginkgo.By(fmt.Sprintf("Patching high-priority pod to expand CPU from %s to %s (step 1: evicts non-PDB victim)", initialHighCPUStr, step1HighCPUStr))
		step1Containers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: step1HighCPUStr,
					CPULim: step1HighCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patchStep1 := podresize.MakeResizePatch(originalContainers, step1Containers, nil, nil)
		_, patchErr := cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patchStep1, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch pod for step 1 resize")

		ginkgo.By("Verifying scheduler preempts non-PDB victim pod while preserving PDB-protected pod")
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, nonPdbVictimPod.Name, metav1.GetOptions{})
			if err != nil {
				return apierrors.IsNotFound(err)
			}
			return p.DeletionTimestamp != nil
		}).WithTimeout(45 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "Non-PDB victim pod should be preempted")

		livePdbVictim, err := cs.CoreV1().Pods(ns).Get(ctx, pdbVictimPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(livePdbVictim.DeletionTimestamp).To(gomega.BeNil(), "PDB-protected victim pod must not be preempted when alternative non-PDB victim exists")

		ginkgo.By("Waiting for step 1 resize actuation to complete on node")
		step1Expected := podresize.UpdateExpectedContainerRestarts(ctx, highPod, step1Containers)
		step1Pod := podresize.WaitForPodResizeActuation(ctx, f, podClient, highPod, step1Expected)
		gomega.Expect(step1Pod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
		gomega.Expect(step1Pod.Status.ContainerStatuses[0].AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(step1HighCPUStr)))

		ginkgo.By(fmt.Sprintf("Patching high-priority pod to expand CPU from %s to %s (step 2: forces preemption of PDB victim)", step1HighCPUStr, step2HighCPUStr))
		step2Containers := []podresize.ResizableContainerInfo{
			{
				Name: "c1",
				Resources: &cgroups.ContainerResources{
					CPUReq: step2HighCPUStr,
					CPULim: step2HighCPUStr,
					MemReq: "50Mi",
					MemLim: "50Mi",
				},
				CPUPolicy: ptr.To(v1.NotRequired),
				MemPolicy: ptr.To(v1.NotRequired),
			},
		}

		patchStep2 := podresize.MakeResizePatch(step1Containers, step2Containers, nil, nil)
		_, patchErr = cs.CoreV1().Pods(ns).Patch(ctx, highPod.Name, types.StrategicMergePatchType, patchStep2, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(patchErr, "failed to patch pod for step 2 resize")

		ginkgo.By("Verifying scheduler disrupts PDB-protected pod when no alternative victims exist")
		gomega.Eventually(ctx, func(ctx context.Context) bool {
			p, err := cs.CoreV1().Pods(ns).Get(ctx, pdbVictimPod.Name, metav1.GetOptions{})
			if err != nil {
				return apierrors.IsNotFound(err)
			}
			return p.DeletionTimestamp != nil
		}).WithTimeout(45 * time.Second).WithPolling(500 * time.Millisecond).Should(gomega.BeTrue(), "PDB-protected pod should be preempted when no alternative victims exist")

		ginkgo.By("Waiting for step 2 resize actuation to complete on node")
		step2Expected := podresize.UpdateExpectedContainerRestarts(ctx, step1Pod, step2Containers)
		step2Pod := podresize.WaitForPodResizeActuation(ctx, f, podClient, step1Pod, step2Expected)
		gomega.Expect(step2Pod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
		gomega.Expect(step2Pod.Status.ContainerStatuses[0].AllocatedResources[v1.ResourceCPU]).To(gomega.Equal(resource.MustParse(step2HighCPUStr)))
	})
})
