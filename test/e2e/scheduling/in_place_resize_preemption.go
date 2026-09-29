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
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

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

		// Calculate available allocatable CPU on the node
		allocatableCPU := targetNodeObj.Status.Allocatable[v1.ResourceCPU]
		totalAllocatableMilli := allocatableCPU.MilliValue()

		// Get currently requested CPU from all active pods on this node
		podList, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{FieldSelector: "spec.nodeName=" + targetNode})
		framework.ExpectNoError(err, "failed to list pods assigned to node %s", targetNode)

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
		framework.Logf("Node %s has total allocatable CPU: %dm, used: %dm, free: %dm", targetNode, totalAllocatableMilli, usedMilliCPU, freeMilliCPU)
		if freeMilliCPU < 200 {
			freeMilliCPU = 200
		}

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
			Name:              "victim-pod",
			Namespace:         ns,
			PriorityClassName: lowPriorityClassName,
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchFields: []v1.NodeSelectorRequirement{
									{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{targetNode}},
								},
							},
						},
					},
				},
			},
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
		victimPod.Spec.TerminationGracePeriodSeconds = &zeroGracePeriod
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
		highPodSpec.Spec.Affinity = &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchFields: []v1.NodeSelectorRequirement{
								{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{targetNode}},
							},
						},
					},
				},
			},
		}

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
})
