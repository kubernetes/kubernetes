/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	schedulerapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

var _ = SIGDescribe("SchedulerPreemption [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var ns string
	f := framework.NewDefaultFramework("sched-preemption")

	lowPriority, mediumPriority, highPriority := int32(1), int32(100), int32(1000)
	lowPriorityClassName := f.BaseName + "-low-priority"
	mediumPriorityClassName := f.BaseName + "-medium-priority"
	highPriorityClassName := f.BaseName + "-high-priority"
	AfterEach(func() {
	})

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		_, err := f.ClientSet.SchedulingV1beta1().PriorityClasses().Create(&schedulerapi.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: highPriorityClassName}, Value: highPriority})
		Expect(err == nil || errors.IsAlreadyExists(err)).To(Equal(true))
		_, err = f.ClientSet.SchedulingV1beta1().PriorityClasses().Create(&schedulerapi.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: mediumPriorityClassName}, Value: mediumPriority})
		Expect(err == nil || errors.IsAlreadyExists(err)).To(Equal(true))
		_, err = f.ClientSet.SchedulingV1beta1().PriorityClasses().Create(&schedulerapi.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: lowPriorityClassName}, Value: lowPriority})
		Expect(err == nil || errors.IsAlreadyExists(err)).To(Equal(true))

		framework.WaitForAllNodesHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		err = framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	// This test verifies that when a higher priority pod is created and no node with
	// enough resources is found, scheduler preempts a lower priority pod to schedule
	// the high priority pod.
	It("validates basic preemption works", func() {
		var podRes v1.ResourceList
		// Create one pod per node that uses a lot of the node's resources.
		By("Create pods that use 60% of node resources.")
		pods := make([]*v1.Pod, len(nodeList.Items))
		for i, node := range nodeList.Items {
			cpuAllocatable, found := node.Status.Allocatable["cpu"]
			Expect(found).To(Equal(true))
			milliCPU := cpuAllocatable.MilliValue() * 40 / 100
			memAllocatable, found := node.Status.Allocatable["memory"]
			Expect(found).To(Equal(true))
			memory := memAllocatable.Value() * 60 / 100
			podRes = v1.ResourceList{}
			podRes[v1.ResourceCPU] = *resource.NewMilliQuantity(int64(milliCPU), resource.DecimalSI)
			podRes[v1.ResourceMemory] = *resource.NewQuantity(int64(memory), resource.BinarySI)

			// make the first pod low priority and the rest medium priority.
			priorityName := mediumPriorityClassName
			if i == 0 {
				priorityName = lowPriorityClassName
			}
			pods[i] = createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, priorityName),
				PriorityClassName: priorityName,
				Resources: &v1.ResourceRequirements{
					Requests: podRes,
				},
			})
			framework.Logf("Created pod: %v", pods[i].Name)
		}
		By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(cs, pod))
		}

		By("Run a high priority pod that use 60% of a node resources.")
		// Create a high priority pod and make sure it is scheduled.
		runPausePod(f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
			},
		})
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(pods[0].Name, metav1.GetOptions{})
		podDeleted := (err != nil && errors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		Expect(podDeleted).To(BeTrue())
		// Other pods (mid priority ones) should be present.
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(livePod.DeletionTimestamp).To(BeNil())
		}
	})

	// This test verifies that when a critical pod is created and no node with
	// enough resources is found, scheduler preempts a lower priority pod to schedule
	// this critical pod.
	It("validates lower priority pod preemption by critical pod", func() {
		var podRes v1.ResourceList
		// Create one pod per node that uses a lot of the node's resources.
		By("Create pods that use 60% of node resources.")
		pods := make([]*v1.Pod, len(nodeList.Items))
		for i, node := range nodeList.Items {
			cpuAllocatable, found := node.Status.Allocatable["cpu"]
			Expect(found).To(Equal(true))
			milliCPU := cpuAllocatable.MilliValue() * 40 / 100
			memAllocatable, found := node.Status.Allocatable["memory"]
			Expect(found).To(Equal(true))
			memory := memAllocatable.Value() * 60 / 100
			podRes = v1.ResourceList{}
			podRes[v1.ResourceCPU] = *resource.NewMilliQuantity(int64(milliCPU), resource.DecimalSI)
			podRes[v1.ResourceMemory] = *resource.NewQuantity(int64(memory), resource.BinarySI)

			// make the first pod low priority and the rest medium priority.
			priorityName := mediumPriorityClassName
			if i == 0 {
				priorityName = lowPriorityClassName
			}
			pods[i] = createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, priorityName),
				PriorityClassName: priorityName,
				Resources: &v1.ResourceRequirements{
					Requests: podRes,
				},
			})
			framework.Logf("Created pod: %v", pods[i].Name)
		}
		By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(cs, pod))
		}

		By("Run a critical pod that use 60% of a node resources.")
		// Create a critical pod and make sure it is scheduled.
		runPausePod(f, pausePodConfig{
			Name:              "critical-pod",
			Namespace:         metav1.NamespaceSystem,
			PriorityClassName: scheduling.SystemClusterCritical,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
			},
		})
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(pods[0].Name, metav1.GetOptions{})
		podDeleted := (err != nil && errors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		Expect(podDeleted).To(BeTrue())
		// Other pods (mid priority ones) should be present.
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(livePod.DeletionTimestamp).To(BeNil())
		}
		// Clean-up the critical pod
		err = f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete("critical-pod", metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err)
	})

	// This test verifies that when a high priority pod is pending and its
	// scheduling violates a medium priority pod anti-affinity, the medium priority
	// pod is preempted to allow the higher priority pod schedule.
	// It also verifies that existing low priority pods are not preempted as their
	// preemption wouldn't help.
	It("validates pod anti-affinity works in preemption", func() {
		var podRes v1.ResourceList
		// Create a few pods that uses a small amount of resources.
		By("Create pods that use 10% of node resources.")
		numPods := 4
		if len(nodeList.Items) < numPods {
			numPods = len(nodeList.Items)
		}
		pods := make([]*v1.Pod, numPods)
		for i := 0; i < numPods; i++ {
			node := nodeList.Items[i]
			cpuAllocatable, found := node.Status.Allocatable["cpu"]
			Expect(found).To(BeTrue())
			milliCPU := cpuAllocatable.MilliValue() * 10 / 100
			memAllocatable, found := node.Status.Allocatable["memory"]
			Expect(found).To(BeTrue())
			memory := memAllocatable.Value() * 10 / 100
			podRes = v1.ResourceList{}
			podRes[v1.ResourceCPU] = *resource.NewMilliQuantity(int64(milliCPU), resource.DecimalSI)
			podRes[v1.ResourceMemory] = *resource.NewQuantity(int64(memory), resource.BinarySI)

			// Apply node label to each node
			framework.AddOrUpdateLabelOnNode(cs, node.Name, "node", node.Name)
			framework.ExpectNodeHasLabel(cs, node.Name, "node", node.Name)

			// make the first pod medium priority and the rest low priority.
			priorityName := lowPriorityClassName
			if i == 0 {
				priorityName = mediumPriorityClassName
			}
			pods[i] = createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, priorityName),
				PriorityClassName: priorityName,
				Resources: &v1.ResourceRequirements{
					Requests: podRes,
				},
				Affinity: &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "service",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"blah", "foo"},
										},
									},
								},
								TopologyKey: "node",
							},
						},
					},
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      "node",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{node.Name},
										},
									},
								},
							},
						},
					},
				},
			})
			framework.Logf("Created pod: %v", pods[i].Name)
		}
		defer func() { // Remove added labels
			for i := 0; i < numPods; i++ {
				framework.RemoveLabelOffNode(cs, nodeList.Items[i].Name, "node")
			}
		}()

		By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(framework.WaitForPodRunningInNamespace(cs, pod))
		}

		By("Run a high priority pod with node affinity to the first node.")
		// Create a high priority pod and make sure it is scheduled.
		runPausePod(f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Labels:            map[string]string{"service": "blah"},
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "node",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{nodeList.Items[0].Name},
									},
								},
							},
						},
					},
				},
			},
		})
		// Make sure that the medium priority pod on the first node is preempted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(pods[0].Name, metav1.GetOptions{})
		podDeleted := (err != nil && errors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		Expect(podDeleted).To(BeTrue())
		// Other pods (low priority ones) should be present.
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			Expect(livePod.DeletionTimestamp).To(BeNil())
		}
	})
})

var _ = SIGDescribe("PodPriorityResolution [Serial]", func() {
	var cs clientset.Interface
	var ns string
	f := framework.NewDefaultFramework("sched-pod-priority")

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	// This test verifies that system critical priorities are created automatically and resolved properly.
	It("validates critical system priorities are created and resolved", func() {
		// Create pods that use system critical priorities and
		By("Create pods that use critical system priorities.")
		systemPriorityClasses := []string{
			scheduling.SystemNodeCritical, scheduling.SystemClusterCritical,
		}
		for i, spc := range systemPriorityClasses {
			pod := createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, spc),
				Namespace:         metav1.NamespaceSystem,
				PriorityClassName: spc,
			})
			Expect(pod.Spec.Priority).NotTo(BeNil())
			framework.Logf("Created pod: %v", pod.Name)
			// Clean-up the pod.
			err := f.ClientSet.CoreV1().Pods(pod.Namespace).Delete(pod.Name, metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)
		}
	})
})
