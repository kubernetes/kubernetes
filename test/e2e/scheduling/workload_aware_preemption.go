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

package scheduling

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const extendedResourceDomain = "example.com/"

var (
	gangPolicy = schedulingv1beta1.PodGroupSchedulingPolicy{
		Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 2},
	}
	singleDisruption = schedulingv1beta1.DisruptionMode{
		Single: &schedulingv1beta1.SingleDisruptionMode{},
	}
	allDisruption = schedulingv1beta1.DisruptionMode{
		All: &schedulingv1beta1.AllDisruptionMode{},
	}

	cpgGangPolicy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.CompositeGangSchedulingPolicy{MinGroupCount: 2},
	}
	cpgBasicPolicy = schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
		Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
	}
	singleCompositeDisruption = schedulingv1alpha3.CompositeDisruptionMode{
		Single: &schedulingv1alpha3.SingleCompositeDisruptionMode{},
	}
	allCompositeDisruption = schedulingv1alpha3.CompositeDisruptionMode{
		All: &schedulingv1alpha3.AllCompositeDisruptionMode{},
	}
)

type preemptorType int

const (
	pod preemptorType = iota
	podGroup
	compositePodGroup
)

var _ = SIGDescribe("WorkloadAwarePreemption", framework.WithFeatureGate(features.GenericWorkload), framework.WithFeatureGate(features.CompositePodGroup), func() {
	f := framework.NewDefaultFramework("workload-aware-preemption")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	createPodGroup := func(ctx context.Context, pg *schedulingv1beta1.PodGroup) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		_, err := cs.SchedulingV1beta1().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup %s", pg.Name)
	}

	deletePodGroup := func(ctx context.Context, name string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		ginkgo.By("Deleting PodGroup")
		err := cs.SchedulingV1beta1().PodGroups(ns).Delete(ctx, name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "failed to delete PodGroup")
		}
	}

	createCompositePodGroup := func(ctx context.Context, cpg *schedulingv1alpha3.CompositePodGroup) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		_, err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Create(ctx, cpg, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create CompositePodGroup %s", cpg.Name)
	}

	deleteCompositePodGroup := func(ctx context.Context, name string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		ginkgo.By("Deleting CompositePodGroup")
		err := cs.SchedulingV1alpha3().CompositePodGroups(ns).Delete(ctx, name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "failed to delete CompositePodGroup")
		}
	}

	createPriorityClass := func(ctx context.Context, pc *schedulingv1.PriorityClass) {
		cs := f.ClientSet
		_, err := cs.SchedulingV1().PriorityClasses().Create(ctx, pc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create priority class %s", pc.Name)
	}

	deletePriorityClass := func(ctx context.Context, name string) {
		cs := f.ClientSet
		ginkgo.By("Deleting priority class")
		err := cs.SchedulingV1().PriorityClasses().Delete(ctx, name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "failed to delete priority class: %s", name)
		}
	}

	addExtendedResource := func(ctx context.Context, nodeName string, resourceName v1.ResourceName) {
		cs := f.ClientSet
		e2enode.AddExtendedResource(ctx, cs, nodeName, resourceName, resource.MustParse("2"))
	}

	removeExtendedResource := func(ctx context.Context, nodeName string, resourceName v1.ResourceName) {
		cs := f.ClientSet
		ginkgo.By("Removing extended resource from node")
		e2enode.RemoveExtendedResource(ctx, cs, nodeName, resourceName)
	}

	makePod := func(nodeName string, name, pgName string, priority string, resourceName v1.ResourceName) *v1.Pod {
		ns := f.Namespace.Name
		p := e2epod.MakePod(ns, map[string]string{"kubernetes.io/hostname": nodeName}, nil, admissionapi.LevelPrivileged, "")
		p.ObjectMeta.GenerateName = name + "-"
		p.Spec.PriorityClassName = priority
		if pgName != "" {
			p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: &pgName}
		}
		p.Spec.Containers[0].Resources.Requests = v1.ResourceList{resourceName: resource.MustParse("1")}
		p.Spec.Containers[0].Resources.Limits = v1.ResourceList{resourceName: resource.MustParse("1")}
		return p
	}

	makePodGroup := func(pgName string, priorityName string, schedulingPolicy schedulingv1beta1.PodGroupSchedulingPolicy, disruptionMode schedulingv1beta1.DisruptionMode) *schedulingv1beta1.PodGroup {
		ns := f.Namespace.Name

		return &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: pgName, Namespace: ns},
			Spec: schedulingv1beta1.PodGroupSpec{
				PriorityClassName: priorityName,
				SchedulingPolicy:  schedulingPolicy,
				DisruptionMode:    &disruptionMode,
			},
		}
	}

	makeCompositePodGroup := func(name, priorityName string, policy schedulingv1alpha3.CompositePodGroupSchedulingPolicy, disruptionMode schedulingv1alpha3.CompositeDisruptionMode) *schedulingv1alpha3.CompositePodGroup {
		ns := f.Namespace.Name
		return &schedulingv1alpha3.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
			Spec: schedulingv1alpha3.CompositePodGroupSpec{
				PriorityClassName: priorityName,
				WorkloadRef: &schedulingv1alpha3.WorkloadReference{
					WorkloadName: "workload-" + name,
					TemplateName: "template-" + name,
				},
				SchedulingPolicy: policy,
				DisruptionMode:   &disruptionMode,
			},
		}
	}

	makeChildPodGroup := func(pgName, priorityName, parentCPG string, schedulingPolicy schedulingv1beta1.PodGroupSchedulingPolicy, disruptionMode schedulingv1beta1.DisruptionMode) *schedulingv1beta1.PodGroup {
		ns := f.Namespace.Name
		return &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: pgName, Namespace: ns},
			Spec: schedulingv1beta1.PodGroupSpec{
				PriorityClassName:           priorityName,
				ParentCompositePodGroupName: &parentCPG,
				WorkloadRef: &schedulingv1beta1.WorkloadReference{
					WorkloadName: "workload-" + parentCPG,
					TemplateName: "template-" + pgName,
				},
				SchedulingPolicy: schedulingPolicy,
				DisruptionMode:   &disruptionMode,
			},
		}
	}

	getNodeName := func(ctx context.Context) string {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		return node.Name
	}

	getTwoNodeNames := func(ctx context.Context) (string, string) {
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get ready schedulable nodes")
		node1 := nodeList.Items[0].Name
		node2 := nodeList.Items[0].Name
		if len(nodeList.Items) > 1 {
			node2 = nodeList.Items[1].Name
		}
		return node1, node2
	}

	isPodPreempted := func(ctx context.Context, podName string) bool {
		cs := f.ClientSet
		ns := f.Namespace.Name
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				return true
			}
			framework.ExpectNoError(err, "failed to get pod %s", podName)
		}
		return pod.DeletionTimestamp != nil
	}

	verifyPodRunningOnNode := func(ctx context.Context, podName, nodeName string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, podName, ns), "pod %s failed to run", podName)
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %s", podName)
		gomega.Expect(pod.Spec.NodeName).To(gomega.Equal(nodeName))
	}

	verifyAllPreempted := func(ctx context.Context, pods []*v1.Pod) {
		ginkgo.By("Verifying all pods in victim workload are preempted")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			for _, p := range pods {
				if !isPodPreempted(ctx, p.Name) {
					return fmt.Errorf("pod %s is not preempted yet", p.Name)
				}
			}
			return nil
		}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Succeed(), "All pods in victim workload should eventually be preempted")
	}

	verifyPartialPreempted := func(ctx context.Context, pods []*v1.Pod) {
		ginkgo.By("Verifying at least one pod in victim is preempted and one remains running")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			preemptedCount := 0
			for _, p := range pods {
				if isPodPreempted(ctx, p.Name) {
					preemptedCount++
				}
			}
			if preemptedCount == 0 {
				return fmt.Errorf("no pods preempted yet")
			}
			return nil
		}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Succeed(), "Expected at least one pod from victim to be preempted")

		gomega.Consistently(ctx, func(ctx context.Context) error {
			preemptedCount := 0
			for _, p := range pods {
				if isPodPreempted(ctx, p.Name) {
					preemptedCount++
				}
			}
			if preemptedCount >= len(pods) {
				return fmt.Errorf("expected at least one pod to remain running, but all pods were preempted")
			}
			return nil
		}).WithTimeout(5 * time.Second).WithPolling(1 * time.Second).Should(gomega.Succeed())
	}

	type preemptionTestArgs struct {
		preemptorType        preemptorType
		victimDisruptionMode schedulingv1beta1.DisruptionMode
		verify               func(context.Context, []*v1.Pod)
	}

	runPreemptionTest := func(ctx context.Context, args preemptionTestArgs) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

		ginkgo.By("Creating PriorityClasses")
		lowPriorityName := "low-priority-" + ns
		highPriorityName := "high-priority-" + ns

		createPriorityClass(ctx, &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
			Value:      100,
		})
		defer deletePriorityClass(ctx, lowPriorityName)
		createPriorityClass(ctx, &schedulingv1.PriorityClass{
			ObjectMeta: metav1.ObjectMeta{Name: highPriorityName},
			Value:      1000,
		})
		defer deletePriorityClass(ctx, highPriorityName)

		nodeName := getNodeName(ctx)
		ginkgo.By("Adding extended resource to node")
		addExtendedResource(ctx, nodeName, extendedResourceName)
		defer removeExtendedResource(ctx, nodeName, extendedResourceName)

		ginkgo.By(fmt.Sprintf("Creating low-priority pod group PG-victim with disruptionMode %v", args.victimDisruptionMode))
		pgVictimName := "pg-victim-" + ns
		pgVictim := makePodGroup(pgVictimName, lowPriorityName, gangPolicy, args.victimDisruptionMode)
		createPodGroup(ctx, pgVictim)
		defer deletePodGroup(ctx, pgVictim.Name)

		var pods []*v1.Pod
		for i := range gangPolicy.Gang.MinCount {
			name := fmt.Sprintf("victim%d", i+1)
			ginkgo.By(fmt.Sprintf("Creating low-priority pod %s belonging to PG-victim", name))
			p := makePod(nodeName, name, pgVictimName, lowPriorityName, extendedResourceName)
			createdPod, err := cs.CoreV1().Pods(ns).Create(ctx, p, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create pod %s", name)
			pods = append(pods, createdPod)
		}

		ginkgo.By("Verifying all low priority pods are running")
		for _, p := range pods {
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns), "pod %s failed to run", p.Name)
		}

		var pgPreemptorName string
		if args.preemptorType == podGroup {
			pgPreemptorName = "pg-preemptor-" + ns
			ginkgo.By("Creating high-priority pod group PG-preemptor with gang policy")
			pgPreemptor := makePodGroup(pgPreemptorName, highPriorityName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor)
			defer deletePodGroup(ctx, pgPreemptor.Name)
		}

		ginkgo.By("Creating high-priority individual pod hp1")
		hp1 := makePod(nodeName, "hp1", pgPreemptorName, highPriorityName, extendedResourceName)
		var err error
		hp1, err = cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod hp1")

		ginkgo.By("Verifying high priority pods are running")
		verifyPodRunningOnNode(ctx, hp1.Name, nodeName)

		args.verify(ctx, pods)
	}

	ginkgo.DescribeTable("workload-aware preemption", runPreemptionTest,
		ginkgo.Entry("should preempt entire group with All disruption mode by a pod group", preemptionTestArgs{
			preemptorType:        podGroup,
			victimDisruptionMode: allDisruption,
			verify:               verifyAllPreempted,
		}),
		ginkgo.Entry("should preempt partial group with Single disruption mode by a pod group", preemptionTestArgs{
			preemptorType:        podGroup,
			victimDisruptionMode: singleDisruption,
			verify:               verifyPartialPreempted,
		}),
		ginkgo.Entry("should preempt partial group with Single disruption mode by an individual pod", preemptionTestArgs{
			preemptorType:        pod,
			victimDisruptionMode: singleDisruption,
			verify:               verifyPartialPreempted,
		}),
	)

	ginkgo.Describe("CompositePodGroup with child pod groups preemption", func() {
		ginkgo.It("should preempt entire CompositePodGroup with child pod groups across multiple nodes when DisruptionMode is All", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-all-" + ns
			highPriorityName := "high-priority-cpg-all-" + ns

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
				Value:      100,
			})
			defer deletePriorityClass(ctx, lowPriorityName)
			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: highPriorityName},
				Value:      1000,
			})
			defer deletePriorityClass(ctx, highPriorityName)

			node1, node2 := getTwoNodeNames(ctx)
			addExtendedResource(ctx, node1, extendedResourceName)
			defer removeExtendedResource(ctx, node1, extendedResourceName)
			if node2 != node1 {
				addExtendedResource(ctx, node2, extendedResourceName)
				defer removeExtendedResource(ctx, node2, extendedResourceName)
			}

			cpgVictimName := "cpg-victim-all-" + ns
			cpgVictim := makeCompositePodGroup(cpgVictimName, lowPriorityName, cpgBasicPolicy, allCompositeDisruption)
			createCompositePodGroup(ctx, cpgVictim)
			defer deleteCompositePodGroup(ctx, cpgVictimName)

			pg1Name := "pg-child1-" + ns
			pg1 := makeChildPodGroup(pg1Name, lowPriorityName, cpgVictimName, gangPolicy, allDisruption)
			createPodGroup(ctx, pg1)
			defer deletePodGroup(ctx, pg1Name)

			pg2Name := "pg-child2-" + ns
			pg2 := makeChildPodGroup(pg2Name, lowPriorityName, cpgVictimName, gangPolicy, allDisruption)
			createPodGroup(ctx, pg2)
			defer deletePodGroup(ctx, pg2Name)

			var victimPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("v1-%d", i), pg1Name, lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				victimPods = append(victimPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("v2-%d", i), pg2Name, lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				victimPods = append(victimPods, createdP2)
			}

			for _, p := range victimPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high priority preemptor targeting node1")
			hpPod := makePod(node1, "cpg-hp1", "", highPriorityName, extendedResourceName)
			createdHP, err := cs.CoreV1().Pods(ns).Create(ctx, hpPod, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			verifyPodRunningOnNode(ctx, createdHP.Name, node1)
			verifyAllPreempted(ctx, victimPods)
		})

		ginkgo.It("should preempt only affected child pod group in CompositePodGroup when DisruptionMode is Single", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-single-" + ns
			highPriorityName := "high-priority-cpg-single-" + ns

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
				Value:      100,
			})
			defer deletePriorityClass(ctx, lowPriorityName)
			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: highPriorityName},
				Value:      1000,
			})
			defer deletePriorityClass(ctx, highPriorityName)

			node1, node2 := getTwoNodeNames(ctx)
			addExtendedResource(ctx, node1, extendedResourceName)
			defer removeExtendedResource(ctx, node1, extendedResourceName)
			if node2 != node1 {
				addExtendedResource(ctx, node2, extendedResourceName)
				defer removeExtendedResource(ctx, node2, extendedResourceName)
			}

			cpgVictimName := "cpg-victim-single-" + ns
			cpgVictim := makeCompositePodGroup(cpgVictimName, lowPriorityName, cpgBasicPolicy, singleCompositeDisruption)
			createCompositePodGroup(ctx, cpgVictim)
			defer deleteCompositePodGroup(ctx, cpgVictimName)

			pg1Name := "pg-single-child1-" + ns
			pg1 := makeChildPodGroup(pg1Name, lowPriorityName, cpgVictimName, gangPolicy, allDisruption)
			createPodGroup(ctx, pg1)
			defer deletePodGroup(ctx, pg1Name)

			pg2Name := "pg-single-child2-" + ns
			pg2 := makeChildPodGroup(pg2Name, lowPriorityName, cpgVictimName, gangPolicy, allDisruption)
			createPodGroup(ctx, pg2)
			defer deletePodGroup(ctx, pg2Name)

			var node1Pods []*v1.Pod
			var node2Pods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("v1-%d", i), pg1Name, lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				node1Pods = append(node1Pods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("v2-%d", i), pg2Name, lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				node2Pods = append(node2Pods, createdP2)
			}

			for _, p := range append(node1Pods, node2Pods...) {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high priority preemptor targeting node1")
			hpPod := makePod(node1, "cpg-single-hp1", "", highPriorityName, extendedResourceName)
			createdHP, err := cs.CoreV1().Pods(ns).Create(ctx, hpPod, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			verifyPodRunningOnNode(ctx, createdHP.Name, node1)

			ginkgo.By("Verifying node1 child pod group is preempted while node2 child pod group remains running")
			verifyAllPreempted(ctx, node1Pods)
			if node2 != node1 {
				gomega.Consistently(ctx, func(ctx context.Context) error {
					for _, p := range node2Pods {
						if isPodPreempted(ctx, p.Name) {
							return fmt.Errorf("pod %s on node2 should not be preempted", p.Name)
						}
					}
					return nil
				}).WithTimeout(5 * time.Second).WithPolling(1 * time.Second).Should(gomega.Succeed())
			}
		})

		ginkgo.It("should perform multi-node gang preemption with CompositePodGroup under capacity constraints", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-gang-" + ns
			highPriorityName := "high-priority-cpg-gang-" + ns

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
				Value:      100,
			})
			defer deletePriorityClass(ctx, lowPriorityName)
			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: highPriorityName},
				Value:      1000,
			})
			defer deletePriorityClass(ctx, highPriorityName)

			node1, node2 := getTwoNodeNames(ctx)
			addExtendedResource(ctx, node1, extendedResourceName)
			defer removeExtendedResource(ctx, node1, extendedResourceName)
			if node2 != node1 {
				addExtendedResource(ctx, node2, extendedResourceName)
				defer removeExtendedResource(ctx, node2, extendedResourceName)
			}

			ginkgo.By("Creating low-priority victim pods occupying resources on both nodes")
			var victimPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("victim-n1-%d", i), "", lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				victimPods = append(victimPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("victim-n2-%d", i), "", lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				victimPods = append(victimPods, createdP2)
			}

			for _, p := range victimPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high-priority CompositePodGroup with multi-node gang requirements")
			cpgPreemptorName := "cpg-preemptor-gang-" + ns
			cpgPreemptor := makeCompositePodGroup(cpgPreemptorName, highPriorityName, cpgGangPolicy, singleCompositeDisruption)
			createCompositePodGroup(ctx, cpgPreemptor)
			defer deleteCompositePodGroup(ctx, cpgPreemptorName)

			pgPreemptor1Name := "pg-preemptor-child1-" + ns
			pgPreemptor1 := makeChildPodGroup(pgPreemptor1Name, highPriorityName, cpgPreemptorName, gangPolicy, singleDisruption)
			createPodGroup(ctx, pgPreemptor1)
			defer deletePodGroup(ctx, pgPreemptor1Name)

			pgPreemptor2Name := "pg-preemptor-child2-" + ns
			pgPreemptor2 := makeChildPodGroup(pgPreemptor2Name, highPriorityName, cpgPreemptorName, gangPolicy, singleDisruption)
			createPodGroup(ctx, pgPreemptor2)
			defer deletePodGroup(ctx, pgPreemptor2Name)

			ginkgo.By("Creating high-priority gang pods across node1 and node2")
			var hpPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				hp1 := makePod(node1, fmt.Sprintf("hp-n1-%d", i), pgPreemptor1Name, highPriorityName, extendedResourceName)
				createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				hpPods = append(hpPods, createdHP1)

				hp2 := makePod(node2, fmt.Sprintf("hp-n2-%d", i), pgPreemptor2Name, highPriorityName, extendedResourceName)
				createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				hpPods = append(hpPods, createdHP2)
			}

			ginkgo.By("Verifying all high-priority gang pods are scheduled and running across nodes")
			for _, p := range hpPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns), "hp pod %s failed to run", p.Name)
			}

			ginkgo.By("Verifying victim pods were preempted to accommodate the multi-node gang")
			verifyAllPreempted(ctx, victimPods)
		})
	})
})
