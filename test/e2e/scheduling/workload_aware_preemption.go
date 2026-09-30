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
	policyv1 "k8s.io/api/policy/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
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

var _ = SIGDescribe("WorkloadAwarePreemption", framework.WithFeatureGate(features.GenericWorkload), framework.WithFeatureGate(features.CompositePodGroup), framework.WithFeatureGate(features.PodGroupPreemptionPolicy), func() {
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
		var nodeSelector map[string]string
		if nodeName != "" {
			nodeSelector = map[string]string{"kubernetes.io/hostname": nodeName}
		}
		p := e2epod.MakePod(ns, nodeSelector, nil, admissionapi.LevelPrivileged, "")
		p.ObjectMeta.GenerateName = name + "-"
		p.Spec.PriorityClassName = priority
		if pgName != "" {
			p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: &pgName}
		}
		p.Spec.Containers[0].Resources.Requests = v1.ResourceList{resourceName: resource.MustParse("1")}
		p.Spec.Containers[0].Resources.Limits = v1.ResourceList{resourceName: resource.MustParse("1")}
		return p
	}

	makePodWithLabels := func(nodeName string, name, pgName string, priority string, resourceName v1.ResourceName, labels map[string]string) *v1.Pod {
		p := makePod(nodeName, name, pgName, priority, resourceName)
		if p.Labels == nil {
			p.Labels = make(map[string]string)
		}
		for k, v := range labels {
			p.Labels[k] = v
		}
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

	makePodGroupWithPreemptionPolicy := func(pgName string, priorityName string, schedulingPolicy schedulingv1beta1.PodGroupSchedulingPolicy, disruptionMode schedulingv1beta1.DisruptionMode, preemptionPolicy schedulingv1beta1.PreemptionPolicy) *schedulingv1beta1.PodGroup {
		pg := makePodGroup(pgName, priorityName, schedulingPolicy, disruptionMode)
		pg.Spec.PreemptionPolicy = &preemptionPolicy
		return pg
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

	makeCompositePodGroupWithPreemptionPolicy := func(name, priorityName string, policy schedulingv1alpha3.CompositePodGroupSchedulingPolicy, disruptionMode schedulingv1alpha3.CompositeDisruptionMode, preemptionPolicy schedulingv1alpha3.PreemptionPolicy) *schedulingv1alpha3.CompositePodGroup {
		cpg := makeCompositePodGroup(name, priorityName, policy, disruptionMode)
		cpg.Spec.PreemptionPolicy = &preemptionPolicy
		return cpg
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

	makeChildPodGroupWithPreemptionPolicy := func(pgName, priorityName, parentCPG string, schedulingPolicy schedulingv1beta1.PodGroupSchedulingPolicy, disruptionMode schedulingv1beta1.DisruptionMode, preemptionPolicy schedulingv1beta1.PreemptionPolicy) *schedulingv1beta1.PodGroup {
		pg := makeChildPodGroup(pgName, priorityName, parentCPG, schedulingPolicy, disruptionMode)
		pg.Spec.PreemptionPolicy = &preemptionPolicy
		return pg
	}

	getNodeName := func(ctx context.Context) string {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		return node.Name
	}

	getTwoNodeNames := func(ctx context.Context) (string, string) {
		e2eskipper.SkipUnlessNodeCountIsAtLeast(2)
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get ready schedulable nodes")
		return nodeList.Items[0].Name, nodeList.Items[1].Name
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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

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
			gomega.Consistently(ctx, func(ctx context.Context) error {
				for _, p := range node2Pods {
					if isPodPreempted(ctx, p.Name) {
						return fmt.Errorf("pod %s on node2 should not be preempted", p.Name)
					}
				}
				return nil
			}).WithTimeout(5 * time.Second).WithPolling(1 * time.Second).Should(gomega.Succeed())
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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

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

	ginkgo.Describe("Gang PreemptionPolicy PreemptNever", func() {
		ginkgo.It("should not preempt running lower-priority pods when high-priority PodGroup has PreemptionPolicy: PreemptNever", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-pg-never-" + ns
			highPriorityName := "high-priority-pg-never-" + ns

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
				Value:      100,
			})
			defer deletePriorityClass(ctx, lowPriorityName)

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: highPriorityName},
				Value:            1000,
				PreemptionPolicy: ptr.To(v1.PreemptNever),
			})
			defer deletePriorityClass(ctx, highPriorityName)

			node1, node2 := getTwoNodeNames(ctx)
			addExtendedResource(ctx, node1, extendedResourceName)
			defer removeExtendedResource(ctx, node1, extendedResourceName)
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority running pods saturating capacity across nodes")
			var lowPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("low-pg-never-n1-%d", i), "", lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("low-pg-never-n2-%d", i), "", lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP2)
			}
			for _, p := range lowPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high-priority PodGroup with PreemptionPolicy: PreemptNever")
			pgName := "hp-never-pg-" + ns
			pg := makePodGroupWithPreemptionPolicy(pgName, highPriorityName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 2},
			}, singleDisruption, schedulingv1beta1.PreemptNever)
			createPodGroup(ctx, pg)
			defer deletePodGroup(ctx, pgName)

			ginkgo.By("Creating high-priority gang pods requiring resources on saturated nodes")
			var hpPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				hp := makePod(node1, fmt.Sprintf("hp-never-pod-%d", i), pgName, highPriorityName, extendedResourceName)
				hp.Spec.PreemptionPolicy = ptr.To(v1.PreemptNever)
				createdHP, err := cs.CoreV1().Pods(ns).Create(ctx, hp, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				hpPods = append(hpPods, createdHP)
			}

			ginkgo.By("Verifying high-priority gang pods remain Pending without preemption")
			gomega.Consistently(ctx, func() bool {
				for _, p := range hpPods {
					pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
					if err != nil || pod.Status.Phase != v1.PodPending {
						return false
					}
				}
				return true
			}, 10*time.Second, 1*time.Second).Should(gomega.BeTrue(), "High-priority gang pods with PreemptNever should remain Pending")

			ginkgo.By("Verifying no low-priority pods were preempted")
			for _, p := range lowPods {
				pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				gomega.Expect(pod.DeletionTimestamp).To(gomega.BeNil(), "Low priority pod must not be preempted")
				gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
			}
		})

		ginkgo.It("should not preempt running lower-priority pods when high-priority CompositePodGroup has PreemptionPolicy: PreemptNever", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-never-" + ns
			highPriorityName := "high-priority-cpg-never-" + ns

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta: metav1.ObjectMeta{Name: lowPriorityName},
				Value:      100,
			})
			defer deletePriorityClass(ctx, lowPriorityName)

			createPriorityClass(ctx, &schedulingv1.PriorityClass{
				ObjectMeta:       metav1.ObjectMeta{Name: highPriorityName},
				Value:            1000,
				PreemptionPolicy: ptr.To(v1.PreemptNever),
			})
			defer deletePriorityClass(ctx, highPriorityName)

			node1, node2 := getTwoNodeNames(ctx)
			addExtendedResource(ctx, node1, extendedResourceName)
			defer removeExtendedResource(ctx, node1, extendedResourceName)
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority running pods saturating capacity across nodes")
			var lowPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("low-cpg-never-n1-%d", i), "", lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("low-cpg-never-n2-%d", i), "", lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP2)
			}
			for _, p := range lowPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high-priority CompositePodGroup with PreemptionPolicy: PreemptNever")
			cpgName := "hp-never-cpg-" + ns
			cpg := makeCompositePodGroupWithPreemptionPolicy(cpgName, highPriorityName, cpgGangPolicy, singleCompositeDisruption, schedulingv1alpha3.PreemptNever)
			createCompositePodGroup(ctx, cpg)
			defer deleteCompositePodGroup(ctx, cpgName)

			pg1Name := "hp-never-cpg-child1-" + ns
			pg1 := makeChildPodGroupWithPreemptionPolicy(pg1Name, highPriorityName, cpgName, gangPolicy, singleDisruption, schedulingv1beta1.PreemptNever)
			createPodGroup(ctx, pg1)
			defer deletePodGroup(ctx, pg1Name)

			pg2Name := "hp-never-cpg-child2-" + ns
			pg2 := makeChildPodGroupWithPreemptionPolicy(pg2Name, highPriorityName, cpgName, gangPolicy, singleDisruption, schedulingv1beta1.PreemptNever)
			createPodGroup(ctx, pg2)
			defer deletePodGroup(ctx, pg2Name)

			ginkgo.By("Creating high-priority gang pods across nodes with PreemptionPolicy: PreemptNever")
			var hpPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				hp1 := makePod(node1, fmt.Sprintf("hp-never-cpg-n1-%d", i), pg1Name, highPriorityName, extendedResourceName)
				hp1.Spec.PreemptionPolicy = ptr.To(v1.PreemptNever)
				createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				hpPods = append(hpPods, createdHP1)

				hp2 := makePod(node2, fmt.Sprintf("hp-never-cpg-n2-%d", i), pg2Name, highPriorityName, extendedResourceName)
				hp2.Spec.PreemptionPolicy = ptr.To(v1.PreemptNever)
				createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				hpPods = append(hpPods, createdHP2)
			}

			ginkgo.By("Verifying high-priority gang pods remain Pending without preemption")
			gomega.Consistently(ctx, func() bool {
				for _, p := range hpPods {
					pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
					if err != nil || pod.Status.Phase != v1.PodPending {
						return false
					}
				}
				return true
			}, 10*time.Second, 1*time.Second).Should(gomega.BeTrue(), "High-priority CPG gang pods with PreemptNever should remain Pending")

			ginkgo.By("Verifying no low-priority pods were preempted across nodes")
			for _, p := range lowPods {
				pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				gomega.Expect(pod.DeletionTimestamp).To(gomega.BeNil(), "Low priority pod must not be preempted")
				gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
			}
		})
	})

	ginkgo.Describe("Workload-Aware Gang Preemption with Multi-Node PodDisruptionBudgets (PDBs)", func() {
		ginkgo.It("should respect multi-node PodDisruptionBudgets during PodGroup gang preemption and evict candidates with minimal disruption", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-pdb-gang-" + ns
			highPriorityName := "high-priority-pdb-gang-" + ns

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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority PDB-protected pods across node1 and node2")
			pdbPod1 := makePodWithLabels(node1, "pdb-protected-n1", "", lowPriorityName, extendedResourceName, map[string]string{"app": "pdb-gang-protected"})
			createdPDBPod1, err := cs.CoreV1().Pods(ns).Create(ctx, pdbPod1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			pdbPod2 := makePodWithLabels(node2, "pdb-protected-n2", "", lowPriorityName, extendedResourceName, map[string]string{"app": "pdb-gang-protected"})
			createdPDBPod2, err := cs.CoreV1().Pods(ns).Create(ctx, pdbPod2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Creating low-priority unprotected victim pods across node1 and node2")
			unprotPod1 := makePodWithLabels(node1, "unprotected-victim-n1", "", lowPriorityName, extendedResourceName, map[string]string{"app": "unprotected-n1"})
			createdUnprotPod1, err := cs.CoreV1().Pods(ns).Create(ctx, unprotPod1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			unprotPod2 := makePodWithLabels(node2, "unprotected-victim-n2", "", lowPriorityName, extendedResourceName, map[string]string{"app": "unprotected-n2"})
			createdUnprotPod2, err := cs.CoreV1().Pods(ns).Create(ctx, unprotPod2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPDBPod1.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPDBPod2.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdUnprotPod1.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdUnprotPod2.Name, ns))

			ginkgo.By("Creating PodDisruptionBudget protecting the 2 pdb-gang-protected pods (minAvailable=2)")
			minAvail := intstr.FromInt32(2)
			pdb := &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "gang-victim-pdb",
					Namespace: ns,
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &minAvail,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "pdb-gang-protected"},
					},
				},
			}
			_, err = cs.PolicyV1().PodDisruptionBudgets(ns).Create(ctx, pdb, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for PDB status to reflect 2 healthy pods and 0 disruptions allowed")
			err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
				curPDB, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(ctx, pdb.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return curPDB.Status.CurrentHealthy == 2 && curPDB.Status.DisruptionsAllowed == 0, nil
			})
			framework.ExpectNoError(err, "PDB status failed to become ready")

			ginkgo.By("Creating high-priority gang PodGroup requiring multi-node placement")
			pgName := "hp-pdb-gang-" + ns
			pg := makePodGroup(pgName, highPriorityName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 2},
			}, singleDisruption)
			createPodGroup(ctx, pg)
			defer deletePodGroup(ctx, pgName)

			ginkgo.By("Creating high-priority gang pods targeting node1 and node2")
			hp1 := makePod(node1, "hp-pdb-gang-n1", pgName, highPriorityName, extendedResourceName)
			createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			hp2 := makePod(node2, "hp-pdb-gang-n2", pgName, highPriorityName, extendedResourceName)
			createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying both high-priority gang pods are scheduled and running")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP1.Name, ns), "hp1 failed to run")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP2.Name, ns), "hp2 failed to run")

			ginkgo.By("Verifying unprotected low-priority pods were preempted while PDB-protected pods were preserved")
			verifyAllPreempted(ctx, []*v1.Pod{createdUnprotPod1, createdUnprotPod2})

			gomega.Consistently(ctx, func(ctx context.Context) error {
				for _, p := range []*v1.Pod{createdPDBPod1, createdPDBPod2} {
					if isPodPreempted(ctx, p.Name) {
						return fmt.Errorf("PDB protected pod %s should not be preempted", p.Name)
					}
				}
				return nil
			}).WithTimeout(5 * time.Second).WithPolling(1 * time.Second).Should(gomega.Succeed(), "PDB-protected pods must not be preempted when alternative victims exist")
		})

		ginkgo.It("should respect multi-node PodDisruptionBudgets during CompositePodGroup gang preemption", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-pdb-" + ns
			highPriorityName := "high-priority-cpg-pdb-" + ns

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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority PDB-protected pods across node1 and node2")
			pdbPod1 := makePodWithLabels(node1, "cpg-pdb-prot-n1", "", lowPriorityName, extendedResourceName, map[string]string{"app": "cpg-pdb-gang-protected"})
			createdPDBPod1, err := cs.CoreV1().Pods(ns).Create(ctx, pdbPod1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			pdbPod2 := makePodWithLabels(node2, "cpg-pdb-prot-n2", "", lowPriorityName, extendedResourceName, map[string]string{"app": "cpg-pdb-gang-protected"})
			createdPDBPod2, err := cs.CoreV1().Pods(ns).Create(ctx, pdbPod2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Creating low-priority unprotected victim pods across node1 and node2")
			unprotPod1 := makePodWithLabels(node1, "cpg-unprot-n1", "", lowPriorityName, extendedResourceName, map[string]string{"app": "cpg-unprot-n1"})
			createdUnprotPod1, err := cs.CoreV1().Pods(ns).Create(ctx, unprotPod1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			unprotPod2 := makePodWithLabels(node2, "cpg-unprot-n2", "", lowPriorityName, extendedResourceName, map[string]string{"app": "cpg-unprot-n2"})
			createdUnprotPod2, err := cs.CoreV1().Pods(ns).Create(ctx, unprotPod2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPDBPod1.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPDBPod2.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdUnprotPod1.Name, ns))
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdUnprotPod2.Name, ns))

			ginkgo.By("Creating PodDisruptionBudget protecting the 2 cpg-pdb-gang-protected pods (minAvailable=2)")
			minAvail := intstr.FromInt32(2)
			pdb := &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "cpg-gang-victim-pdb",
					Namespace: ns,
				},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &minAvail,
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{"app": "cpg-pdb-gang-protected"},
					},
				},
			}
			_, err = cs.PolicyV1().PodDisruptionBudgets(ns).Create(ctx, pdb, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for PDB status to reflect 2 healthy pods and 0 disruptions allowed")
			err = wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 30*time.Second, false, func(ctx context.Context) (bool, error) {
				curPDB, err := cs.PolicyV1().PodDisruptionBudgets(ns).Get(ctx, pdb.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return curPDB.Status.CurrentHealthy == 2 && curPDB.Status.DisruptionsAllowed == 0, nil
			})
			framework.ExpectNoError(err, "PDB status failed to become ready")

			ginkgo.By("Creating high-priority CompositePodGroup with multi-node gang requirements")
			cpgPreemptorName := "cpg-preemptor-pdb-" + ns
			cpgPreemptor := makeCompositePodGroup(cpgPreemptorName, highPriorityName, cpgGangPolicy, singleCompositeDisruption)
			createCompositePodGroup(ctx, cpgPreemptor)
			defer deleteCompositePodGroup(ctx, cpgPreemptorName)

			pgPreemptor1Name := "pg-cpg-pdb-child1-" + ns
			pgPreemptor1 := makeChildPodGroup(pgPreemptor1Name, highPriorityName, cpgPreemptorName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor1)
			defer deletePodGroup(ctx, pgPreemptor1Name)

			pgPreemptor2Name := "pg-cpg-pdb-child2-" + ns
			pgPreemptor2 := makeChildPodGroup(pgPreemptor2Name, highPriorityName, cpgPreemptorName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor2)
			defer deletePodGroup(ctx, pgPreemptor2Name)

			ginkgo.By("Creating high-priority gang pods targeting node1 and node2 under child PodGroups")
			hp1 := makePod(node1, "hp-cpg-pdb-n1", pgPreemptor1Name, highPriorityName, extendedResourceName)
			createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			hp2 := makePod(node2, "hp-cpg-pdb-n2", pgPreemptor2Name, highPriorityName, extendedResourceName)
			createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying both high-priority CPG gang pods are scheduled and running")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP1.Name, ns), "hp1 failed to run")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP2.Name, ns), "hp2 failed to run")

			ginkgo.By("Verifying unprotected low-priority pods were preempted while PDB-protected pods were preserved")
			verifyAllPreempted(ctx, []*v1.Pod{createdUnprotPod1, createdUnprotPod2})

			gomega.Consistently(ctx, func(ctx context.Context) error {
				for _, p := range []*v1.Pod{createdPDBPod1, createdPDBPod2} {
					if isPodPreempted(ctx, p.Name) {
						return fmt.Errorf("PDB protected pod %s should not be preempted", p.Name)
					}
				}
				return nil
			}).WithTimeout(5 * time.Second).WithPolling(1 * time.Second).Should(gomega.Succeed(), "PDB-protected pods must not be preempted")
		})
	})

	ginkgo.Describe("Gang Preemption with Inter-Pod Anti-Affinity", func() {
		ginkgo.It("should schedule high-priority gang pods on distinct nodes and preempt victims accordingly when inter-pod anti-affinity is specified", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-anti-" + ns
			highPriorityName := "high-priority-anti-" + ns

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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority pods saturating capacity on both nodes")
			var lowPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("low-anti-n1-%d", i), "", lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("low-anti-n2-%d", i), "", lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP2)
			}
			for _, p := range lowPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high-priority gang PodGroup with MinCount=2")
			pgName := "hp-gang-anti-" + ns
			pg := makePodGroup(pgName, highPriorityName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 2},
			}, singleDisruption)
			createPodGroup(ctx, pg)
			defer deletePodGroup(ctx, pgName)

			ginkgo.By("Creating high-priority gang pods with strict inter-pod anti-affinity on hostname")
			antiAffinity := &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "hp-gang-anti"},
							},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			}

			hp1 := makePodWithLabels("", "hp-anti-1", pgName, highPriorityName, extendedResourceName, map[string]string{"app": "hp-gang-anti"})
			hp1.Spec.Affinity = antiAffinity
			createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			hp2 := makePodWithLabels("", "hp-anti-2", pgName, highPriorityName, extendedResourceName, map[string]string{"app": "hp-gang-anti"})
			hp2.Spec.Affinity = antiAffinity
			createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying both high-priority gang pods are scheduled and running")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP1.Name, ns), "hp-anti-1 failed to run")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP2.Name, ns), "hp-anti-2 failed to run")

			ginkgo.By("Verifying gang pods were scheduled on distinct nodes satisfying anti-affinity")
			p1Live, err := cs.CoreV1().Pods(ns).Get(ctx, createdHP1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			p2Live, err := cs.CoreV1().Pods(ns).Get(ctx, createdHP2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(p1Live.Spec.NodeName).NotTo(gomega.Equal(p2Live.Spec.NodeName), "Gang pods must be scheduled on distinct nodes to satisfy anti-affinity")

			ginkgo.By("Verifying victims were preempted across the distinct nodes")
			verifyPartialPreempted(ctx, lowPods)
		})

		ginkgo.It("should schedule CompositePodGroup gang pods on distinct nodes and preempt victims satisfying inter-pod anti-affinity", func(ctx context.Context) {
			cs := f.ClientSet
			ns := f.Namespace.Name
			extendedResourceName := v1.ResourceName(extendedResourceDomain + ns)

			lowPriorityName := "low-priority-cpg-anti-" + ns
			highPriorityName := "high-priority-cpg-anti-" + ns

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
			addExtendedResource(ctx, node2, extendedResourceName)
			defer removeExtendedResource(ctx, node2, extendedResourceName)

			ginkgo.By("Creating low-priority pods saturating capacity on both nodes")
			var lowPods []*v1.Pod
			for i := 1; i <= 2; i++ {
				p1 := makePod(node1, fmt.Sprintf("low-cpg-anti-n1-%d", i), "", lowPriorityName, extendedResourceName)
				createdP1, err := cs.CoreV1().Pods(ns).Create(ctx, p1, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP1)

				p2 := makePod(node2, fmt.Sprintf("low-cpg-anti-n2-%d", i), "", lowPriorityName, extendedResourceName)
				createdP2, err := cs.CoreV1().Pods(ns).Create(ctx, p2, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				lowPods = append(lowPods, createdP2)
			}
			for _, p := range lowPods {
				framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns))
			}

			ginkgo.By("Creating high-priority CompositePodGroup with multi-node gang requirements")
			cpgPreemptorName := "cpg-preemptor-anti-" + ns
			cpgPreemptor := makeCompositePodGroup(cpgPreemptorName, highPriorityName, cpgGangPolicy, singleCompositeDisruption)
			createCompositePodGroup(ctx, cpgPreemptor)
			defer deleteCompositePodGroup(ctx, cpgPreemptorName)

			pgPreemptor1Name := "pg-cpg-anti-child1-" + ns
			pgPreemptor1 := makeChildPodGroup(pgPreemptor1Name, highPriorityName, cpgPreemptorName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor1)
			defer deletePodGroup(ctx, pgPreemptor1Name)

			pgPreemptor2Name := "pg-cpg-anti-child2-" + ns
			pgPreemptor2 := makeChildPodGroup(pgPreemptor2Name, highPriorityName, cpgPreemptorName, schedulingv1beta1.PodGroupSchedulingPolicy{
				Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor2)
			defer deletePodGroup(ctx, pgPreemptor2Name)

			antiAffinity := &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchLabels: map[string]string{"app": "cpg-gang-anti"},
							},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			}

			ginkgo.By("Creating high-priority CPG gang pods with strict inter-pod anti-affinity on hostname")
			hp1 := makePodWithLabels("", "hp-cpg-anti-1", pgPreemptor1Name, highPriorityName, extendedResourceName, map[string]string{"app": "cpg-gang-anti"})
			hp1.Spec.Affinity = antiAffinity
			createdHP1, err := cs.CoreV1().Pods(ns).Create(ctx, hp1, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			hp2 := makePodWithLabels("", "hp-cpg-anti-2", pgPreemptor2Name, highPriorityName, extendedResourceName, map[string]string{"app": "cpg-gang-anti"})
			hp2.Spec.Affinity = antiAffinity
			createdHP2, err := cs.CoreV1().Pods(ns).Create(ctx, hp2, metav1.CreateOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Verifying both high-priority CPG gang pods are scheduled and running")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP1.Name, ns), "hp-cpg-anti-1 failed to run")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdHP2.Name, ns), "hp-cpg-anti-2 failed to run")

			ginkgo.By("Verifying gang pods were scheduled on distinct nodes satisfying anti-affinity")
			p1Live, err := cs.CoreV1().Pods(ns).Get(ctx, createdHP1.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			p2Live, err := cs.CoreV1().Pods(ns).Get(ctx, createdHP2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(p1Live.Spec.NodeName).NotTo(gomega.Equal(p2Live.Spec.NodeName), "CPG gang pods must be scheduled on distinct nodes to satisfy anti-affinity")

			ginkgo.By("Verifying victims were preempted across the distinct nodes")
			verifyPartialPreempted(ctx, lowPods)
		})
	})
})
