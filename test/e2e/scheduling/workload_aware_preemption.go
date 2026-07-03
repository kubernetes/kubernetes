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
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const extendedResourceName = "example.com/combined-resource"

var (
	gangPolicy = schedulingv1alpha3.PodGroupSchedulingPolicy{
		Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 2},
	}
	singleDisruption = schedulingv1alpha3.DisruptionMode{
		Single: &schedulingv1alpha3.SingleDisruptionMode{},
	}
	allDisruption = schedulingv1alpha3.DisruptionMode{
		All: &schedulingv1alpha3.AllDisruptionMode{},
	}
)

type preemptorType int

const (
	pod = iota
	podGroup
)

var _ = SIGDescribe("WorkloadAwarePreemption", framework.WithFeatureGate(features.GenericWorkload), func() {
	f := framework.NewDefaultFramework("workload-aware-preemption")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	createPodGroup := func(ctx context.Context, pg *schedulingv1alpha3.PodGroup) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		_, err := cs.SchedulingV1alpha3().PodGroups(ns).Create(ctx, pg, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup %s", pg.Name)
	}

	deletePodGroup := func(ctx context.Context, name string) {
		cs := f.ClientSet
		ns := f.Namespace.Name
		ginkgo.By("Deleting PodGroup")
		err := cs.SchedulingV1alpha3().PodGroups(ns).Delete(ctx, name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete PodGroup")
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
		framework.ExpectNoError(err, "failed to delete priority class: %s", name)
	}

	addExtendedResource := func(ctx context.Context, nodeName string) {
		cs := f.ClientSet
		resName := v1.ResourceName(extendedResourceName)
		e2enode.AddExtendedResource(ctx, cs, nodeName, resName, resource.MustParse("2"))
	}

	removeExtendedResource := func(ctx context.Context, nodeName string) {
		cs := f.ClientSet
		resName := v1.ResourceName(extendedResourceName)
		ginkgo.By("Removing extended resource from node")
		e2enode.RemoveExtendedResource(ctx, cs, nodeName, resName)
	}

	makePod := func(nodeName string, name, pgName string, priority string) *v1.Pod {
		ns := f.Namespace.Name
		p := e2epod.MakePod(ns, map[string]string{"kubernetes.io/hostname": nodeName}, nil, admissionapi.LevelPrivileged, "")
		p.ObjectMeta.GenerateName = name + "-"
		p.Spec.PriorityClassName = priority
		if pgName != "" {
			p.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: &pgName}
		}
		p.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceName(extendedResourceName): resource.MustParse("1")}
		p.Spec.Containers[0].Resources.Limits = v1.ResourceList{v1.ResourceName(extendedResourceName): resource.MustParse("1")}
		return p
	}

	makePodGroup := func(pgName string, priorityName string, schedulingPolicy schedulingv1alpha3.PodGroupSchedulingPolicy, disruptionMode schedulingv1alpha3.DisruptionMode) *schedulingv1alpha3.PodGroup {
		ns := f.Namespace.Name

		return &schedulingv1alpha3.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: pgName, Namespace: ns},
			Spec: schedulingv1alpha3.PodGroupSpec{
				PriorityClassName: priorityName,
				SchedulingPolicy:  schedulingPolicy,
				DisruptionMode:    &disruptionMode,
			},
		}
	}

	getNodeName := func(ctx context.Context) string {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get a ready schedulable node")
		return node.Name
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
		ginkgo.By("Verifying all pods in PG-victim are preempted")
		gomega.Eventually(ctx, func(ctx context.Context) error {
			for _, p := range pods {
				if !isPodPreempted(ctx, p.Name) {
					return fmt.Errorf("pod %s is not preempted yet", p.Name)
				}
			}
			return nil
		}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Succeed(), "All pods in PG-victim should eventually be preempted")
	}

	verifyPartialPreempted := func(ctx context.Context, pods []*v1.Pod) {
		ginkgo.By("Verifying at least one pod in PG-victim is preempted and one remains running")
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
		}).WithTimeout(30*time.Second).WithPolling(1*time.Second).Should(gomega.Succeed(), "Expected at least one pod from PG-victim to be preempted")

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
		victimDisruptionMode schedulingv1alpha3.DisruptionMode
		verify               func(context.Context, []*v1.Pod)
	}

	runPreemptionTest := func(ctx context.Context, args preemptionTestArgs) {
		cs := f.ClientSet
		ns := f.Namespace.Name

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
		addExtendedResource(ctx, nodeName)
		defer removeExtendedResource(ctx, nodeName)

		ginkgo.By(fmt.Sprintf("Creating low-priority pod group PG-victim with disruptionMode %v", args.victimDisruptionMode))
		pgVictimName := "pg-victim-" + ns
		pgVictim := makePodGroup(pgVictimName, lowPriorityName, gangPolicy, args.victimDisruptionMode)
		createPodGroup(ctx, pgVictim)
		defer deletePodGroup(ctx, pgVictim.Name)

		var pods []*v1.Pod
		for i := range gangPolicy.Gang.MinCount {
			name := fmt.Sprintf("victim%d", i+1)
			ginkgo.By(fmt.Sprintf("Creating low-priority pod %s belonging to PG-victim", name))
			p := makePod(nodeName, name, pgVictimName, lowPriorityName)
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
			pgPreemptor := makePodGroup(pgPreemptorName, highPriorityName, schedulingv1alpha3.PodGroupSchedulingPolicy{
				Gang: &schedulingv1alpha3.GangSchedulingPolicy{MinCount: 1},
			}, singleDisruption)
			createPodGroup(ctx, pgPreemptor)
			defer deletePodGroup(ctx, pgPreemptor.Name)
		}

		ginkgo.By("Creating high-priority individual pod hp1")
		hp1 := makePod(nodeName, "hp1", pgPreemptorName, highPriorityName)
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
})
