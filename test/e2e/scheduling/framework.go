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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// nodeCount is the number of nodes used by tests.
	// In Topology Aware scheduling tests, each node is assigned a domain label.
	// We need at least two or more topology domains to exercise placements.
	nodeCount = 2

	lowPriorityValue  int32 = 100
	highPriorityValue int32 = 1000
)

var (
	timeout  = 10 * time.Minute
	waitTime = 2 * time.Second

	// SIGDescribe annotates the test with the SIG label.
	SIGDescribe = framework.SIGDescribe("scheduling")
)

// WaitForStableCluster waits until all existing pods are scheduled and returns their amount.
func WaitForStableCluster(c clientset.Interface, workerNodes sets.Set[string]) int {
	startTime := time.Now()
	// Wait for all pods to be scheduled.
	allScheduledPods, allNotScheduledPods := getScheduledAndUnscheduledPods(c, workerNodes)
	for len(allNotScheduledPods) != 0 {
		time.Sleep(waitTime)
		if startTime.Add(timeout).Before(time.Now()) {
			framework.Logf("Timed out waiting for the following pods to schedule")
			for _, p := range allNotScheduledPods {
				framework.Logf("%v/%v", p.Namespace, p.Name)
			}
			framework.Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
		allScheduledPods, allNotScheduledPods = getScheduledAndUnscheduledPods(c, workerNodes)
	}
	return len(allScheduledPods)
}

// getScheduledAndUnscheduledPods lists scheduled and not scheduled pods in all namespaces, with succeeded and failed pods filtered out.
func getScheduledAndUnscheduledPods(c clientset.Interface, workerNodes sets.Set[string]) (scheduledPods, notScheduledPods []v1.Pod) {
	pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("listing all pods in namespace %q while waiting for stable cluster", metav1.NamespaceAll))

	// API server returns also Pods that succeeded. We need to filter them out.
	filteredPods := make([]v1.Pod, 0, len(pods.Items))
	for _, p := range pods.Items {
		if !podTerminated(p) {
			filteredPods = append(filteredPods, p)
		}
	}
	pods.Items = filteredPods
	return GetPodsScheduled(workerNodes, pods)
}

func podTerminated(p v1.Pod) bool {
	return p.Status.Phase == v1.PodSucceeded || p.Status.Phase == v1.PodFailed
}

func topologyKey(f *framework.Framework) string {
	return "kubernetes.io/e2e-tas-" + f.Namespace.Name
}

func resourceName(f *framework.Framework) v1.ResourceName {
	return v1.ResourceName(extendedResourceDomain + f.Namespace.Name)
}

func newGangPolicy(minCount int32) schedulingv1beta1.PodGroupSchedulingPolicy {
	return schedulingv1beta1.PodGroupSchedulingPolicy{
		Gang: &schedulingv1beta1.GangSchedulingPolicy{MinCount: minCount},
	}
}

func newBasicPolicy() schedulingv1beta1.PodGroupSchedulingPolicy {
	return schedulingv1beta1.PodGroupSchedulingPolicy{
		Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
	}
}

// newPod creates a pod object with resource requests of 1 unit.
func newPod(f *framework.Framework, name string) *v1.Pod {
	slot := v1.ResourceList{resourceName(f): resource.MustParse("1")}
	pod := e2epod.MakePod(f.Namespace.Name, nil, nil, admissionapi.LevelRestricted, "")
	pod.GenerateName = ""
	pod.Name = name
	pod.Spec.Containers[0].Resources.Requests = slot
	pod.Spec.Containers[0].Resources.Limits = slot
	return pod
}

// createPodGroupPods creates the given number of pods that belong to the PodGroup.
func createPodGroupPods(ctx context.Context, f *framework.Framework, pg *schedulingv1beta1.PodGroup, count int) []*v1.Pod {
	podClient := e2epod.NewPodClient(f)
	pods := make([]*v1.Pod, 0, count)
	for i := range count {
		pod := newPod(f, fmt.Sprintf("%s-%d", pg.Name, i))
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: new(pg.Name)}
		pod.Spec.PriorityClassName = pg.Spec.PriorityClassName

		pods = append(pods, podClient.Create(ctx, pod))
	}
	return pods
}

// createPriorityClass creates a PriorityClass and waits until priority admission resolves it.
func createPriorityClass(ctx context.Context, f *framework.Framework, name string, value int32) {
	cs := f.ClientSet
	_, err := cs.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Value:      value,
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create PriorityClass %s", name)
	ginkgo.DeferCleanup(framework.IgnoreNotFound(cs.SchedulingV1().PriorityClasses().Delete), name, metav1.DeleteOptions{})

	probe := &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: "priority-class-probe"},
		Spec: schedulingv1beta1.PodGroupSpec{
			SchedulingPolicy:  newBasicPolicy(),
			PriorityClassName: name,
		},
	}
	gomega.Eventually(ctx, func(ctx context.Context) error {
		_, err := cs.SchedulingV1beta1().PodGroups(f.Namespace.Name).Create(ctx, probe, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
		return err
	}).WithTimeout(f.Timeouts.PodStart).WithPolling(framework.Poll).Should(gomega.Succeed(), "priority admission doesn't resolve PriorityClass %s", name)
}

// getNodes returns names of ready schedulable nodes, skipping the test if fewer than nodeCount exist.
func getNodes(ctx context.Context, f *framework.Framework) []string {
	nodeList, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, nodeCount)
	framework.ExpectNoError(err, "failed to list ready schedulable nodes")
	e2eskipper.SkipUnlessAtLeast(len(nodeList.Items), nodeCount,
		fmt.Sprintf("Need %d ready schedulable nodes, found %d", nodeCount, len(nodeList.Items)))
	nodes := make([]string, len(nodeList.Items))
	for i, node := range nodeList.Items {
		nodes[i] = node.Name
	}
	return nodes
}

// labelNode puts the node in the given topology domain by labeling it with topology key.
// Because topologyKey(f) includes the unique per-test f.Namespace.Name, concurrent tests
// do not collide when mutating cluster-scoped Node objects.
// A node without label belongs to no domain, won't schedule any topology-constrained group.
func labelNode(f *framework.Framework, node, domain string) {
	e2enode.AddOrUpdateLabelOnNode(f.ClientSet, node, topologyKey(f), domain)
	ginkgo.DeferCleanup(e2enode.RemoveLabelOffNode, f.ClientSet, node, topologyKey(f))
}

// setCapacity sets how many of the test's pods the node can host until the test ends.
// Because resourceName(f) includes the unique per-test f.Namespace.Name, concurrent tests
// do not collide when mutating cluster-scoped Node objects.
func setCapacity(ctx context.Context, f *framework.Framework, node string, pods int64) {
	e2enode.AddExtendedResource(ctx, f.ClientSet, node, resourceName(f), *resource.NewQuantity(pods, resource.DecimalSI))
	ginkgo.DeferCleanup(e2enode.RemoveExtendedResource, f.ClientSet, node, resourceName(f))
}

func expectPodsScheduled(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	for _, p := range pods {
		framework.ExpectNoError(e2epod.WaitForPodScheduled(ctx, f.ClientSet, f.Namespace.Name, p.Name), "pod %s failed to schedule", p.Name)
	}
}

// expectPodsScheduledInOneDomain waits for the pods to be scheduled and checks whether they share a topology domain.
// It checks the topologyKey(f) domain by default, or customTopologyKey if provided.
func expectPodsScheduledInOneDomain(ctx context.Context, f *framework.Framework, pods []*v1.Pod, customTopologyKey ...string) string {
	expectPodsScheduled(ctx, f, pods)
	cs, ns, key := f.ClientSet, f.Namespace.Name, topologyKey(f)
	if len(customTopologyKey) > 0 {
		key = customTopologyKey[0]
	}
	domains := sets.New[string]()
	for _, p := range pods {
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %s", p.Name)
		node, err := cs.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get node %s", pod.Spec.NodeName)
		gomega.Expect(node.Labels).To(gomega.HaveKey(key), "Pod %s is scheduled on node %s, which is not in any topology domain of %s", pod.Name, node.Name, key)
		domains.Insert(node.Labels[key])
	}
	gomega.Expect(domains).To(gomega.HaveLen(1), "Pods are scheduled in topology domains %v, want a single domain", sets.List(domains))
	return sets.List(domains)[0]
}

// expectPodsUnscheduled waits for each pod to be marked Unschedulable.
func expectPodsUnscheduled(ctx context.Context, f *framework.Framework, pods []*v1.Pod) {
	for _, p := range pods {
		framework.ExpectNoError(e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, p.Name, f.Namespace.Name), "pod %s was not marked unschedulable", p.Name)
	}
}

// expectPodPreempted waits until the pod is deleted or has a deletion timestamp set.
func expectPodPreempted(ctx context.Context, f *framework.Framework, podName string) {
	gomega.Eventually(ctx, func(ctx context.Context) error {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if pod.DeletionTimestamp == nil {
			return fmt.Errorf("pod %s is not being deleted", pod.Name)
		}
		return nil
	}).WithTimeout(f.Timeouts.PodStart).WithPolling(framework.Poll).Should(gomega.Succeed())
}

// expectPodNotPreempted verifies that a scheduled pod is not preempted over a short window.
func expectPodNotPreempted(ctx context.Context, f *framework.Framework, podName string) {
	gomega.Consistently(ctx, func(ctx context.Context) error {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if pod.DeletionTimestamp != nil || pod.Spec.NodeName == "" {
			return fmt.Errorf("pod %s has node %q and deletionTimestamp %v", pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
		}
		return nil
	}).WithTimeout(5 * time.Second).WithPolling(framework.Poll).Should(gomega.Succeed())
}

// expectPodGroupCondition waits until the PodGroup's PodGroupInitiallyScheduled condition has
// the given status and reason.
// The scheduler binds pods and sets True only once it can place the group's minimum:
// - minCount pods for a gang
// - a single pod for a basic PodGroup
// False/Unschedulable means no pod is bound.
// Because PodGroupInitiallyScheduled never transitions from True back to False,
// polling stops early if True is observed when waiting for False.
func expectPodGroupCondition(ctx context.Context, f *framework.Framework, pg *schedulingv1beta1.PodGroup, status metav1.ConditionStatus, reason string) {
	expectCondition(ctx, f,
		framework.GetObject(f.ClientSet.SchedulingV1beta1().PodGroups(f.Namespace.Name).Get, pg.Name, metav1.GetOptions{}),
		func(pg *schedulingv1beta1.PodGroup) []metav1.Condition { return pg.Status.Conditions },
		schedulingv1beta1.PodGroupInitiallyScheduled, status, reason,
	)
}

// expectCondition waits until the condition of the given type on the object returned by get
// has the given status and reason.
func expectCondition[T metav1.Object](ctx context.Context, f *framework.Framework, get framework.GetFunc[T], getConditions func(T) []metav1.Condition, conditionType string, status metav1.ConditionStatus, reason string) {
	gomega.Eventually(ctx, func(ctx context.Context) error {
		obj, err := get(ctx)
		if err != nil {
			return err
		}
		cond := apimeta.FindStatusCondition(getConditions(obj), conditionType)
		if cond == nil {
			return fmt.Errorf("%T %s has no %s condition yet", obj, obj.GetName(), conditionType)
		}
		if cond.Status == status && cond.Reason == reason {
			return nil
		}
		err = fmt.Errorf("%T %s condition %s is %s with reason %q (%s), want %s with reason %q",
			obj, obj.GetName(), cond.Type, cond.Status, cond.Reason, cond.Message, status, reason)
		// The scheduler never sets these conditions from True back to False, polling can stop early.
		if cond.Status == metav1.ConditionTrue {
			return gomega.StopTrying(err.Error())
		}
		return err
	}).WithTimeout(f.Timeouts.PodStart).WithPolling(framework.Poll).Should(gomega.Succeed())
}
