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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

// nodeCount is the number of nodes used by tests, each of which are assigned a domain
// label. We need at least two or more topology domains to exercise placements.
const nodeCount = 2

var _ = SIGDescribe("TopologyAwareScheduling", framework.WithFeatureGate(features.TopologyAwareWorkloadScheduling), func() {
	f := framework.NewDefaultFramework("topology-aware-scheduling")

	var nodes []string

	ginkgo.BeforeEach(func(ctx context.Context) {
		nodes = getNodes(ctx, f)
	})

	f.It("should schedule each PodGroup created from a topology-constrained Workload template into a single domain", func(ctx context.Context) {
		cs, ns := f.ClientSet, f.Namespace.Name

		ginkgo.By("Turning two nodes into topology domains that can host two pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Creating a Workload whose PodGroup template requires a single topology domain")
		const gangSize = 2
		wantConstraints := newTopologyConstraints(f)
		workload, err := cs.SchedulingV1beta1().Workloads(ns).Create(ctx, &schedulingv1beta1.Workload{
			ObjectMeta: metav1.ObjectMeta{Name: "training"},
			Spec: schedulingv1beta1.WorkloadSpec{
				PodGroupTemplates: []schedulingv1beta1.PodGroupTemplate{{
					Name:                  "workers",
					SchedulingPolicy:      newGangPolicy(gangSize),
					SchedulingConstraints: wantConstraints,
				}},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Workload")
		template := workload.Spec.PodGroupTemplates[0]
		gomega.Expect(template.SchedulingConstraints).To(gomega.Equal(wantConstraints), "Workload template lost its scheduling constraints")

		var domains []string
		for _, name := range []string{"workers-0", "workers-1"} {
			ginkgo.By(fmt.Sprintf("Creating PodGroup %s from the Workload template, as a workload controller would", name))
			pg, err := cs.SchedulingV1beta1().PodGroups(ns).Create(ctx, &schedulingv1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{Name: name},
				Spec: schedulingv1beta1.PodGroupSpec{
					WorkloadRef: &schedulingv1beta1.WorkloadReference{
						WorkloadName: workload.Name,
						TemplateName: template.Name,
					},
					SchedulingPolicy:      *template.SchedulingPolicy.DeepCopy(),
					SchedulingConstraints: template.SchedulingConstraints.DeepCopy(),
				},
			}, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create PodGroup %s", name)
			gomega.Expect(pg.Spec.SchedulingConstraints).To(gomega.Equal(wantConstraints), "PodGroup %s lost its scheduling constraints", name)
			pods := createPodGroupPods(ctx, f, pg, gangSize)

			ginkgo.By(fmt.Sprintf("Verifying all pods of PodGroup %s run in a single topology domain", name))
			domains = append(domains, expectPodsInOneDomain(ctx, f, pods))
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}

		ginkgo.By("Verifying the second PodGroup runs in the other domain, the only one with free capacity left")
		gomega.Expect(domains[1]).NotTo(gomega.Equal(domains[0]), "both PodGroups run in %s", domains[0])
	})

	f.It("should keep a gang pending rather than split it across topology domains", func(ctx context.Context) {
		ginkgo.By("Turning two nodes into topology domains that can host two pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Creating a gang of three pods that requires a single topology domain")
		const gangSize = 3
		pg := createPodGroup(ctx, f, "gang", newGangPolicy(gangSize))
		createPodGroupPods(ctx, f, pg, gangSize)

		ginkgo.By("Verifying the gang stays pending although the two domains together have room for four pods")
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
	})

	f.It("should schedule all pods of a basic PodGroup into a single topology domain", func(ctx context.Context) {
		ginkgo.By("Turning two nodes into topology domains that can host three pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 3)
		setCapacity(ctx, f, nodes[1], 3)

		ginkgo.By("Creating a basic PodGroup of three pods that requires a single topology domain")
		pg := createPodGroup(ctx, f, "basic", newBasicPolicy())
		pods := createPodGroupPods(ctx, f, pg, 3)

		ginkgo.By("Verifying all pods of the PodGroup run in a single topology domain")
		expectPodsInOneDomain(ctx, f, pods)
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
	})

	f.It("should keep a gang pending rather than place it on nodes without the topology label", func(ctx context.Context) {
		labeled, unlabeled := nodes[0], nodes[1]

		ginkgo.By("Turning one node into a topology domain that can host one pod, and giving an unlabeled node room for two")
		labelNode(f, labeled, "domain-a")
		setCapacity(ctx, f, labeled, 1)
		setCapacity(ctx, f, unlabeled, 2)

		ginkgo.By("Creating a gang of two pods that requires a single topology domain")
		const gangSize = 2
		pg := createPodGroup(ctx, f, "gang", newGangPolicy(gangSize))
		createPodGroupPods(ctx, f, pg, gangSize)

		ginkgo.By("Verifying the gang stays pending because the node with room is not part of any domain")
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
	})
})

func topologyKey(f *framework.Framework) string {
	return "kubernetes.io/e2e-tas-" + f.Namespace.Name
}

func resourceName(f *framework.Framework) v1.ResourceName {
	return v1.ResourceName(extendedResourceDomain + f.Namespace.Name)
}

// newTopologyConstraints returns scheduling constraints that keep all pods
// of a PodGroup within a single domain of the test's topology key.
func newTopologyConstraints(f *framework.Framework) *schedulingv1beta1.PodGroupSchedulingConstraints {
	return &schedulingv1beta1.PodGroupSchedulingConstraints{
		Topology: []schedulingv1beta1.TopologyConstraint{{Key: topologyKey(f)}},
	}
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

// createPodGroup creates a PodGroup whose pods must all run in one topology domain.
func createPodGroup(ctx context.Context, f *framework.Framework, name string, policy schedulingv1beta1.PodGroupSchedulingPolicy) *schedulingv1beta1.PodGroup {
	pg, err := f.ClientSet.SchedulingV1beta1().PodGroups(f.Namespace.Name).Create(ctx, &schedulingv1beta1.PodGroup{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: schedulingv1beta1.PodGroupSpec{
			SchedulingPolicy:      policy,
			SchedulingConstraints: newTopologyConstraints(f),
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create PodGroup %s", name)
	return pg
}

// createPodGroupPods creates the given number of pods that belong to the PodGroup.
func createPodGroupPods(ctx context.Context, f *framework.Framework, pg *schedulingv1beta1.PodGroup, count int) []*v1.Pod {
	slot := v1.ResourceList{resourceName(f): resource.MustParse("1")}
	pods := make([]*v1.Pod, 0, count)
	for i := range count {
		pod := e2epod.MakePod(f.Namespace.Name, nil, nil, admissionapi.LevelRestricted, "")
		pod.GenerateName = ""
		pod.Name = fmt.Sprintf("%s-%d", pg.Name, i)
		pod.Spec.SchedulingGroup = &v1.PodSchedulingGroup{PodGroupName: ptr.To(pg.Name)}
		pod.Spec.Containers[0].Resources.Requests = slot
		pod.Spec.Containers[0].Resources.Limits = slot

		created, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod %s", pod.Name)
		pods = append(pods, created)
	}
	return pods
}

// getNodes returns names of ready schedulable nodes, we need at least nodeCount nodes.
func getNodes(ctx context.Context, f *framework.Framework) []string {
	nodeList, err := e2enode.GetBoundedReadySchedulableNodes(ctx, f.ClientSet, nodeCount)
	framework.ExpectNoError(err, "failed to list ready schedulable nodes")
	e2eskipper.SkipUnlessAtLeast(len(nodeList.Items), nodeCount,
		fmt.Sprintf("Topology-aware scheduling tests need %d ready schedulable nodes, found %d", nodeCount, len(nodeList.Items)))
	nodes := make([]string, len(nodeList.Items))
	for i, node := range nodeList.Items {
		nodes[i] = node.Name
	}
	return nodes
}

// labelNode puts the node in the given topology domain by labeling it with topology key.
// A node without label belongs to no domain, won't schedule any topology-constrained PodGroup.
func labelNode(f *framework.Framework, node, domain string) {
	e2enode.AddOrUpdateLabelOnNode(f.ClientSet, node, topologyKey(f), domain)
	ginkgo.DeferCleanup(e2enode.RemoveLabelOffNode, f.ClientSet, node, topologyKey(f))
}

// setCapacity sets how many of the test's pods the node can host until the test ends.
func setCapacity(ctx context.Context, f *framework.Framework, node string, pods int64) {
	e2enode.AddExtendedResource(ctx, f.ClientSet, node, resourceName(f), *resource.NewQuantity(pods, resource.DecimalSI))
	ginkgo.DeferCleanup(e2enode.RemoveExtendedResource, f.ClientSet, node, resourceName(f))
}

// expectPodsInOneDomain waits for the pods to run and returns the topology domain they share.
func expectPodsInOneDomain(ctx context.Context, f *framework.Framework, pods []*v1.Pod) string {
	cs, ns, key := f.ClientSet, f.Namespace.Name, topologyKey(f)
	domains := sets.New[string]()
	for _, p := range pods {
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, cs, p.Name, ns), "pod %s failed to run", p.Name)
		pod, err := cs.CoreV1().Pods(ns).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod %s", p.Name)
		node, err := cs.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get node %s", pod.Spec.NodeName)
		domain, ok := node.Labels[key]
		if !ok {
			framework.Failf("Pod %s runs on node %s, which is not in any topology domain of %s", pod.Name, node.Name, key)
		}
		domains.Insert(domain)
	}
	if domains.Len() != 1 {
		framework.Failf("Pods run in topology domains %v, want a single domain", sets.List(domains))
	}
	return sets.List(domains)[0]
}

// expectPodGroupCondition waits until the PodGroup's PodGroupInitiallyScheduled condition has
// the given status and reason.
// The scheduler binds pods and sets True only once it can place the group's minimum:
// - minCount pods for a gang
// - a single pod for a basic PodGroup
// False/Unschedulable means no pod is bound.
func expectPodGroupCondition(ctx context.Context, f *framework.Framework, pg *schedulingv1beta1.PodGroup, status metav1.ConditionStatus, reason string) {
	cs, ns := f.ClientSet, f.Namespace.Name
	gomega.Eventually(ctx, func(ctx context.Context) error {
		current, err := cs.SchedulingV1beta1().PodGroups(ns).Get(ctx, pg.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		cond := apimeta.FindStatusCondition(current.Status.Conditions, schedulingv1beta1.PodGroupInitiallyScheduled)
		if cond == nil {
			return fmt.Errorf("PodGroup %s has no %s condition yet", pg.Name, schedulingv1beta1.PodGroupInitiallyScheduled)
		}
		if cond.Status == status && cond.Reason == reason {
			return nil
		}
		err = fmt.Errorf("PodGroup %s condition %s is %s with reason %q (%s), want %s with reason %q",
			pg.Name, cond.Type, cond.Status, cond.Reason, cond.Message, status, reason)
		if cond.Status == metav1.ConditionTrue {
			return gomega.StopTrying(err.Error())
		}
		return err
	}).WithTimeout(f.Timeouts.PodStart).WithPolling(framework.Poll).Should(gomega.Succeed())
}
