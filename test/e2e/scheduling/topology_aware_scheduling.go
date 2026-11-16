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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var _ = SIGDescribe("TopologyAwareScheduling", framework.WithFeatureGate(features.TopologyAwareWorkloadScheduling), func() {
	f := framework.NewDefaultFramework("topology-aware-scheduling")

	var nodes []string

	ginkgo.BeforeEach(func(ctx context.Context) {
		nodes = getNodes(ctx, f)
	})

	// newTopologyConstraints returns scheduling constraints that keep all pods
	// of a PodGroup within a single domain of the test's topology key.
	newTopologyConstraints := func() *schedulingv1beta1.PodGroupSchedulingConstraints {
		return &schedulingv1beta1.PodGroupSchedulingConstraints{
			Topology: []schedulingv1beta1.TopologyConstraint{{Key: topologyKey(f)}},
		}
	}

	// createPodGroup creates a PodGroup whose pods must all run in one topology domain.
	createPodGroup := func(ctx context.Context, name string, policy schedulingv1beta1.PodGroupSchedulingPolicy, priorityClassName string) *schedulingv1beta1.PodGroup {
		pg, err := f.ClientSet.SchedulingV1beta1().PodGroups(f.Namespace.Name).Create(ctx, &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: schedulingv1beta1.PodGroupSpec{
				SchedulingPolicy:      policy,
				SchedulingConstraints: newTopologyConstraints(),
				PriorityClassName:     priorityClassName,
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup %s", name)
		return pg
	}

	f.It("should schedule each PodGroup into a single topology domain", func(ctx context.Context) {
		ginkgo.By("Turning two nodes into topology domains that can host two pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		const gangSize = 2
		var domains []string
		for _, name := range []string{"workers-0", "workers-1"} {
			ginkgo.By(fmt.Sprintf("Creating PodGroup %s that requires a single topology domain", name))
			pg := createPodGroup(ctx, name, newGangPolicy(gangSize), "")
			pods := createPodGroupPods(ctx, f, pg, gangSize)

			ginkgo.By(fmt.Sprintf("Verifying all pods of PodGroup %s are scheduled in a single topology domain", name))
			domains = append(domains, expectPodsScheduledInOneDomain(ctx, f, pods))
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}

		ginkgo.By("Verifying the second PodGroup is scheduled in the other domain, the only one with free capacity left")
		gomega.Expect(domains[1]).NotTo(gomega.Equal(domains[0]), "both PodGroups are scheduled in %s", domains[0])
	})

	f.It("should keep a gang pending rather than split it across topology domains", func(ctx context.Context) {
		ginkgo.By("Turning two nodes into topology domains that can host two pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Creating a gang of three pods that requires a single topology domain")
		const gangSize = 3
		pg := createPodGroup(ctx, "gang", newGangPolicy(gangSize), "")
		pods := createPodGroupPods(ctx, f, pg, gangSize)

		ginkgo.By("Verifying the gang stays pending although the two domains together have room for four pods")
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		expectPodsUnscheduled(ctx, f, pods)

		ginkgo.By("Giving one domain enough capacity to fit the entire gang")
		setCapacity(ctx, f, nodes[0], gangSize)

		ginkgo.By("Verifying the gang is now scheduled in a single topology domain")
		expectPodsScheduledInOneDomain(ctx, f, pods)
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
	})

	f.It("should schedule all pods of a basic PodGroup into a single topology domain", func(ctx context.Context) {
		ginkgo.By("Turning two nodes into topology domains that can host three pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 3)
		setCapacity(ctx, f, nodes[1], 3)

		ginkgo.By("Creating a basic PodGroup of three pods that requires a single topology domain")
		pg := createPodGroup(ctx, "basic", newBasicPolicy(), "")
		pods := createPodGroupPods(ctx, f, pg, 3)

		ginkgo.By("Verifying all pods of the PodGroup are scheduled in a single topology domain")
		expectPodsScheduledInOneDomain(ctx, f, pods)
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
	})

	f.It("should preempt lower-priority pods in only one topology domain to keep a gang together", func(ctx context.Context) {
		ns := f.Namespace.Name
		lowPriority, highPriority := ns+"-low", ns+"-high"

		ginkgo.By("Creating a low and a high PriorityClass")
		createPriorityClass(ctx, f, lowPriority, lowPriorityValue)
		createPriorityClass(ctx, f, highPriority, highPriorityValue)

		ginkgo.By("Turning two nodes into topology domains that can host two pods each")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		// Without the topology constraint the gang would fit one pod per domain and
		// nothing would be preempted.
		ginkgo.By("Scheduling one low-priority pod in each domain, leaving room for one more pod per domain")
		podClient := e2epod.NewPodClient(f)
		victims := make(map[string]*v1.Pod, 2)
		for _, domain := range []string{"domain-a", "domain-b"} {
			pod := newPod(f, "victim-"+domain)
			pod.Spec.PriorityClassName = lowPriority
			pod.Spec.NodeSelector = map[string]string{topologyKey(f): domain}
			victims[domain] = podClient.Create(ctx, pod)
		}
		expectPodsScheduled(ctx, f, []*v1.Pod{victims["domain-a"], victims["domain-b"]})

		ginkgo.By("Creating a high-priority gang of two pods that requires a single topology domain")
		const gangSize = 2
		pg := createPodGroup(ctx, "gang", newGangPolicy(gangSize), highPriority)
		pods := createPodGroupPods(ctx, f, pg, gangSize)

		ginkgo.By("Verifying the gang is scheduled in a single topology domain")
		domain := expectPodsScheduledInOneDomain(ctx, f, pods)
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		otherDomain := "domain-a"
		if domain == otherDomain {
			otherDomain = "domain-b"
		}

		ginkgo.By(fmt.Sprintf("Verifying the low-priority pod in %s is preempted while the one in %s is not", domain, otherDomain))
		expectPodPreempted(ctx, f, victims[domain].Name)
		expectPodNotPreempted(ctx, f, victims[otherDomain].Name)
	})

	f.It("should keep a gang pending rather than place it on nodes without the topology label", func(ctx context.Context) {
		labeled, unlabeled := nodes[0], nodes[1]

		ginkgo.By("Turning one node into a topology domain that can host one pod, and giving an unlabeled node room for two")
		labelNode(f, labeled, "domain-a")
		setCapacity(ctx, f, labeled, 1)
		setCapacity(ctx, f, unlabeled, 2)

		ginkgo.By("Creating a gang of two pods that requires a single topology domain")
		const gangSize = 2
		pg := createPodGroup(ctx, "gang", newGangPolicy(gangSize), "")
		pods := createPodGroupPods(ctx, f, pg, gangSize)

		ginkgo.By("Verifying the gang stays pending because the node with room is not part of any domain")
		expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		expectPodsUnscheduled(ctx, f, pods)
	})
})
