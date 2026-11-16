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

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// workloadName is the Workload name that all groups in a hierarchy reference.
// The scheduler requires every group in a hierarchy to reference the same WorkloadName.
const workloadName = "workload"

var _ = SIGDescribe("CompositePodGroup", framework.WithFeatureGate(features.CompositePodGroup), func() {
	f := framework.NewDefaultFramework("composite-pod-group")

	var nodes []string

	ginkgo.BeforeEach(func(ctx context.Context) {
		nodes = getNodes(ctx, f)
	})

	newCompositeGangPolicy := func(minGroupCount int32) schedulingv1beta1.CompositePodGroupSchedulingPolicy {
		return schedulingv1beta1.CompositePodGroupSchedulingPolicy{
			Gang: &schedulingv1beta1.CompositeGangSchedulingPolicy{MinGroupCount: minGroupCount},
		}
	}

	newCompositeBasicPolicy := func() schedulingv1beta1.CompositePodGroupSchedulingPolicy {
		return schedulingv1beta1.CompositePodGroupSchedulingPolicy{
			Basic: &schedulingv1beta1.CompositeBasicSchedulingPolicy{},
		}
	}

	// newCompositeTopologyConstraints returns scheduling constraints that keep all pods under a
	// CompositePodGroup within a single domain of the test's topology key.
	newCompositeTopologyConstraints := func() *schedulingv1beta1.CompositePodGroupSchedulingConstraints {
		return &schedulingv1beta1.CompositePodGroupSchedulingConstraints{
			Topology: []schedulingv1beta1.TopologyConstraint{{Key: topologyKey(f)}},
		}
	}

	// createCompositePodGroup creates a CompositePodGroup in the hierarchy (root when parent is nil).
	createCompositePodGroup := func(ctx context.Context, parent *schedulingv1beta1.CompositePodGroup, name string, policy schedulingv1beta1.CompositePodGroupSchedulingPolicy, constraints *schedulingv1beta1.CompositePodGroupSchedulingConstraints, priorityClassName string) *schedulingv1beta1.CompositePodGroup {
		var parentName *string
		if parent != nil {
			parentName = new(parent.Name)
		}
		cpg, err := f.ClientSet.SchedulingV1beta1().CompositePodGroups(f.Namespace.Name).Create(ctx, &schedulingv1beta1.CompositePodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: schedulingv1beta1.CompositePodGroupSpec{
				ParentCompositePodGroupName: parentName,
				WorkloadRef:                 &schedulingv1beta1.WorkloadReference{WorkloadName: workloadName, TemplateName: name},
				SchedulingPolicy:            policy,
				SchedulingConstraints:       constraints,
				PriorityClassName:           priorityClassName,
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create CompositePodGroup %s", name)
		return cpg
	}

	// createChildPodGroup creates a PodGroup given its parent CompositePodGroup.
	createChildPodGroup := func(ctx context.Context, parent *schedulingv1beta1.CompositePodGroup, name string, policy schedulingv1beta1.PodGroupSchedulingPolicy, constraints *schedulingv1beta1.PodGroupSchedulingConstraints) *schedulingv1beta1.PodGroup {
		pg, err := f.ClientSet.SchedulingV1beta1().PodGroups(f.Namespace.Name).Create(ctx, &schedulingv1beta1.PodGroup{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: schedulingv1beta1.PodGroupSpec{
				ParentCompositePodGroupName: new(parent.Name),
				WorkloadRef:                 &schedulingv1beta1.WorkloadReference{WorkloadName: parent.Spec.WorkloadRef.WorkloadName, TemplateName: name},
				SchedulingPolicy:            policy,
				SchedulingConstraints:       constraints,
				PriorityClassName:           parent.Spec.PriorityClassName,
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create PodGroup %s", name)
		return pg
	}

	// hierarchyConfig configures the Workload hierarchy of depth 4 (root -> child -> grandchild -> leaf podgroups).
	type hierarchyConfig struct {
		policy                  schedulingv1beta1.CompositePodGroupSchedulingPolicy
		rootConstraints         *schedulingv1beta1.CompositePodGroupSchedulingConstraints
		leafPodGroupConstraints *schedulingv1beta1.PodGroupSchedulingConstraints
		priorityClassName       string
		leafPodGroupSizes       []int
	}

	createHierarchy := func(ctx context.Context, hierarchyConfig hierarchyConfig) ([]*schedulingv1beta1.CompositePodGroup, []*schedulingv1beta1.PodGroup, []*v1.Pod) {
		parentPolicy := newCompositeGangPolicy(1)
		if hierarchyConfig.policy.Basic != nil {
			parentPolicy = newCompositeBasicPolicy()
		}
		root := createCompositePodGroup(ctx, nil, "root", parentPolicy, hierarchyConfig.rootConstraints, hierarchyConfig.priorityClassName)
		child := createCompositePodGroup(ctx, root, "child", parentPolicy, nil, hierarchyConfig.priorityClassName)
		grandchild := createCompositePodGroup(ctx, child, "grandchild", hierarchyConfig.policy, nil, hierarchyConfig.priorityClassName)

		pgs := make([]*schedulingv1beta1.PodGroup, len(hierarchyConfig.leafPodGroupSizes))
		var pods []*v1.Pod
		for i, size := range hierarchyConfig.leafPodGroupSizes {
			pg := createChildPodGroup(ctx, grandchild, fmt.Sprintf("leaf-%d", i+1), newGangPolicy(int32(size)), hierarchyConfig.leafPodGroupConstraints)
			pgs[i] = pg
			pods = append(pods, createPodGroupPods(ctx, f, pg, size)...)
		}
		return []*schedulingv1beta1.CompositePodGroup{root, child, grandchild}, pgs, pods
	}

	// expectCompositePodGroupCondition waits until the CompositePodGroup's
	// CompositePodGroupInitiallyScheduled condition has the given status and reason.
	// The scheduler sets True only once it can place the group's minimum of child groups:
	// - minGroupCount children for a gang
	// - a single child for a basic CompositePodGroup
	// False/Unschedulable means no pod is bound.
	// Because CompositePodGroupInitiallyScheduled never transitions from True back to False,
	// polling stops early if True is observed when waiting for False.
	expectCompositePodGroupCondition := func(ctx context.Context, cpg *schedulingv1beta1.CompositePodGroup, status metav1.ConditionStatus, reason string) {
		expectCondition(ctx, f,
			framework.GetObject(f.ClientSet.SchedulingV1beta1().CompositePodGroups(f.Namespace.Name).Get, cpg.Name, metav1.GetOptions{}),
			func(cpg *schedulingv1beta1.CompositePodGroup) []metav1.Condition { return cpg.Status.Conditions },
			schedulingv1beta1.CompositePodGroupInitiallyScheduled, status, reason,
		)
	}

	f.It("should schedule a hierarchy", func(ctx context.Context) {
		ginkgo.By("Giving the nodes capacity for all pods")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Creating a hierarchy")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeGangPolicy(2),
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying all pods and all groups of the hierarchy are scheduled")
		expectPodsScheduled(ctx, f, pods)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})

	f.It("should keep a hierarchy pending when not all of its gangs fit", func(ctx context.Context) {
		ginkgo.By("Giving the nodes capacity for all but one pod")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 1)

		ginkgo.By("Creating a hierarchy")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeGangPolicy(2),
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying all groups in the hierarchy are marked Unschedulable and no pods are bound")
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionFalse, schedulingv1beta1.CompositePodGroupReasonUnschedulable)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		}
		expectPodsUnscheduled(ctx, f, pods)

		ginkgo.By("Giving the second node enough capacity for the remaining gang")
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Verifying all pods and all groups of the hierarchy are now scheduled")
		expectPodsScheduled(ctx, f, pods)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})

	f.It("should schedule the child PodGroups of a basic CompositePodGroup independently", func(ctx context.Context) {
		ginkgo.By("Giving the nodes capacity for the smaller leaf gang only")
		setCapacity(ctx, f, nodes[0], 1)
		setCapacity(ctx, f, nodes[1], 1)

		ginkgo.By("Creating a basic CompositePodGroup hierarchy with leaf gangs of size 3 and 2")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeBasicPolicy(),
			leafPodGroupSizes: []int{3, 2},
		})

		ginkgo.By("Verifying the basic hierarchy and the smaller second leaf gang are scheduled while the larger first leaf gang stays pending")
		expectPodsScheduled(ctx, f, pods[3:])
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		expectPodGroupCondition(ctx, f, pgs[0], metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		expectPodGroupCondition(ctx, f, pgs[1], metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		expectPodsUnscheduled(ctx, f, pods[:3])
	})

	f.It("should schedule a CompositePodGroup into a single topology domain", func(ctx context.Context) {
		ginkgo.By("Creating two topology domains, each with room for all pods")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 4)
		setCapacity(ctx, f, nodes[1], 4)

		ginkgo.By("Creating a CompositePodGroup hierarchy that requires a single topology domain")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeGangPolicy(2),
			rootConstraints:   newCompositeTopologyConstraints(),
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying all pods are scheduled in a single topology domain and all groups are scheduled")
		expectPodsScheduledInOneDomain(ctx, f, pods)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})

	f.It("should keep a CompositePodGroup pending if it doesn't fit a single topology domain", func(ctx context.Context) {
		ginkgo.By("Creating two topology domains, each with room for one leaf gang only")
		labelNode(f, nodes[0], "domain-a")
		labelNode(f, nodes[1], "domain-b")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Creating a CompositePodGroup hierarchy that requires a single topology domain")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeGangPolicy(2),
			rootConstraints:   newCompositeTopologyConstraints(),
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying all groups in the hierarchy and all pods stay pending")
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionFalse, schedulingv1beta1.CompositePodGroupReasonUnschedulable)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		}
		expectPodsUnscheduled(ctx, f, pods)

		ginkgo.By("Giving one domain enough capacity to fit the entire hierarchy")
		setCapacity(ctx, f, nodes[0], 4)

		ginkgo.By("Verifying all pods are now scheduled in a single topology domain and all groups are scheduled")
		expectPodsScheduledInOneDomain(ctx, f, pods)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})

	f.It("should schedule a CompositePodGroup with multi-level topology constraints", func(ctx context.Context) {
		ginkgo.By("Placing both nodes in the same zone, with capacity 3 on the first node and 1 on the second")
		labelNode(f, nodes[0], "zone-a")
		labelNode(f, nodes[1], "zone-a")
		setCapacity(ctx, f, nodes[0], 3)
		setCapacity(ctx, f, nodes[1], 1)

		// Although "zone-a" has 4 slots in total, once one leaf gang takes 2 slots on nodes[0],
		// the remaining 2 slots are split across nodes[0] and nodes[1], so the second leaf gang
		// cannot fit on a single node.
		ginkgo.By("Creating a hierarchy where the root requires a single zone and each leaf gang requires a single node")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:          newCompositeGangPolicy(2),
			rootConstraints: newCompositeTopologyConstraints(),
			leafPodGroupConstraints: &schedulingv1beta1.PodGroupSchedulingConstraints{
				Topology: []schedulingv1beta1.TopologyConstraint{{Key: v1.LabelHostname}},
			},
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying all groups in the hierarchy and all pods stay pending")
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionFalse, schedulingv1beta1.CompositePodGroupReasonUnschedulable)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionFalse, schedulingv1beta1.PodGroupReasonUnschedulable)
		}
		expectPodsUnscheduled(ctx, f, pods)

		ginkgo.By("Giving the second node enough capacity for the second leaf gang")
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Verifying all pods are in the same zone, each leaf gang is on a single node, and all groups are scheduled")
		expectPodsScheduledInOneDomain(ctx, f, pods)
		expectPodsScheduledInOneDomain(ctx, f, pods[:2], v1.LabelHostname)
		expectPodsScheduledInOneDomain(ctx, f, pods[2:], v1.LabelHostname)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})

	f.It("should preempt lower-priority pods to schedule a CompositePodGroup", func(ctx context.Context) {
		ns := f.Namespace.Name
		lowPriority, highPriority := ns+"-low", ns+"-high"

		ginkgo.By("Creating a low and a high PriorityClass")
		createPriorityClass(ctx, f, lowPriority, lowPriorityValue)
		createPriorityClass(ctx, f, highPriority, highPriorityValue)

		ginkgo.By("Giving each node capacity for two pods")
		setCapacity(ctx, f, nodes[0], 2)
		setCapacity(ctx, f, nodes[1], 2)

		ginkgo.By("Scheduling one low-priority victim pod on each node, leaving room for one pod per node")
		podClient := e2epod.NewPodClient(f)
		victims := make([]*v1.Pod, 2)
		for i := range victims {
			v := newPod(f, fmt.Sprintf("victim-%d", i))
			v.Spec.PriorityClassName = lowPriority
			v.Spec.NodeSelector = map[string]string{v1.LabelHostname: nodes[i]}
			victims[i] = podClient.Create(ctx, v)
		}
		expectPodsScheduled(ctx, f, victims)

		ginkgo.By("Creating a high-priority CompositePodGroup hierarchy of two leaf gangs")
		cpgs, pgs, pods := createHierarchy(ctx, hierarchyConfig{
			policy:            newCompositeGangPolicy(2),
			priorityClassName: highPriority,
			leafPodGroupSizes: []int{2, 2},
		})

		ginkgo.By("Verifying the victims are preempted and the entire hierarchy is scheduled")
		for _, v := range victims {
			expectPodPreempted(ctx, f, v.Name)
		}
		expectPodsScheduled(ctx, f, pods)
		for _, cpg := range cpgs {
			expectCompositePodGroupCondition(ctx, cpg, metav1.ConditionTrue, schedulingv1beta1.CompositePodGroupReasonScheduled)
		}
		for _, pg := range pgs {
			expectPodGroupCondition(ctx, f, pg, metav1.ConditionTrue, schedulingv1beta1.PodGroupReasonScheduled)
		}
	})
})
