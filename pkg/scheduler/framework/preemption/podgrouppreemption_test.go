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

package preemption

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type mockPodGroupLister struct {
	podGroups map[string]*schedulingapi.PodGroup
}

func (m *mockPodGroupLister) Get(namespace, name string) (*schedulingapi.PodGroup, error) {
	if pg, ok := m.podGroups[name]; ok {
		return pg, nil
	}
	return nil, fmt.Errorf("pod group %s not found", name)
}

func makePodGroupPreemptor(pg *schedulingapi.PodGroup, pods []*v1.Pod) *podGroupPreemptor {
	return makePodGroupPreemptorWithPreemptionPolicy(pg, pods, schedulingapi.PreemptLowerPriority)
}

func makePodGroupPreemptorWithPreemptionPolicy(pg *schedulingapi.PodGroup, pods []*v1.Pod, policy schedulingapi.PreemptionPolicy) *podGroupPreemptor {
	return &podGroupPreemptor{
		priority:         util.PodGroupPriority(pg),
		pods:             pods,
		podGroup:         pg,
		preemptionPolicy: policy,
	}
}

func TestPodGroupEvaluator_SelectVictimsOnDomain(t *testing.T) {
	// blockingRule mocks complex scheduling constraints for tests.
	// It states: "Node 'nodeName' can host 'capacity' preempting pods,
	// but ONLY IF the pods in 'blockingVictims' are removed first."
	type blockingRule struct {
		nodeName        string
		capacity        int              // Available slots for preemptor once victims are removed.
		blockingVictims sets.Set[string] // Pods on the node preventing capacity usage.
	}

	tests := []struct {
		name                     string
		nodeNames                []string
		initPods                 []*v1.Pod
		initPodGroups            []*schedulingapi.PodGroup
		preemptor                *podGroupPreemptor
		pdbs                     []*policy.PodDisruptionBudget
		blockingRules            []blockingRule
		customMockSchedulingFunc func(ctx context.Context, domainNodes []fwk.NodeInfo) (*fwk.PodGroupAssignments, *fwk.Status)
		expectedPods             []string
		expectedStatus           *fwk.Status
	}{
		{
			name:      "Mix of no-group and single-pod-groups",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"}, // p1 is less important than p2 because it's not part of a pod group
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Priority: Shared group vs no group",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(0, 0)).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"}, // p1 is less important than p2 because of later StartTime
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Shared Group: Preempt separately",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(0, 0)).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
			},
			expectedPods:   []string{"p1"}, // p1 is less important than p2
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Complex Mixed: Shared, different, and no groups",
			nodeNames: []string{"node1", "node2", "node3", "node4", "node5"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(2, 0)).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").StartTime(metav1.Unix(1, 0)).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").StartTime(metav1.Unix(0, 0)).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(midPriority).Obj(),
				st.MakePod().Name("p5").UID("v5").Node("node5").Priority(highPriority).PodGroupName("pg3").StartTime(metav1.Unix(0, 0)).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg3").UID("pg3").DisruptionModeSingle().Priority(highPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj(),
					st.MakePod().Name("p-2").UID("p-2").Priority(highPriority).Obj(),
					st.MakePod().Name("p-3").UID("p-3").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("p4")},
				{nodeName: "node5", capacity: 1, blockingVictims: sets.New("p5")},
			},
			expectedPods:   []string{"p1", "p2", "p3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PDB: Mixed groups",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-pdb").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("victim-no-pdb").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("victim-pdb")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("victim-no-pdb")},
			},
			expectedPods:   []string{"victim-no-pdb"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptLowerPriority preemption policy performs preemption",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj()},
				schedulingapi.PreemptLowerPriority,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptLowerPriority preemption policy performs preemption, ignoring member pod preemption policy",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).PreemptionPolicy(schedulingapi.PreemptLowerPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).PreemptionPolicy(v1.PreemptNever).Obj(),
				},
				schedulingapi.PreemptLowerPriority,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptNever policy does not perform preemption",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj(),
					st.MakePod().Name("p-2").UID("p-2").Priority(highPriority).Obj(),
				},
				schedulingapi.PreemptNever,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.Unschedulable, "not eligible due to preemptionPolicy=Never."),
		},
		{
			name:      "Preemptor group is not eligible if any member has nominated node with terminating pods",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(lowPriority).Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules:  []blockingRule{},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.Unschedulable, "not eligible due to a terminating pod on the nominated node."),
		},
		{
			name:      "Preemptor group is eligible if terminating pods are on non-nominated nodes",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node2").Priority(lowPriority).Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
				st.MakePod().Name("other-victim").UID("v2").Node("node1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("other-victim"), capacity: 1},
			},
			expectedPods:   []string{"other-victim"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Preemptor group is not eligible if nominated node has terminating pod belonging to a pod group of lower priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(highPriority).PodGroupName("victim-pg").Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Priority(lowPriority).DisruptionModeAll().Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules:  []blockingRule{},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.Unschedulable, "not eligible due to a terminating pod on the nominated node."),
		},
		{
			name:      "Preemptor group is eligible if nominated node has terminating pod belonging to a pod group of higher priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("victim-pg").Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
				st.MakePod().Name("other-victim").UID("v2").Node("node1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Priority(highPriority).DisruptionModeAll().Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(midPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(midPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(midPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("other-victim"), capacity: 1},
			},
			expectedPods:   []string{"other-victim"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Preemptor group is not eligible if nominated node has terminating pod belonging to a pod group of lower priority with DisruptionModePod",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(highPriority).PodGroupName("victim-pg").Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Priority(lowPriority).DisruptionModeSingle().Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules:  []blockingRule{},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.Unschedulable, "not eligible due to a terminating pod on the nominated node."),
		},
		{
			name:      "Preemptor group is eligible if nominated node has terminating pod belonging to a pod group of higher priority with nil DisruptionMode",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("victim-pg").Condition(v1.DisruptionTarget, v1.ConditionTrue, v1.PodReasonPreemptionByScheduler).Terminating().Obj(),
				st.MakePod().Name("other-victim").UID("v2").Node("node1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Priority(highPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(midPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(midPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(midPriority).NominatedNodeName("node1").Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("other-victim"), capacity: 1},
			},
			expectedPods:   []string{"other-victim"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Preempt single lower priority pod",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("p1"), capacity: 1},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Priority: Prefer lower priority victim",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node1").Priority(midPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node1").Priority(highPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("p1"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("p2"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("p3"), capacity: 1},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Efficiency: Preempt minimum number of victims",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node1").Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("p1", "p2"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("p1"), capacity: 1},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PDB: Prefer non-violating victim",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("victim-pdb").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).Obj(),
				st.MakePod().Name("victim-no-pdb").UID("v2").Node("node1").Priority(lowPriority).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("victim-pdb"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("victim-no-pdb"), capacity: 1},
			},
			expectedPods:   []string{"victim-no-pdb"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptLowerPriority preemption policy performs preemption with PodGroup victims",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj(),
				},
				schedulingapi.PreemptLowerPriority,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptLowerPriority preemption policy performs preemption with PodGroup victims, ignoring member pod preemption policy",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).PreemptionPolicy(v1.PreemptNever).Obj(),
				},
				schedulingapi.PreemptLowerPriority,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup preemptor with PreemptNever preemption policy does not perform preemption with PodGroup victims",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).PodGroupName("pg2").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeSingle().Priority(lowPriority).Obj(),
				st.MakePodGroup().Name("pg2").UID("pg2").DisruptionModeSingle().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptorWithPreemptionPolicy(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-1").UID("p-1").Priority(highPriority).Obj(),
					st.MakePod().Name("p-2").UID("p-2").Priority(highPriority).Obj(),
				},
				schedulingapi.PreemptNever,
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.Unschedulable, "not eligible due to preemptionPolicy=Never."),
		},
		{
			name:      "PDB: Prefer lower priority pod for preemption, when preemption without pdb violation is not possible",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Label("app", "foo").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node1").Label("app", "foo").Priority(midPriority).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("p1"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("p2"), capacity: 1},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup: Preempt group as a whole",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{"p1", "p2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup: Prefer single pod over podGroup for preemption candidate",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-1").UID("g1").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("g1-1", "g1-2"), capacity: 1},
				{nodeName: "node1", blockingVictims: sets.New("p1"), capacity: 1},
			},
			expectedPods:   []string{"p1"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup: Preempt group as a whole on single node",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("g1-1").UID("g1").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("g1-1")}, // Only g1-1 is blocking
			},
			expectedPods:   []string{"g1-1", "g1-2"}, // Both must be preempted
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PDB: Unit violation if any member violates",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("g1-1").UID("g1").Node("node1").Label("app", "foo").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(lowPriority).Obj(),
			},
			pdbs: []*policy.PodDisruptionBudget{
				{
					Spec:   policy.PodDisruptionBudgetSpec{Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}}},
					Status: policy.PodDisruptionBudgetStatus{DisruptionsAllowed: 0},
				},
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("g1-1")}, // Only g1-1 is blocking
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},   // p1 is also blocking
			},
			expectedPods:   []string{"p1"}, // p1 is preferred because pg1 unit-violates PDB (via g1-1)
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "PodGroup: Prefer preempting single pod over group of same priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("p1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-1").UID("g1").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
				st.MakePod().Name("g1-2").UID("g2").Node("node1").PodGroupName("pg1").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(lowPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("g1-1", "g1-2")},
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{"p1"}, // p1 is preempted because the PodGroup is "more important" at the same priority level
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Failure: Cannot preempt the victim with higher priority",
			nodeNames: []string{"node1"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(highPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(midPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(midPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
		},
		{
			name:      "Failure: Cannot preempt if node is empty",
			nodeNames: []string{"node1"},
			initPods:  []*v1.Pod{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(midPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(midPriority).Obj()},
			),
			blockingRules:  []blockingRule{},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
		},
		{
			name:      "Priority divergence: candidate victim PodGroup has lower priority than the Pods from that group",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(highPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(highPriority).PodGroupName("pg1").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(midPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{"p1", "p2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Priority divergence: candidate victim PodGroup has higher priority than the Pods from that group",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).PodGroupName("pg1").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(midPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(midPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(midPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
		},
		{
			name:      "Priority divergence: preemptor PodGroup has higher priority than the Pod from preemptor PodGroup",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(midPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(midPriority).PodGroupName("pg1").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(midPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(lowPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{"p1", "p2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Priority divergence: preemptor PodGroup has lower priority than the Pod from preemptor PodGroup",
			nodeNames: []string{"node1", "node2"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(midPriority).PodGroupName("pg1").Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(midPriority).PodGroupName("pg1").Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").UID("pg1").DisruptionModeAll().Priority(midPriority).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(lowPriority).Obj(),
				[]*v1.Pod{st.MakePod().Name("p").UID("p").Priority(highPriority).Obj()},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
			},
			expectedPods:   []string{},
			expectedStatus: fwk.NewStatus(fwk.UnschedulableAndUnresolvable),
		},
		{
			name:      "Gang scheduling: do not reprieve if it reduces scheduled pods below max possible",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(midPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(2).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			// maxScheduledCount will be 3 (all 3 preemptor pods can be scheduled if p1, p2, p3 are preempted).
			// If any of p1, p2, p3 are reprieved, scheduledCount drops to 2.
			// Even though 2 >= MinCount(2), we shouldn't reprieve because it reduces scheduledCount < maxScheduledCount(3).
			// So, all 3 should be preempted.
			expectedPods:   []string{"p1", "p2", "p3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Gang scheduling: reprieve if it does not reduce scheduled pods below max possible",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(2).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("p4")},
			},
			// Preemptor has 3 pods and minCount of 2, so maxScheduledCount will be 3.
			// Pod p4 with highest priority of all victims could be reprieved and we will still be able to schedule all of pods.
			expectedPods:   []string{"p1", "p2", "p3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Gang scheduling: schedule as many pods as possible without preempting higher priority pods, but still more than minCount",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(highPriority).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(highPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(1).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("p4")},
			},
			// Preemptor has 3 pods and minCount of 1, but maxScheduledCount will be 2, because there are higher priority pods p3, p4.
			expectedPods:   []string{"p1", "p2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Gang scheduling: do not reprieve victim pod group of lower priority",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Namespace(v1.NamespaceDefault).Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Namespace(v1.NamespaceDefault).Priority(midPriority).DisruptionModeAll().MinCount(1).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(1).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("v1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("v2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("v3")},
			},
			expectedPods:   []string{"v1", "v2", "v3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Gang scheduling: preempt a pod group victim but do not schedule full pod group",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg2").Priority(highPriority).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg2").Priority(highPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Namespace(v1.NamespaceDefault).Priority(midPriority).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").UID("victim-pg2").Namespace(v1.NamespaceDefault).Priority(highPriority).DisruptionModeAll().MinCount(2).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(1).DisruptionModeAll().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("v1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("v2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("v3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("v4")},
			},
			expectedPods:   []string{"v1", "v2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Basic scheduling: do not reprieve if it reduces scheduled pods below max possible",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(midPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
			},
			expectedPods:   []string{"p1", "p2", "p3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Basic scheduling: reprieve if it does not reduce scheduled pods below max possible",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(lowPriority).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(midPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("p4")},
			},
			expectedPods:   []string{"p1", "p2", "p3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Basic scheduling: schedule as many pods as possible without preempting higher priority pods",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("p1").UID("v1").Node("node1").Priority(lowPriority).Obj(),
				st.MakePod().Name("p2").UID("v2").Node("node2").Priority(lowPriority).Obj(),
				st.MakePod().Name("p3").UID("v3").Node("node3").Priority(highPriority).Obj(),
				st.MakePod().Name("p4").UID("v4").Node("node4").Priority(highPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("p1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("p2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("p3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("p4")},
			},
			expectedPods:   []string{"p1", "p2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement",
			nodeNames: []string{"nodeA", "nodeB"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("nodeA").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("nodeB").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).Obj(),
					st.MakePod().Name("p3").UID("p3").Priority(highPriority).Obj(),
				},
			),
			customMockSchedulingFunc: func(ctx context.Context, domainNodes []fwk.NodeInfo) (*fwk.PodGroupAssignments, *fwk.Status) {
				v1Present, v2Present := false, false
				for _, n := range domainNodes {
					for _, pod := range n.GetPods() {
						if pod.GetPod().Name == "v1" {
							v1Present = true
						} else if pod.GetPod().Name == "v2" {
							v2Present = true
						}
					}
				}

				scheduledCount := 2
				if v1Present && !v2Present {
					scheduledCount = 3 // v1 reprieved: this increases scheduledCount!
				}

				proposedAssignments := make([]fwk.ProposedAssignment, scheduledCount)
				for i := range scheduledCount {
					pod := st.MakePod().Name(fmt.Sprintf("p%d", i+1)).UID(fmt.Sprintf("p%d", i+1)).Obj()
					proposedAssignments[i] = &testProposedAssignment{pod: pod, nodeName: "node1"}
				}
				return &fwk.PodGroupAssignments{
					ProposedAssignments: proposedAssignments,
				}, fwk.NewStatus(fwk.Success)
			},
			expectedPods:   []string{"v2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement (gang > minCount)",
			nodeNames: []string{"nodeA", "nodeB"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("nodeA").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("nodeB").Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).MinCount(3).Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p1").UID("p1").Priority(highPriority).Obj(),
					st.MakePod().Name("p2").UID("p2").Priority(highPriority).Obj(),
					st.MakePod().Name("p3").UID("p3").Priority(highPriority).Obj(),
					st.MakePod().Name("p4").UID("p4").Priority(highPriority).Obj(),
				},
			),
			customMockSchedulingFunc: func(ctx context.Context, domainNodes []fwk.NodeInfo) (*fwk.PodGroupAssignments, *fwk.Status) {
				v1Present, v2Present := false, false
				for _, n := range domainNodes {
					for _, pod := range n.GetPods() {
						if pod.GetPod().Name == "v1" {
							v1Present = true
						} else if pod.GetPod().Name == "v2" {
							v2Present = true
						}
					}
				}

				scheduledCount := 3
				if v1Present && !v2Present {
					scheduledCount = 4 // v1 reprieved: this increases scheduledCount!
				}

				proposedAssignments := make([]fwk.ProposedAssignment, scheduledCount)
				for i := range scheduledCount {
					pod := st.MakePod().Name(fmt.Sprintf("p%d", i+1)).UID(fmt.Sprintf("p%d", i+1)).Obj()
					proposedAssignments[i] = &testProposedAssignment{pod: pod, nodeName: "node1"}
				}

				return &fwk.PodGroupAssignments{
					ProposedAssignments: proposedAssignments,
				}, fwk.NewStatus(fwk.Success)
			},
			expectedPods:   []string{"v2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Basic scheduling: do not reprieve victim pod group of lower priority",
			nodeNames: []string{"node1", "node2", "node3"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Namespace(v1.NamespaceDefault).Priority(lowPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Namespace(v1.NamespaceDefault).Priority(midPriority).DisruptionModeAll().MinCount(1).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("v1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("v2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("v3")},
			},
			expectedPods:   []string{"v1", "v2", "v3"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
		{
			name:      "Basic scheduling: preempt a pod group victim but do not schedule full pod group",
			nodeNames: []string{"node1", "node2", "node3", "node4"},
			initPods: []*v1.Pod{
				st.MakePod().Name("v1").UID("v1").Node("node1").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v2").UID("v2").Node("node2").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg").Priority(midPriority).Obj(),
				st.MakePod().Name("v3").UID("v3").Node("node3").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg2").Priority(highPriority).Obj(),
				st.MakePod().Name("v4").UID("v4").Node("node4").Namespace(v1.NamespaceDefault).PodGroupName("victim-pg2").Priority(highPriority).Obj(),
			},
			initPodGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").UID("victim-pg").Namespace(v1.NamespaceDefault).Priority(midPriority).DisruptionModeAll().MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").UID("victim-pg2").Namespace(v1.NamespaceDefault).Priority(highPriority).DisruptionModeAll().MinCount(2).Obj(),
			},
			preemptor: makePodGroupPreemptor(
				st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).BasicPolicy().DisruptionModeAll().Obj(),
				[]*v1.Pod{
					st.MakePod().Name("p-a").UID("p-a").Priority(highPriority).Obj(),
					st.MakePod().Name("p-b").UID("p-b").Priority(highPriority).Obj(),
					st.MakePod().Name("p-c").UID("p-c").Priority(highPriority).Obj(),
				},
			),
			blockingRules: []blockingRule{
				{nodeName: "node1", capacity: 1, blockingVictims: sets.New("v1")},
				{nodeName: "node2", capacity: 1, blockingVictims: sets.New("v2")},
				{nodeName: "node3", capacity: 1, blockingVictims: sets.New("v3")},
				{nodeName: "node4", capacity: 1, blockingVictims: sets.New("v4")},
			},
			expectedPods:   []string{"v1", "v2"},
			expectedStatus: fwk.NewStatus(fwk.Success),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			nodes := make([]*v1.Node, len(tt.nodeNames))
			for i, nodeName := range tt.nodeNames {
				nodes[i] = st.MakeNode().Name(nodeName).Obj()
			}

			// Build nodes with pods
			nodeInfos := make(map[string]fwk.NodeInfo)
			for _, node := range nodes {
				nodeInfos[node.Name] = framework.NewNodeInfo()
				nodeInfos[node.Name].SetNode(node)
			}
			for _, p := range tt.initPods {
				podInfo, _ := framework.NewPodInfo(p)
				nodeInfos[p.Spec.NodeName].AddPodInfo(podInfo)
			}

			var domainNodes []fwk.NodeInfo
			for _, name := range tt.nodeNames {
				domainNodes = append(domainNodes, nodeInfos[name])
			}

			podGroups := make(map[string]*schedulingapi.PodGroup)
			for _, pg := range tt.initPodGroups {
				podGroups[pg.Name] = pg
			}
			pgLister := &mockPodGroupLister{podGroups: podGroups}
			domain := newDomainForWorkloadPreemption(domainNodes, pgLister, "test-domain")

			// Create a mock podGroupSchedulingFunc.
			// This simulates whether the preempting PodGroup can schedule given the current
			// hypothetical state of the cluster (where some candidate victims might be removed).
			var mockSchedulingFunc fwk.PodGroupSchedulingFunc = func(ctx context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
				if tt.customMockSchedulingFunc != nil {
					return tt.customMockSchedulingFunc(ctx, domainNodes)
				}
				neededSlots := len(tt.preemptor.Members())
				if tt.preemptor.podGroup != nil {
					if tt.preemptor.podGroup.Spec.SchedulingPolicy.Gang != nil {
						neededSlots = int(tt.preemptor.podGroup.Spec.SchedulingPolicy.Gang.MinCount)
					} else {
						// For non-gang pod groups (e.g., BasicPolicy), scheduling even 1 pod is considered a success
						// since there's no all-or-nothing constraint.
						neededSlots = 1
					}
				}

				availableSlots := 0
				nodeMap := make(map[string]fwk.NodeInfo)
				for _, n := range domainNodes {
					nodeMap[n.Node().Name] = n
				}

				for _, rule := range tt.blockingRules {
					node, exists := nodeMap[rule.nodeName]
					if !exists {
						continue
					}

					isBlocked := false
					for _, pod := range node.GetPods() {
						// If any of the blocking victims are still on the simulated node, it provides 0 capacity.
						if rule.blockingVictims.Has(pod.GetPod().Name) {
							isBlocked = true
							break
						}
					}

					// If all blocking victims are removed, the node provides its capacity.
					if !isBlocked {
						availableSlots += rule.capacity
					}
				}

				if availableSlots >= neededSlots {
					assignmentsCount := min(availableSlots, len(tt.preemptor.Members()))
					proposedAssignments := make([]fwk.ProposedAssignment, assignmentsCount)
					for i := range assignmentsCount {
						proposedAssignments[i] = &testProposedAssignment{pod: tt.preemptor.Members()[i], nodeName: "node1"}
					}
					return &fwk.PodGroupAssignments{
						ProposedAssignments: proposedAssignments,
					}, fwk.NewStatus(fwk.Success)
				}
				return nil, fwk.NewStatus(fwk.Unschedulable)
			}

			pl := &PodGroupEvaluator{
				podGroupSnapshot: pgLister,
			}

			res, gotStatus := pl.selectVictimsOnDomain(ctx, tt.preemptor, domain, tt.pdbs, mockSchedulingFunc)
			if !gotStatus.IsSuccess() {
				t.Logf("SelectVictimsOnDomain failed: %v", gotStatus.Message())
			}

			wantCode := tt.expectedStatus.Code()
			gotCode := gotStatus.Code()
			if gotCode != wantCode {
				t.Errorf("Status mismatch. Want %v, Got %v", wantCode, gotCode)
			}
			if wantCode != fwk.Success {
				return
			}
			if res == nil {
				t.Fatalf("expected non-nil victims on success")
			}

			gotNames := sets.Set[string]{}
			for _, p := range res.victims.Pods {
				gotNames.Insert(p.Name)
			}
			wantNames := sets.New(tt.expectedPods...)
			if diff := cmp.Diff(wantNames, gotNames); diff != "" {
				t.Errorf("Victims mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestMoreImportantVictim(t *testing.T) {
	newPodInfo := func(p *v1.Pod) fwk.PodInfo {
		pi, _ := framework.NewPodInfo(p)
		return pi
	}

	now := &metav1.Time{Time: time.Unix(1000, 0)}
	before := &metav1.Time{Time: time.Unix(500, 0)}

	tests := []struct {
		name string
		vi1  *victim
		vi2  *victim
		want bool
	}{
		{
			name: "vi1 has higher priority",
			vi1:  &victim{priority: 20},
			vi2:  &victim{priority: 10},
			want: true,
		},
		{
			name: "vi2 has higher priority",
			vi1:  &victim{priority: 10},
			vi2:  &victim{priority: 20},
			want: false,
		},
		{
			name: "vi1 is PG, vi2 is Pod, same priority",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}},
			want: true,
		},
		{
			name: "vi1 is Pod, vi2 is PG, same priority",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().PodGroupName("pg").Obj())}},
			want: false,
		},
		{
			name: "both Pods, vi1 older",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now},
			want: true,
		},
		{
			name: "both Pods, vi2 older",
			vi1:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: now},
			vi2:  &victim{priority: 10, pods: []fwk.PodInfo{newPodInfo(st.MakePod().Obj())}, earliestStartTime: before},
			want: false,
		},
		{
			name: "both PGs, vi1 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			want: true,
		},
		{
			name: "both PGs, vi2 larger",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
			},
			want: false,
		},
		{
			name: "both PGs, same size, vi1 older",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
			},
			want: true,
		},
		{
			name: "both PGs, same size, vi2 older",
			vi1: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: now,
			},
			vi2: &victim{
				priority: 10,
				pods: []fwk.PodInfo{
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
					newPodInfo(st.MakePod().PodGroupName("pg").Obj()),
				},
				earliestStartTime: before,
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := moreImportantVictim(tt.vi1, tt.vi2)
			if got != tt.want {
				t.Errorf("MoreImportantVictim() = %v, want %v", got, tt.want)
			}
		})
	}
}

type testProposedAssignment struct {
	pod      *v1.Pod
	nodeName string
}

func (a *testProposedAssignment) GetPod() *v1.Pod {
	return a.pod
}

func (a *testProposedAssignment) GetNodeName() string {
	return a.nodeName
}

func TestPodGroupEvaluator_SelectVictimsOnDomain_NominatedNodes(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	p1 := st.MakePod().Name("p1").UID("p1").Obj()
	p2 := st.MakePod().Name("p2").UID("p2").Obj()

	preemptor := makePodGroupPreemptor(
		st.MakePodGroup().Name("preemptor-pg").Priority(highPriority).Obj(),
		[]*v1.Pod{p1, p2},
	)

	domainNodes := []fwk.NodeInfo{
		framework.NewNodeInfo(),
	}
	domainNodes[0].SetNode(st.MakeNode().Name("node1").Obj())

	// Add a low priority pod as a potential victim to satisfy the check
	p3 := st.MakePod().Name("p3").UID("p3").Node("node1").Priority(lowPriority).Obj()
	podInfo, _ := framework.NewPodInfo(p3)
	domainNodes[0].AddPodInfo(podInfo)

	pgLister := &mockPodGroupLister{podGroups: make(map[string]*schedulingapi.PodGroup)}
	domain := newDomainForWorkloadPreemption(domainNodes, pgLister, "test-domain")

	mockSchedulingFunc := func(ctx context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
		return &fwk.PodGroupAssignments{
			ProposedAssignments: []fwk.ProposedAssignment{
				&testProposedAssignment{pod: p1, nodeName: "node1"},
				&testProposedAssignment{pod: p2, nodeName: ""}, // No node assigned
			},
		}, fwk.NewStatus(fwk.Success)
	}

	pl := &PodGroupEvaluator{}

	result, gotStatus := pl.selectVictimsOnDomain(ctx, preemptor, domain, nil, mockSchedulingFunc)
	if !gotStatus.IsSuccess() {
		t.Fatalf("SelectVictimsOnDomain failed: %v", gotStatus.Message())
	}

	if result == nil {
		t.Fatalf("expected non-nil result")
	}

	if len(result.nominatedNodeNames) != 1 {
		t.Errorf("Expected 1 nominated node name, got %d", len(result.nominatedNodeNames))
	}

	namespacedName := types.NamespacedName{Namespace: p1.Namespace, Name: p1.Name}
	if info, ok := result.nominatedNodeNames[namespacedName]; !ok || info.NominatedNodeName != "node1" {
		t.Errorf("Expected p1 to be nominated for node1, got %v", info)
	}
}
