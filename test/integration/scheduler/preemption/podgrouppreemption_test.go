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
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

// TestPodGroupPreemption tests preemption scenarios involving pod groups.
func TestPodGroupPreemption(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:         true,
		features.GangScheduling:          true,
		features.WorkloadAwarePreemption: true,
	})

	tests := []struct {
		name                       string
		nodes                      []*v1.Node
		podGroups                  []*schedulingapi.PodGroup
		initialPods                []*v1.Pod // pods that should be scheduled before preemption starts
		preemptorPods              []*v1.Pod // pods that belong to a group and should trigger preemption
		pdb                        *policyv1.PodDisruptionBudget
		expectedScheduled          []string
		expectedPreempted          []string
		expectedPreemptedAnyOf     []string
		expectedUnschedulable      []string
		expectedPodsPreemptedByWAP int
	}{
		{
			name: "Full PodGroup Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Full PodGroup Preemption for basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Partial Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				// low-1 takes all CPU on node1
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				// low-2 takes half CPU on node2
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node2").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// high-3 will fit on node2 (it has 1 CPU free).
			// high-1 and high-2 will fit on node1 if low-1 is preempted.
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "low-2"},
			expectedPreempted:          []string{"low-1"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Partial Preemption with basic policy",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				// low-1 takes half CPU on node1
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Node("node1").Obj(),
				// very-low-1 takes all CPU on node2
				st.MakePod().Name("very-low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(5).Node("node2").Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// high-1 will fit on node1 (it has 1 CPU free).
			// high-2 and high-3 will fit on node2 if very-low-1 is preempted.
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "low-1"},
			expectedPreempted:          []string{"very-low-1"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "PDB Violation Handling (Reprieve)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Label("app", "foo").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			pdb: &policyv1.PodDisruptionBudget{
				ObjectMeta: metav1.ObjectMeta{Name: "foo-pdb"},
				Spec: policyv1.PodDisruptionBudgetSpec{
					MinAvailable: &intstr.IntOrString{IntVal: 2},
					Selector:     &metav1.LabelSelector{MatchLabels: map[string]string{"app": "foo"}},
				},
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.5"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "0.5"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2"},
			expectedPreempted:          []string{"low-3"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Multi-node Preemption",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(4).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-4").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "high-3", "high-4"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3", "low-4"},
			expectedPodsPreemptedByWAP: 4,
		},
		{
			name: "Insufficient Resources (No Preemption)",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("mid-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(500).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"mid-1", "low-1", "low-2"},
			expectedPreempted:          []string{},
			expectedUnschedulable:      []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 0,
		},
		{
			name: "Priority-based Victim Selection",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("mid-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "high-2", "mid-1"},
			expectedPreempted:          []string{"low-1", "low-2"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Preempt the whole PodGroup even if preempting a single Pod would suffice",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(10).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Preempt the whole basic PodGroup with a PodGroup disruption mode",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(10).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1"},
			expectedPreempted:          []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Priority divergence in PodGroups - preemptor PodGroup has higher priority than the victim candidate PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(10).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).Obj(),
				st.MakePod().Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			expectedScheduled:          []string{"low-1"},
			expectedPreempted:          []string{"high-1", "high-2", "high-3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Priority divergence in PodGroups - preemptor PodGroup has too low priority",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"low-1", "low-2", "low-3"},
			expectedPreempted:          []string{},
			expectedUnschedulable:      []string{"high-1"},
			expectedPodsPreemptedByWAP: 0,
		},
		{
			name: "Preemptor Pod without PodGroupName does not respect the PodGroup disruption mode",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(10).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(30).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(20).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "low-1", "low-2"},
			expectedPreempted:          []string{"low-3"},
			expectedPodsPreemptedByWAP: 0,
		},
		{
			name: "Gang scheduling: do not reprieve if it reduces scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Gang scheduling: reprieve if it does not reduce scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c", "p4"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Gang scheduling: schedule as many pods as possible without preempting higher priority pods, but still more than minCount",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p3", "p4"},
			expectedPreempted:          []string{"p1", "p2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Gang scheduling: do not reprieve victim pod group of lower priority",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(1).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"v1", "v2", "v3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Gang scheduling: preempt a pod group victim but do not schedule full pod group",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").Namespace("default").Priority(200).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(2).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).DisruptionMode(schedulingapi.DisruptionModePodGroup).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("v4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "v3", "v4"},
			expectedPreempted:          []string{"v1", "v2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Basic scheduling: do not reprieve if it reduces scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: reprieve if it does not reduce scheduled pods below max possible",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(50).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c", "p4"},
			expectedPreempted:          []string{"p1", "p2", "p3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: schedule as many pods as possible without preempting higher priority pods",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("p1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("p3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("p4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p3", "p4"},
			expectedPreempted:          []string{"p1", "p2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Basic scheduling: do not reprieve victim pod group of lower priority",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(1).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "p-c"},
			expectedPreempted:          []string{"v1", "v2", "v3"},
			expectedPodsPreemptedByWAP: 3,
		},
		{
			name: "Basic scheduling: preempt a pod group victim but do not schedule full pod group",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node3").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("node4").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(50).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(2).Obj(),
				st.MakePodGroup().Name("victim-pg2").Namespace("default").Priority(200).DisruptionMode(schedulingapi.DisruptionModePodGroup).MinCount(2).Obj(),
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().DisruptionMode(schedulingapi.DisruptionModePodGroup).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("v1").Node("node1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v2").Node("node2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(50).Obj(),
				st.MakePod().Name("v3").Node("node3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
				st.MakePod().Name("v4").Node("node4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg2").ZeroTerminationGracePeriod().Priority(200).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p-a").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(102).Obj(),
				st.MakePod().Name("p-b").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(101).Obj(),
				st.MakePod().Name("p-c").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"p-a", "p-b", "v3", "v4"},
			expectedPreempted:          []string{"v1", "v2"},
			expectedUnschedulable:      []string{"p-c"},
			expectedPodsPreemptedByWAP: 2,
		},
		{
			name: "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement",
			nodes: []*v1.Node{
				st.MakeNode().Name("nodea").Label("topology.kubernetes.io/zone", "zoneA").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("nodeb").Label("topology.kubernetes.io/zone", "zoneB").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("va").Node("nodea").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(50).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneA"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("vb").Node("nodeb").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneB"}, st.NodeSelectorTypeMatchExpressions).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p1").Label("pod", "p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{
								Weight: 100,
								Preference: v1.NodeSelectorTerm{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}},
									},
								},
							},
						},
					}).Obj(),
				st.MakePod().Name("p2").Label("pod", "p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p3").Label("pod", "p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"p1", "p2", "p3", "va"},
			expectedPreempted:          []string{"vb"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Reprieval allows more pods to schedule than initial maxScheduledCount due to greedy placement (gang > minCount)",
			nodes: []*v1.Node{
				st.MakeNode().Name("nodea").Label("topology.kubernetes.io/zone", "zoneA").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
				st.MakeNode().Name("nodeb").Label("topology.kubernetes.io/zone", "zoneB").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(3).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("va").Node("nodea").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").ZeroTerminationGracePeriod().Priority(50).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneA"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("vb").Node("nodeb").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(10).NodeAffinityIn("topology.kubernetes.io/zone", []string{"zoneB"}, st.NodeSelectorTypeMatchExpressions).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("p1").Label("pod", "p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					NodeAffinity(&v1.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
							{
								Weight: 100,
								Preference: v1.NodeSelectorTerm{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{Key: "topology.kubernetes.io/zone", Operator: v1.NodeSelectorOpIn, Values: []string{"zoneA"}},
									},
								},
							},
						},
					}).Obj(),
				st.MakePod().Name("p2").Label("pod", "p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p3").Label("pod", "p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
				st.MakePod().Name("p4").Label("pod", "p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).
					PodAffinityExists("pod", "topology.kubernetes.io/zone", st.PodAffinityWithRequiredReq).Obj(),
			},
			expectedScheduled:          []string{"p1", "p2", "p3", "p4", "va"},
			expectedPreempted:          []string{"vb"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Pods from a single gang PodGroup with DisruptionMode=Pod can be preempted individually by the higher priority gang PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(10).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:          []string{"high-1"},
			expectedPreemptedAnyOf:     []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Pods from a single gang PodGroup with DisruptionMode=Pod can be preempted individually by the higher priority basic PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("preemptor-pg").Namespace("default").Priority(100).BasicPolicy().Obj(),
				st.MakePodGroup().Name("victim-pg").Namespace("default").Priority(10).MinCount(2).Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("victim-pg").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("preemptor-pg").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			// We expect one of the pods from victim-pg to be preempted, but do not choose specific pod.
			expectedScheduled:          []string{"high-1"},
			expectedPreemptedAnyOf:     []string{"low-1", "low-2", "low-3"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Pods from a single basic PodGroup with DisruptionMode=Pod can be preempted individually by the higher priority gang PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).MinCount(1).Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(30).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(20).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "low-1", "low-2"},
			expectedPreempted:          []string{"low-3"},
			expectedPodsPreemptedByWAP: 1,
		},
		{
			name: "Pods from a single basic PodGroup with DisruptionMode=Pod can be preempted individually by the higher priority basic PodGroup",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3", v1.ResourceMemory: "4Gi", v1.ResourcePods: "32"}).Obj(),
			},
			podGroups: []*schedulingapi.PodGroup{
				st.MakePodGroup().Name("pg1").Namespace("default").Priority(100).BasicPolicy().Obj(),
				st.MakePodGroup().Name("pg2").Namespace("default").Priority(10).BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(30).Obj(),
				st.MakePod().Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(20).Obj(),
				st.MakePod().Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg2").ZeroTerminationGracePeriod().Priority(10).Obj(),
			},
			preemptorPods: []*v1.Pod{
				st.MakePod().Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").ZeroTerminationGracePeriod().Priority(100).Obj(),
			},
			expectedScheduled:          []string{"high-1", "low-1", "low-2"},
			expectedPreempted:          []string{"low-3"},
			expectedPodsPreemptedByWAP: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-preemption",
				scheduler.WithPodMaxBackoffSeconds(0),
				scheduler.WithPodInitialBackoffSeconds(0))
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			// Create nodes
			for _, n := range tt.nodes {
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create node %s: %v", n.Name, err)
				}
			}

			// Create PDB if specified
			if tt.pdb != nil {
				tt.pdb.Namespace = ns
				if _, err := cs.PolicyV1().PodDisruptionBudgets(ns).Create(testCtx.Ctx, tt.pdb, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PDB: %v", err)
				}
			}

			// 1. Create PodGroups
			for _, pg := range tt.podGroups {
				pg.Namespace = ns
				if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PodGroup %s: %v", pg.Name, err)
				}
			}

			// 2. Create initial pods
			for _, p := range tt.initialPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}

			// Wait for initial pods to be scheduled
			for _, p := range tt.initialPods {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Errorf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
				}
			}

			// 3. Create preemptor pods
			for _, p := range tt.preemptorPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}

			// 4. Wait for preemption to complete if WAP calls are expected
			if tt.expectedPodsPreemptedByWAP > 0 {
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
					wapCalls := 0
					allExpectedPreempted := append(tt.expectedPreempted, tt.expectedPreemptedAnyOf...)
					for _, podName := range allExpectedPreempted {
						events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
							FieldSelector: "involvedObject.name=" + podName,
						})
						if err != nil {
							return false, err
						}
						for _, event := range events.Items {
							if event.Reason == "Preempted" && strings.HasPrefix(event.Message, "Preempted by podgroup") {
								wapCalls++
								break
							}
						}
					}
					return wapCalls == tt.expectedPodsPreemptedByWAP, nil
				})
				if err != nil {
					t.Errorf("WorkloadAwarePreemption was not called %d times within timeout", tt.expectedPodsPreemptedByWAP)
				}
			}

			// 6. Verify unschedulable pods
			for _, podName := range tt.expectedUnschedulable {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodUnschedulable(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be unschedulableso  but wasn't: %v", podName, err)
				}
			}

			// 7. Verify scheduled pods
			for _, podName := range tt.expectedScheduled {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false,
					testutils.PodScheduled(cs, ns, podName)); err != nil {
					t.Errorf("Pod %s was expected to be scheduled but wasn't: %v", podName, err)
				}
			}

			// 8. Verify preempted pods
			for _, podName := range tt.expectedPreempted {
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
					func(ctx context.Context) (bool, error) {
						pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
						if err != nil {
							return apierrors.IsNotFound(err), nil
						}
						if pod.DeletionTimestamp != nil {
							return true, nil
						}
						_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
						return cond != nil, nil
					}); err != nil {
					t.Errorf("Pod %s was expected to be preempted but wasn't", podName)
				}
			}

			// 9. Verify expectedPreemptedAnyOf pods
			if len(tt.expectedPreemptedAnyOf) > 0 {
				preemptedAnyOfCount := 0
				for _, podName := range tt.expectedPreemptedAnyOf {
					err := wait.PollUntilContextTimeout(testCtx.Ctx, 200*time.Millisecond, 5*time.Second, false,
						func(ctx context.Context) (bool, error) {
							pod, err := cs.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
							if err != nil {
								return apierrors.IsNotFound(err), nil
							}
							if pod.DeletionTimestamp != nil {
								return true, nil
							}
							_, cond := podutil.GetPodCondition(&pod.Status, v1.DisruptionTarget)
							return cond != nil, nil
						})
					if err == nil {
						preemptedAnyOfCount++
					}
				}
				expectedCountFromAnyOf := tt.expectedPodsPreemptedByWAP - len(tt.expectedPreempted)
				if preemptedAnyOfCount != expectedCountFromAnyOf {
					t.Errorf("Expected exactly %d pods from %v to be preempted, but found %d", expectedCountFromAnyOf, tt.expectedPreemptedAnyOf, preemptedAnyOfCount)
				}
			}
		})
	}
}

func TestPodGroupAsyncPreemption(t *testing.T) {

	tests := []asyncPreemptionTest{
		{
			// Very basic test case: if it fails, the basic scenario is broken somewhere.
			name: "basic: async preemption happens expectedly",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(2),
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(2),
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod after the preemption",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			name: "basic async preemption with 1 victim, preemptor gated until preemption API call finishes",
			scenarios: []scenario{
				{
					name: "create victim",
					createPod: &createPod{
						pod: st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the preemptor Pod is in the queue and gated",
					podGatedInQueue: "preemptor",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(1),
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			name: "Higher priority Pod takes over the place for lower priority Pod that is running the preemption",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor-high-priority",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(4),
				},
				{
					// This Pod is higher priority than the preemptor Pod.
					// Even though the preemptor Pod is nominated to the node, this Pod can take over the place.
					name: "create pod group for super-high-priority preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor-super-high-priority").MinCount(1).Priority(200).Obj(),
					},
				},
				{
					name: "create a second Pod that is higher priority than the first preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-super-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(200).PodGroupName("pg-preemptor-super-high-priority").Obj(),
					},
				},
				{
					name: "schedule the super-high-priority Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor-super-high-priority",
						expectUnschedulable: true,
					},
				},
				{
					name:                 "check the super-high-priority Pod making the preemption API calls",
					podRunningPreemption: ptr.To(5),
				},
				{
					// the super-high-priority preemptor should enter the preemption
					// and select the place where the preemptor-high-priority selected.
					// So, basically both goroutines are preempting the same Pods.
					name:            "check the super-high-priority pod is in the queue and gated",
					podGatedInQueue: "preemptor-super-high-priority",
				},
				{
					name:               "complete the preemption API calls of super-high-priority",
					completePreemption: "preemptor-super-high-priority",
				},
				{
					name:               "complete the preemption API calls of high-priority",
					completePreemption: "preemptor-high-priority",
				},
				{
					name: "schedule the super-high-priority Pod",
					schedulePod: &schedulePod{
						podName:       "preemptor-super-high-priority",
						expectSuccess: true,
					},
				},
				// In WAP we send the unschedulable pod straight to the queue with backoff time. 
				// By default it's set to 0s, making the pod jump straight to activeQ.
				// We set the time to 1s to give some time for pod to be considered unschedulable.
				{
					name: "schedule the high-priority Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
					},
				},
			},
			initialBackoffSeconds: 1,
			maxBackoffSeconds: 1,
		},
		{
			name: "Lower priority Pod can select the same place where the higher priority Pod is preempting if the node is big enough",
			scenarios: []scenario{
				{
					name: "create scheduled Pod",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					// It will preempt two victims.
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor-high-priority",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the pod is in the queue and gated",
					podGatedInQueue: "preemptor-high-priority",
				},
				{
					name:                 "check the preemptor Pod making the preemption API calls",
					podRunningPreemption: ptr.To(4),
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-second-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					// This Pod is lower priority than the preemptor Pod.
					// Given the preemptor PodGroup don't support nominated node names yet, this Pod should be unschedulable.
					// This Pod will trigger the preemption to target the any of the 4 victims.
					name: "create a second Pod that is lower priority than the first preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(50).PodGroupName("pg-second-preemptor").Obj(),
					},
				},
				{
					name: "schedule the mid-priority Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor-mid-priority",
						expectUnschedulable: true,
					},
				},
				{
					name:            "check the mid-priority pod is in the queue and gated",
					podGatedInQueue: "preemptor-mid-priority",
				},
				{
					name:                 "check the mid-priority Pod making the preemption API calls",
					podRunningPreemption: ptr.To(5),
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-high-priority",
				},
				{
					// the preemptor pod should be popped from the queue before the mid-priority pod.
					name: "schedule the preemptor Pod again",
					schedulePod: &schedulePod{
						podName:       "preemptor-high-priority",
						expectSuccess: true,
					},
				},
				// The last pod will get back to the queue with backoff time, run preemption again.
				{
					name: "schedule the mid-priority Pod again",
					schedulePod: &schedulePod{
						podName:             "preemptor-mid-priority",
						expectUnschedulable: true,
					},
				},
				{
					name:                 "check the mid-priority Pod making the preemption API calls",
					podRunningPreemption: ptr.To(5),
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name: "schedule the mid-priority Pod again",
					schedulePod: &schedulePod{
						podName:       "preemptor-mid-priority",
						expectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim is deleted.
			// Preemptor pod is woken up by the Pod/Delete event and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled after victim-in-binding is deleted",
			scenarios: []scenario{
				{
					name: "create victim Pod that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			scenarios: []scenario{
				{
					name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).TerminationGracePeriodSeconds(1000).Container("image").Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
			},
		},
		{
			// This scenario verifies the fix for https://github.com/kubernetes/kubernetes/issues/134217
			// Scenario reproduces the issue, but with a victim that is under graceful termination:
			// Victim pod takes long in binding. Preemptor pod attempts preemption, goes to unschedulable, then the victim's graceful termination is initiated.
			// Preemptor pod is woken up by the Pod/Update event (working like AssignedPodDeleted) and is being scheduled, even before the victim binding is terminated.
			name: "victim blocked in binding, preemptor pod gets scheduled when victim-in-binding is under graceful termination",
			scenarios: []scenario{
				{
					name: "create victim Pod with long termination grace period that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").TerminationGracePeriodSeconds(1000).Priority(1).Obj(),
					},
				},
				{
					name: "schedule victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the preemptor Pod",
					schedulePod: &schedulePod{
						podName:             "preemptor",
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API call",
					completePreemption: "preemptor",
				},
				{
					name: "schedule the preemptor Pod again and expect it to be scheduled (assumed victim pod was forgotten)",
					schedulePod: &schedulePod{
						podName:       "preemptor",
						expectSuccess: true,
					},
				},
				{
					name:       "resume binding of the blocked pod",
					resumeBind: true,
				},
			},
		},
		{
			// Expected test outcome: lower priority Pod switches to another node, does not get stuck in unschedulable queue forever. (This part is in comment due to test name length limit.)
			name: "While lower priority Pod is waiting for preemption, higher priority Pod takes its place on the node",
			scenarios: []scenario{
				{
					name: "create N-1 victim Pods on the first node",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(3),
					},
				},
				{
					name: "create the last victim Pod on the first node, that is going to be blocked in binding",
					createPod: &createPod{
						pod: st.MakePod().Name(podBlockedInBindingName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
					},
				},
				{
					name: "schedule the last victim Pod",
					schedulePod: &schedulePod{
						podName: podBlockedInBindingName,
					},
				},
				{
					name: "create pod group for preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-preemptor").MinCount(1).Priority(50).Obj(),
					},
				},
				{
					name: "create a mid-priority preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-mid-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(50).PodGroupName("pg-preemptor").Obj(),
					},
				},
				{
					name: "schedule the mid-priority preemptor Pod",
					schedulePod: &schedulePod{
						podName: "preemptor-mid-priority",
					},
				},
				{
					name:               "complete the preemption API calls",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name:            "check the mid-priority preemptor Pod is gated, waiting for the last victim to be preempted",
					podGatedInQueue: "preemptor-mid-priority",
				},
				{
					name:       "create node2",
					createNode: "node2",
				},
				{
					name: "create victim Pods on node2",
					createPod: &createPod{
						pod:   st.MakePod().GenerateName("victim-").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Node("node2").Container("image").ZeroTerminationGracePeriod().Priority(1).Obj(),
						count: ptr.To(4),
					},
				},
				{
					name: "create pod group for high priority preemptor",
					createPodGroup: &createPodGroup{
						podGroup: st.MakePodGroup().Name("pg-high-priority-preemptor").MinCount(1).Priority(100).Obj(),
					},
				},
				{
					name: "create a high-priority preemptor Pod",
					createPod: &createPod{
						pod: st.MakePod().Name("preemptor-high-priority").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).PodGroupName("pg-high-priority-preemptor").Obj(),
					},
				},
				{
					name: "schedule the high-priority preemptor Pod and expect it to get scheduled on node1",
					// While we don't check explicitly that Pod is scheduled on node1, we can assume that because
					// Pod won't fit on node2 without preemption and there are enough resources on node1.
					schedulePod: &schedulePod{
						podName:       "preemptor-high-priority",
						expectSuccess: true,
					},
				},
				{
					name:       "allow the preemption of the last victim Pod on node1 to finish",
					resumeBind: true,
				},
				{
					name: "check that mid-priority preemptor Pod got activated by completed preemption and try scheduling it again",
					schedulePod: &schedulePod{
						podName: "preemptor-mid-priority",
						// Pod won't fit on node1 anymore and should trigger preemptions on node2.
						expectUnschedulable: true,
					},
				},
				{
					name:               "complete the preemption API calls on node2",
					completePreemption: "preemptor-mid-priority",
				},
				{
					name: "check that mid-priority Pod got activated, schedule it on node2",
					schedulePod: &schedulePod{
						podName:       "preemptor-mid-priority",
						expectSuccess: true,
					},
				},
			},
		},
	}

	runAsyncPreemptionScenarios(t, tests, true)
}
