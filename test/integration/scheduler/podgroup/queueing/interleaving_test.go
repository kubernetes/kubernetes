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

package queueing

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func TestPodGroupInterleaving(t *testing.T) {
	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "high priority standalone pod is scheduled before low priority gang",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create low priority gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg-low").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create low priority gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("low-g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
						st.MakePod().Name("low-g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
						st.MakePod().Name("low-g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all gang pods are in active queue",
					WaitForPodsInActiveQ: []string{"low-g1", "low-g2", "low-g3"},
				},
				{
					Name: "Create high priority standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("p-high").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"p-high", "low-g1", "low-g2", "low-g3"},
				},
				{
					Name:           "Schedule high priority standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify high priority standalone pod is scheduled first",
					WaitForPodsScheduled: []string{"p-high"},
				},
				{
					Name:           "Attempt scheduling low priority gang",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify low priority gang cannot fit and becomes unschedulable",
					WaitForPodsUnschedulable: []string{"low-g1", "low-g2", "low-g3"},
				},
				{
					Name: "Verify gang PodGroup condition is set to Unschedulable",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg-low",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonUnschedulable,
					},
				},
			},
		},
		{
			name: "high priority gang is scheduled before low priority standalone pods",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name: "Create low priority standalone pods",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("low-s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
						st.MakePod().Name("low-s2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all standalone pods are in active queue",
					WaitForPodsInActiveQ: []string{"low-s1", "low-s2"},
				},
				{
					Name:           "Create high priority gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg-high").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create high priority gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
						st.MakePod().Name("high-g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
						st.MakePod().Name("high-g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"high-g1", "high-g2", "high-g3", "low-s1", "low-s2"},
				},
				{
					Name:           "Schedule high priority gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify all high priority gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"high-g1", "high-g2", "high-g3"},
				},
				{
					Name:           "Schedule first low priority standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify first low priority standalone pod is scheduled successfully",
					WaitForPodsScheduled: []string{"high-g1", "high-g2", "high-g3", "low-s1"},
				},
				{
					Name:           "Attempt scheduling second low priority standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify second low priority standalone pod cannot fit in remaining capacity",
					WaitForPodsUnschedulable: []string{"low-s2"},
				},
			},
		},
		{
			name: "standalone pod is scheduled before newer, equal priority gang",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"s1"},
				},
				{
					Name:           "Create gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create equal priority gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"s1", "g1", "g2", "g3"},
				},
				{
					Name:           "Schedule older standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify older standalone pod is scheduled first",
					WaitForPodsScheduled: []string{"s1"},
				},
				{
					Name:           "Attempt scheduling newer gang",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify newer gang cannot fit and becomes unschedulable",
					WaitForPodsUnschedulable: []string{"g1", "g2", "g3"},
				},
				{
					Name: "Verify gang PodGroup condition is set to Unschedulable",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonUnschedulable,
					},
				},
			},
		},
		{
			name: "gang is scheduled before newer, equal priority standalone pods",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods are in active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3"},
				},
				{
					Name: "Create equal priority standalone pods",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
						st.MakePod().Name("s2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3", "s1", "s2"},
				},
				{
					Name:           "Schedule older gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify older gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"g1", "g2", "g3"},
				},
				{
					Name:           "Schedule first standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify first standalone pod is scheduled successfully",
					WaitForPodsScheduled: []string{"g1", "g2", "g3", "s1"},
				},
				{
					Name:           "Attempt scheduling second standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify second newer standalone pod cannot fit in remaining capacity",
					WaitForPodsUnschedulable: []string{"s2"},
				},
			},
		},
		{
			name: "standalone pod schedules immediately while gang is gated at PreEnqueue awaiting quorum",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create gang pod group requiring 3 pods",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create subset of gang pods (no quorum)",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                               "Verify incomplete gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"g1", "g2"},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"s1"},
				},
				{
					Name:           "Schedule standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod schedules immediately without being blocked",
					WaitForPodsScheduled: []string{"s1"},
				},
				{
					Name: "Create the 3rd gang pod completing quorum",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods move to active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3"},
				},
				{
					Name:           "Attempt scheduling gang",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify gang unblocks from PreEnqueue but becomes unschedulable due to remaining capacity",
					WaitForPodsUnschedulable: []string{"g1", "g2", "g3"},
				},
			},
		},
		{
			name: "standalone pod schedules immediately while gang is gated at PreEnqueue awaiting quorum, all schedulable",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create gang pod group requiring 3 pods",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create subset of gang pods (no quorum)",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                               "Verify incomplete gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"g1", "g2"},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"s1"},
				},
				{
					Name:           "Schedule standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod schedules immediately without being blocked",
					WaitForPodsScheduled: []string{"s1"},
				},
				{
					Name: "Create the 3rd gang pod completing quorum",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods move to active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3"},
				},
				{
					Name:           "Schedule gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify all pods are scheduled successfully",
					WaitForPodsScheduled: []string{"s1", "g1", "g2", "g3"},
				},
			},
		},
		{
			name: "low priority standalone pod schedules immediately while high priority gang is gated at PreEnqueue awaiting quorum, preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create gang pod group requiring 3 pods",
					CreatePodGroup: st.MakePodGroup().Name("high-pg").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create subset of gang pods (no quorum)",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("high-pg").Priority(100).Obj(),
						st.MakePod().Name("high-g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("high-pg").Priority(100).Obj(),
					},
				},
				{
					Name:                               "Verify incomplete gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"high-g1", "high-g2"},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("low-s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").Priority(10).ZeroTerminationGracePeriod().Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"low-s1"},
				},
				{
					Name:           "Schedule standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod schedules immediately without being blocked",
					WaitForPodsScheduled: []string{"low-s1"},
				},
				{
					Name: "Create the 3rd gang pod completing quorum",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("high-pg").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods move to active queue",
					WaitForPodsInActiveQ: []string{"high-g1", "high-g2", "high-g3"},
				},
				{
					Name:           "Attempt scheduling gang",
					RunScheduleOne: true,
				},
				{
					Name:               "Verify low priority standalone pod is preempted by the high priority gang",
					WaitForPodsRemoved: []string{"low-s1"},
				},
				{
					Name:           "Attempt scheduling gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify gang pods are scheduled after preemption",
					WaitForPodsScheduled: []string{"high-g1", "high-g2", "high-g3"},
				},
			},
		},
		{
			name: "low priority standalone pod schedules immediately while high priority gang is gated at PreEnqueue awaiting quorum, all schedulable",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:           "Create gang pod group requiring 3 pods",
					CreatePodGroup: st.MakePodGroup().Name("pg-high").Priority(100).MinCount(3).Obj(),
				},
				{
					Name: "Create subset of gang pods (no quorum)",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
						st.MakePod().Name("high-g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
					},
				},
				{
					Name:                               "Verify incomplete gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"high-g1", "high-g2"},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("low-s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"low-s1"},
				},
				{
					Name:           "Schedule standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod schedules immediately without being blocked",
					WaitForPodsScheduled: []string{"low-s1"},
				},
				{
					Name: "Create the 3rd gang pod completing quorum",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-high").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods move to active queue",
					WaitForPodsInActiveQ: []string{"high-g1", "high-g2", "high-g3"},
				},
				{
					Name:           "Schedule gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify all pods are scheduled successfully",
					WaitForPodsScheduled: []string{"low-s1", "high-g1", "high-g2", "high-g3"},
				},
			},
		},
		{
			name: "gang fails placement atomically without blocking standalone pod",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
					},
				},
				{
					Name:           "Create oversized gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods are in active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3"},
				},
				{
					Name: "Create standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2", "g3", "s1"},
				},
				{
					Name:           "Schedule gang",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify oversized gang failed scheduling atomically and is unschedulable",
					WaitForPodsUnschedulable: []string{"g1", "g2", "g3"},
				},
				{
					Name:           "Schedule standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod is scheduled successfully",
					WaitForPodsScheduled: []string{"s1"},
				},
			},
		},
		{
			name: "standalone pod fails scheduling without blocking lower priority gang",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
					},
				},
				{
					Name: "Create high priority oversized standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("high-s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"high-s1"},
				},
				{
					Name:           "Create low priority gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg-low").Priority(10).MinCount(3).Obj(),
				},
				{
					Name: "Create low priority gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("low-g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
						st.MakePod().Name("low-g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
						st.MakePod().Name("low-g3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg-low").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"high-s1", "low-g1", "low-g2", "low-g3"},
				},
				{
					Name:           "Schedule high priority standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify oversized standalone pod failed scheduling and is unschedulable",
					WaitForPodsUnschedulable: []string{"high-s1"},
				},
				{
					Name:           "Schedule low priority gang",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify low priority gang is scheduled successfully",
					WaitForPodsScheduled: []string{"low-g1", "low-g2", "low-g3"},
				},
			},
		},
		{
			name: "basic policy pod group allows partial scheduling interleaved with standalone pod",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name: "Create first standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify first standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"s1"},
				},
				{
					Name:           "Create basic policy pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(10).BasicPolicy().Obj(),
				},
				{
					Name: "Create basic group pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("b1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("b2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
						st.MakePod().Name("b3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify basic group pods are in active queue",
					WaitForPodsInActiveQ: []string{"s1", "b1", "b2", "b3"},
				},
				{
					Name: "Create second standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("s2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
					},
				},
				{
					Name:                 "Verify all pods are in active queue",
					WaitForPodsInActiveQ: []string{"s1", "b1", "b2", "b3", "s2"},
				},
				{
					Name:           "Schedule first standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify standalone pod is scheduled",
					WaitForPodsScheduled: []string{"s1"},
				},
				{
					Name:           "Schedule basic policy group",
					RunScheduleOne: true,
				},
				{
					Name: "Verify 1 basic pod is scheduled and 2 basic pods are unschedulable",
					WaitForAnyPodsScheduled: &stepsframework.WaitForAnyPodsScheduled{
						Pods: []*v1.Pod{
							st.MakePod().Name("b1").Obj(),
							st.MakePod().Name("b2").Obj(),
							st.MakePod().Name("b3").Obj(),
						},
						NumScheduled:     1,
						NumUnschedulable: 2,
					},
				},
				{
					Name:           "Attempt scheduling basic policy group (queued with an old timestamp to try preemption)",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify second standalone pod is still in activeQ",
					WaitForPodsInActiveQ: []string{"s2"},
				},
				{
					Name:           "Schedule second standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify second standalone pod is scheduled",
					WaitForPodsScheduled: []string{"s1", "s2"},
				},
			},
		},
		{
			name: "scheduling-gated standalone pod does not block gang scheduling, then becomes unschedulable when ungated",
			steps: []stepsframework.Step{
				{
					Name: "Create node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
					},
				},
				{
					Name: "Create scheduling-gated standalone pod",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("p-gated").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").SchedulingGates([]string{"test-gate"}).Priority(100).Obj(),
					},
				},
				{
					Name:                               "Verify standalone pod is gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"p-gated"},
				},
				{
					Name:           "Create gang pod group",
					CreatePodGroup: st.MakePodGroup().Name("pg").Priority(100).MinCount(2).Obj(),
				},
				{
					Name: "Create gang pods",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("g1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(100).Obj(),
						st.MakePod().Name("g2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify gang pods are in active queue",
					WaitForPodsInActiveQ: []string{"g1", "g2"},
				},
				{
					Name:           "Schedule gang pods",
					RunScheduleOne: true,
				},
				{
					Name:                 "Verify gang pods schedule while standalone pod remains gated",
					WaitForPodsScheduled: []string{"g1", "g2"},
				},
				{
					Name: "Remove scheduling gate from standalone pod",
					UpdatePod: &stepsframework.UpdatePod{
						PodName: "p-gated",
						ModifyFn: func(p *v1.Pod) {
							p.Spec.SchedulingGates = nil
						},
					},
				},
				{
					Name:                 "Verify un-gated standalone pod is in active queue",
					WaitForPodsInActiveQ: []string{"p-gated"},
				},
				{
					Name:           "Attempt scheduling un-gated standalone pod",
					RunScheduleOne: true,
				},
				{
					Name:                     "Verify standalone pod unblocks and becomes unschedulable (node is full)",
					WaitForPodsUnschedulable: []string{"p-gated"},
				},
			},
		},
	}

	for _, tt := range tests {
		for _, tasEnabled := range []bool{true, false} {
			for _, cpgEnabled := range []bool{true, false} {
				if !tasEnabled && cpgEnabled {
					// Cannot happen, skip.
					continue
				}
				t.Run(fmt.Sprintf("%s (TopologyAwareWorkloadScheduling: %v, CompositePodGroup: %v)", tt.name, tasEnabled, cpgEnabled), func(t *testing.T) {
					featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
						features.GenericWorkload:                 true,
						features.TopologyAwareWorkloadScheduling: tasEnabled,
						features.CompositePodGroup:               cpgEnabled,
					})

					testCtx := testutils.InitTestSchedulerWithOptions(
						t,
						testutils.InitTestAPIServer(t, "pg-interleaving", nil),
						0,
						scheduler.WithPodInitialBackoffSeconds(0),
						scheduler.WithPodMaxBackoffSeconds(0),
					)
					testutils.SyncSchedulerInformerFactory(testCtx)

					ns := testCtx.NS.Name

					if err := stepsframework.RunSteps(testCtx, t, ns, tt.steps); err != nil {
						t.Fatal(err)
					}
				})
			}
		}
	}
}
