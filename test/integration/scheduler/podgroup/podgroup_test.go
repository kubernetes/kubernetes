/*
Copyright 2025 The Kubernetes Authors.

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

package podgroup

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	configv1 "k8s.io/kube-scheduler/config/v1"
	framework "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

func TestPodGroupScheduling(t *testing.T) {
	node := st.MakeNode().Name("node").Label("topology.kubernetes.io/zone", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t2").BasicPolicy().Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-mid").MinCount(2).Obj()).
		Obj()
	otherWorkload := st.MakeWorkload().Name("other-workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t").MinCount(3).Obj()).
		Obj()

	gangPodGroup := st.MakePodGroup().Name("pg1").TemplateRef("t1", "workload").
		Priority(100).MinCount(3).Obj()

	otherGangPodGroup := st.MakePodGroup().Name("pg2").TemplateRef("t", "other-workload").
		Priority(100).MinCount(3).Obj()

	basicPodGroup := st.MakePodGroup().Name("pg1").TemplateRef("t2", "workload").BasicPolicy().Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p3 := st.MakePod().Name("p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()
	p4 := st.MakePod().Name("p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg1").Priority(100).Obj()

	blockerPod := st.MakePod().Name("blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Priority(100).Obj()
	smallBlockerPod := st.MakePod().Name("small-blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(100).Obj()
	lowPriorityBlockerPod := st.MakePod().Name("low-priority-blocker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()

	lowP1 := st.MakePod().Name("low-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP2 := st.MakePod().Name("low-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP3 := st.MakePod().Name("low-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()
	lowP4 := st.MakePod().Name("low-p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(10).Obj()

	veryLowP1 := st.MakePod().Name("very-low-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(5).Obj()
	veryLowP2 := st.MakePod().Name("very-low-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		ZeroTerminationGracePeriod().Priority(5).Obj()
	midP1 := st.MakePod().Name("mid-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("mid-pg").Priority(50).Obj()
	midP2 := st.MakePod().Name("mid-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("mid-pg").Priority(50).Obj()

	midPodGroup := st.MakePodGroup().Name("mid-pg").TemplateRef("t-mid", "workload").
		Priority(50).MinCount(2).Obj()
	midPodGroupWithConstraint := st.MakePodGroup().Name("mid-pg").TemplateRef("t-mid", "workload").
		Priority(50).MinCount(2).TopologyKey("topology.kubernetes.io/zone").Obj()

	otherP1 := st.MakePod().Name("other-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP2 := st.MakePod().Name("other-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP3 := st.MakePod().Name("other-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()

	tests := []struct {
		name                          string
		enableWorkloadAwarePreemption bool
		requiresTAS                   bool
		steps                         []stepsframework.Step
	}{
		{
			name: "gang schedules when pod group and resources are available",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:       "Create all pods belonging to the gang",
					CreatePods: []*v1.Pod{p1, p2, p3},
				},
				{
					Name:                 "Verify all gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify PodGroup condition is set to Scheduled",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg1",
						ConditionStatus: metav1.ConditionTrue,
						Reason:          "Scheduled",
					},
				},
			},
		},
		{
			name: "gang waits for quorum to start, then schedules",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:       "Create subset of pods belonging to the gang",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                         "Verify pods are gated at PreEnqueue (no quorum)",
					WaitForPodsGatedOnPreEnqueue: []string{"p1", "p2"},
				},
				{
					Name:       "Create the last pod belonging to the gang to unblock PreEnqueue",
					CreatePods: []*v1.Pod{p3},
				},
				{
					Name:                 "Verify all gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "gang waits for pod group, then for resources, then schedules",
			steps: []stepsframework.Step{
				{
					Name:       "Create the resource-blocking pod",
					CreatePods: []*v1.Pod{blockerPod},
				},
				{
					Name:                 "Schedule the resource-blocking pod",
					WaitForPodsScheduled: []string{"blocker"},
				},
				{
					Name:       "Create gang pods before PodGroup is created",
					CreatePods: []*v1.Pod{p1, p2, p3},
				},
				{
					Name:                         "Verify pods are gated at PreEnqueue (no PodGroup object)",
					WaitForPodsGatedOnPreEnqueue: []string{"p1", "p2", "p3"},
				},
				{
					Name:           "Create the PodGroup to unblock PreEnqueue",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:                     "Verify pods become unschedulable (Permit timeout due to resource blocker)",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify PodGroup condition is set to Unschedulable",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg1",
						ConditionStatus: metav1.ConditionFalse,
						Reason:          schedulingapi.PodGroupReasonUnschedulable,
					},
				},
				{
					Name:       "Delete the resource-blocking pod",
					DeletePods: []string{"blocker"},
				},
				{
					Name:                 "Verify the entire gang is now scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify PodGroup condition transitions to Scheduled",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg1",
						ConditionStatus: metav1.ConditionTrue,
						Reason:          "Scheduled",
					},
				},
			},
		},
		{
			name: "minCount is scheduled, but one pod from a gang remain unschedulable until the blocked resources are released",
			steps: []stepsframework.Step{
				{
					Name:       "Create the resource-blocking pod",
					CreatePods: []*v1.Pod{smallBlockerPod},
				},
				{
					Name:                 "Schedule the resource-blocking pod",
					WaitForPodsScheduled: []string{"small-blocker"},
				},
				{
					Name:       "Create all pods belonging to the gang (more than minCount) before the PodGroup is created",
					CreatePods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					Name:           "Create the PodGroup to unblock PreEnqueue",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name: "Verify minCount pods is scheduled successfully and one becomes unschedulable (resource-blocking pod is blocking the space)",
					WaitForAnyPodsScheduled: &stepsframework.WaitForAnyPodsScheduled{
						Pods:             []*v1.Pod{p1, p2, p3, p4},
						NumScheduled:     3,
						NumUnschedulable: 1,
					},
				},
				{
					Name:       "Delete the resource-blocking pod",
					DeletePods: []string{"small-blocker"},
				},
				{
					Name:                 "Verify the entire gang is now scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
			},
		},
		{
			name: "two gangs competing for the same resources shouldn't deadlock, reversed order",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:           "Create the other PodGroup object",
					CreatePodGroup: otherGangPodGroup,
				},
				{
					Name:       "Create pods from both gangs",
					CreatePods: []*v1.Pod{otherP3, p3, otherP2, p2, otherP1, p1},
				},
				{
					Name:                 "Verify the entire other gang is now scheduled",
					WaitForPodsScheduled: []string{"other-p1", "other-p2", "other-p3"},
				},
				{
					Name:                     "Verify the entire gang becomes unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "two gangs competing for the same resources shouldn't deadlock",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:           "Create the other PodGroup object",
					CreatePodGroup: otherGangPodGroup,
				},
				{
					Name:       "Create pods from both gangs",
					CreatePods: []*v1.Pod{p1, otherP1, p2, otherP2, p3, otherP3},
				},
				{
					Name:                 "Verify the entire gang is now scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:                     "Verify the entire other gang becomes unschedulable",
					WaitForPodsUnschedulable: []string{"other-p1", "other-p2", "other-p3"},
				},
			},
		},
		{
			name: "gang schedules with preemption",
			steps: []stepsframework.Step{
				{
					Name:       "Create a low priority pod taking all resources",
					CreatePods: []*v1.Pod{lowPriorityBlockerPod},
				},
				{
					Name:                 "Schedule the low priority resource-blocking pod",
					WaitForPodsScheduled: []string{"low-priority-blocker"},
				},
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:       "Create high priority gang pods",
					CreatePods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					Name:                 "Verify all gang pods are scheduled successfully (after preemption)",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify PodGroup condition is set to Scheduled after preemption completes",
					WaitForPodGroupCondition: &stepsframework.PodGroupConditionCheck{
						PodGroupName:    "pg1",
						ConditionStatus: metav1.ConditionTrue,
						Reason:          "Scheduled",
					},
				},
				{
					Name:               "Verify preemption victims were removed",
					WaitForPodsRemoved: []string{"low-priority-blocker"},
				},
			},
		},
		{
			name: "basic group schedules when pod group and resources are available, without gang enforcement",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: basicPodGroup,
				},
				{
					Name:       "Create one pod belonging to the group",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                 "Verify group's pod is scheduled successfully",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name:       "Create another pods belonging to the group",
					CreatePods: []*v1.Pod{p2, p3},
				},
				{
					Name:                 "Verify group's pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "basic group waits for pod group, part of it waits for resources, then schedules",
			steps: []stepsframework.Step{
				{
					Name:       "Create the resource-blocking pod",
					CreatePods: []*v1.Pod{blockerPod},
				},
				{
					Name:                 "Schedule the resource-blocking pod",
					WaitForPodsScheduled: []string{"blocker"},
				},
				{
					Name:       "Create basic group pods before PodGroup is created",
					CreatePods: []*v1.Pod{p1, p2, p3},
				},
				{
					Name:                         "Verify pods are gated at PreEnqueue (no PodGroup object)",
					WaitForPodsGatedOnPreEnqueue: []string{"p1", "p2", "p3"},
				},
				{
					Name:           "Create the PodGroup to unblock PreEnqueue",
					CreatePodGroup: basicPodGroup,
				},
				{
					Name: "Verify two pods are scheduled successfully and one becomes unschedulable (resource-blocking pod is blocking the space)",
					WaitForAnyPodsScheduled: &stepsframework.WaitForAnyPodsScheduled{
						Pods:             []*v1.Pod{p1, p2, p3},
						NumScheduled:     2,
						NumUnschedulable: 1,
					},
				},
				{
					Name:       "Delete the resource-blocking pod",
					DeletePods: []string{"blocker"},
				},
				{
					Name:                 "Verify the entire group is now scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "basic group schedules with preemption",
			steps: []stepsframework.Step{
				{
					Name:       "Create a low priority pod taking all resources",
					CreatePods: []*v1.Pod{lowPriorityBlockerPod},
				},
				{
					Name:                 "Schedule the low priority resource-blocking pod",
					WaitForPodsScheduled: []string{"low-priority-blocker"},
				},
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: basicPodGroup,
				},
				{
					Name:       "Create high priority group's pods",
					CreatePods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					Name:                 "Verify all group's pods are scheduled successfully (after preemption)",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify preemption victims were removed",
					WaitForPodsRemoved: []string{"low-priority-blocker"},
				},
			},
		},
		{
			name: "basic group schedules with workload-aware preemption",
			steps: []stepsframework.Step{
				{
					Name:       "Create a low priority pod taking all resources",
					CreatePods: []*v1.Pod{lowPriorityBlockerPod},
				},
				{
					Name:                 "Schedule the low priority resource-blocking pod",
					WaitForPodsScheduled: []string{"low-priority-blocker"},
				},
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: basicPodGroup,
				},
				{
					Name:       "Create high priority group's pods",
					CreatePods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					Name:                 "Verify all group's pods are scheduled successfully (after preemption)",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify preemption victims were removed",
					WaitForPodsRemoved: []string{"low-priority-blocker"},
				},
			},
		},
		{
			name:                          "gang schedules with workload-aware preemption",
			enableWorkloadAwarePreemption: true,
			steps: []stepsframework.Step{
				{
					Name:       "Create low priority pods that take up all node resources",
					CreatePods: []*v1.Pod{lowP1, lowP2, lowP3, lowP4},
				},
				{
					Name:                 "Wait for all low priority pods to be scheduled",
					WaitForPodsScheduled: []string{"low-p1", "low-p2", "low-p3", "low-p4"},
				},
				{
					Name:           "Create the Workload object",
					CreatePodGroup: gangPodGroup,
				},
				{
					Name:       "Create high priority gang pods",
					CreatePods: []*v1.Pod{p1, p2, p3, p4},
				},
				{
					Name:                 "Verify all gang pods are scheduled successfully (after workload-aware preemption)",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify preemption victims were removed",
					WaitForPodsRemoved: []string{"low-p1", "low-p2", "low-p3", "low-p4"},
				},
			},
		},
		{
			name:                          "gang schedules with partial workload-aware preemption",
			enableWorkloadAwarePreemption: true,
			steps: []stepsframework.Step{
				{
					Name:       "Create very low and low priority pods that take up all node resources",
					CreatePods: []*v1.Pod{veryLowP1, veryLowP2, lowP1, lowP2},
				},
				{
					Name:                 "Wait for all very low and low priority pods to be scheduled",
					WaitForPodsScheduled: []string{"very-low-p1", "very-low-p2", "low-p1", "low-p2"},
				},
				{
					Name:           "Create the mid PodGroup object",
					CreatePodGroup: midPodGroup,
				},
				{
					Name:       "Create mid priority gang pods",
					CreatePods: []*v1.Pod{midP1, midP2},
				},
				{
					Name:                 "Verify mid priority pods and low priority pods are scheduled",
					WaitForPodsScheduled: []string{"mid-p1", "mid-p2", "low-p1", "low-p2"},
				},
				{
					Name:               "Verify very low priority preemption victims were removed",
					WaitForPodsRemoved: []string{"very-low-p1", "very-low-p2"},
				},
			},
		},
		{
			name:        "tas gang with constraint does not use pod by pod preemption",
			requiresTAS: true,
			steps: []stepsframework.Step{
				{
					Name:       "Create very low and low priority pods that take up all node resources",
					CreatePods: []*v1.Pod{veryLowP1, veryLowP2, lowP1, lowP2},
				},
				{
					Name:                 "Wait for all very low and low priority pods to be scheduled",
					WaitForPodsScheduled: []string{"very-low-p1", "very-low-p2", "low-p1", "low-p2"},
				},
				{
					Name:           "Create the mid PodGroup object with constraint",
					CreatePodGroup: midPodGroupWithConstraint,
				},
				{
					Name:       "Create mid priority gang pods",
					CreatePods: []*v1.Pod{midP1, midP2},
				},
				{
					Name:                     "Verify the entire gang becomes unschedulable",
					WaitForPodsUnschedulable: []string{"mid-p1", "mid-p2"},
				},
			},
		},
		{
			name:                          "tas gang with constraint does not use workload preemption",
			enableWorkloadAwarePreemption: true,
			requiresTAS:                   true,
			steps: []stepsframework.Step{
				{
					Name:       "Create very low and low priority pods that take up all node resources",
					CreatePods: []*v1.Pod{veryLowP1, veryLowP2, lowP1, lowP2},
				},
				{
					Name:                 "Wait for all very low and low priority pods to be scheduled",
					WaitForPodsScheduled: []string{"very-low-p1", "very-low-p2", "low-p1", "low-p2"},
				},
				{
					Name:           "Create the mid PodGroup object with constraint",
					CreatePodGroup: midPodGroupWithConstraint,
				},
				{
					Name:       "Create mid priority gang pods",
					CreatePods: []*v1.Pod{midP1, midP2},
				},
				{
					Name:                     "Verify the entire gang becomes unschedulable",
					WaitForPodsUnschedulable: []string{"mid-p1", "mid-p2"},
				},
			},
		},
	}

	for _, tt := range tests {
		runWithTasFlagSettings := []bool{true, false}
		for _, tasEnabled := range runWithTasFlagSettings {
			if tt.requiresTAS && !tasEnabled {
				continue
			}
			t.Run(fmt.Sprintf("%s (TopologyAwareWorkloadScheduling enabled: %v)", tt.name, tasEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.GenericWorkload:                 true,
					features.GangScheduling:                  true,
					features.TopologyAwareWorkloadScheduling: tasEnabled,
					features.WorkloadAwarePreemption:         tt.enableWorkloadAwarePreemption,
				})

				testCtx := testutils.InitTestSchedulerWithNS(t, "podgroup-scheduling",
					// disable backoff
					scheduler.WithPodMaxBackoffSeconds(0),
					scheduler.WithPodInitialBackoffSeconds(0))

				ns := testCtx.NS.Name

				commonSteps := []stepsframework.Step{
					{
						Name:        "Create Nodes",
						CreateNodes: []*v1.Node{node},
					},
					{
						Name:            "Create workloads",
						CreateWorkloads: []*schedulingapi.Workload{workload, otherWorkload},
					},
				}

				if err := stepsframework.RunSteps(testCtx, ns, append(commonSteps, tt.steps...)); err != nil {
					t.Fatal(err)
				}
			})
		}
	}
}

func TestWorkloadAwarePreemptionInvocation(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:         true,
		features.GangScheduling:          true,
		features.WorkloadAwarePreemption: true,
	})

	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).Obj()
	pg := st.MakePodGroup().Namespace("default").Name("pg1").TemplateRef("t1", "workload").
		DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(100).MinCount(3).Obj()

	// Low priority pods taking up all resources
	lowPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
	}

	// High priority pods belonging to a group
	highPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
	}

	testCtx := testutils.InitTestSchedulerWithNS(t, "wap-inv",
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0))
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 1. Create low priority pods
	for _, p := range lowPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// Wait for low priority pods to be scheduled
	for _, p := range lowPods {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodScheduled(cs, ns, p.Name)); err != nil {
			t.Fatalf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
		}
	}

	// 2. Create workload
	if _, err := cs.SchedulingV1alpha2().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	// 3. Create PodGroup
	pg.Namespace = ns
	if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create PodGroup: %v", err)
	}

	// 4. Create high priority pods
	for _, p := range highPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// 5. Verify that WorkloadAwarePreemption was called
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		for _, pod := range lowPods {
			events, err := cs.CoreV1().Events(ns).List(ctx, metav1.ListOptions{
				FieldSelector: "involvedObject.name=" + pod.Name,
			})
			if err != nil {
				return false, err
			}
			for _, event := range events.Items {
				if event.Reason == "Preempted" && strings.HasPrefix(event.Message, "Preempted by podgroup") {
					return true, nil
				}
			}
		}
		return false, nil
	})
	if err != nil {
		t.Errorf("WorkloadAwarePreemption was not called within timeout")
	}

	t.Logf("WorkloadAwarePreemption was called (verified via events)")
}

// mockPostFilterPlugin is a custom PostFilter plugin that just counts invocations.
type mockPostFilterPlugin struct {
	count int
}

func (m *mockPostFilterPlugin) Name() string {
	return "MockPostFilter"
}

func (m *mockPostFilterPlugin) PostFilter(ctx context.Context, state framework.CycleState, pod *v1.Pod, filteredNodeStatusMap framework.NodeToStatusReader) (*framework.PostFilterResult, *framework.Status) {
	m.count++
	return nil, framework.NewStatus(framework.Unschedulable)
}

func TestPostFilterInvocationCount(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:         true,
		features.GangScheduling:          true,
		features.WorkloadAwarePreemption: true,
	})

	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).Obj()
	pg := st.MakePodGroup().Namespace("default").Name("pg1").TemplateRef("t1", "workload").
		DisruptionMode(schedulingapi.DisruptionModePodGroup).Priority(100).MinCount(3).Obj()

	// Low priority pods taking up all resources
	lowPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("low-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
		st.MakePod().Namespace("default").Name("low-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").Priority(10).Obj(),
	}

	// High priority pods belonging to a group
	highPods := []*v1.Pod{
		st.MakePod().Namespace("default").Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
		st.MakePod().Namespace("default").Name("high-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
	}

	mockPlugin := &mockPostFilterPlugin{}
	registry := frameworkruntime.Registry{
		"MockPostFilter": func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
			return mockPlugin, nil
		},
	}

	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &configv1.Plugins{
				PostFilter: configv1.PluginSet{
					Enabled: []configv1.Plugin{
						{Name: "MockPostFilter"},
						{Name: "DefaultPreemption"},
					},
				},
			},
		}},
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "post-filter-count",
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithProfiles(cfg.Profiles...),
	)
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	_, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node: %v", err)
	}

	// 1. Create low priority pods
	for _, p := range lowPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// Wait for low priority pods to be scheduled
	for _, p := range lowPods {
		if err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
			testutils.PodScheduled(cs, ns, p.Name)); err != nil {
			t.Fatalf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
		}
	}

	// 2. Create workload
	if _, err := cs.SchedulingV1alpha2().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create workload: %v", err)
	}

	// 3. Create PodGroup
	pg.Namespace = ns
	if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create PodGroup: %v", err)
	}

	// 4. Create high priority pods
	for _, p := range highPods {
		p.Namespace = ns
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Failed to create pod %s: %v", p.Name, err)
		}
	}

	// 5. Verify that MockPostFilter was called exactly 3 times
	// It should be called for each pod from pod group in pod group cycle
	// but should not be called in WAP.
	err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		if mockPlugin.count == 3 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Errorf("MockPostFilter was called %d times, expected exactly 3", mockPlugin.count)
	}
}
