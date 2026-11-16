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
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	configv1 "k8s.io/kube-scheduler/config/v1"
	framework "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

func TestPodGroupScheduling(t *testing.T) {
	node := st.MakeNode().Name("node").Label("topology.kubernetes.io/zone", "zone1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

	workload := st.MakeWorkload().Name("workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-gang-2").MinCount(2).Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-gang-3").MinCount(3).Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-gang-4").MinCount(4).Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-gang-5").MinCount(5).Obj()).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t-basic").BasicPolicy().Obj()).
		Obj()
	otherWorkload := st.MakeWorkload().Name("other-workload").
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t").MinCount(3).Obj()).
		Obj()

	gangPodGroup2 := st.MakePodGroup().Name("pg1").WorkloadRef("workload", "t-gang-2").
		Priority(100).MinCount(2).Obj()
	gangPodGroup3 := st.MakePodGroup().Name("pg1").WorkloadRef("workload", "t-gang-3").
		Priority(100).MinCount(3).Obj()
	gangPodGroup4 := st.MakePodGroup().Name("pg1").WorkloadRef("workload", "t-gang-4").
		Priority(100).MinCount(4).Obj()
	gangPodGroup5 := st.MakePodGroup().Name("pg1").WorkloadRef("workload", "t-gang-5").
		Priority(100).MinCount(5).Obj()

	otherGangPodGroup := st.MakePodGroup().Name("pg2").WorkloadRef("other-workload", "t").
		Priority(100).MinCount(3).Obj()

	basicPodGroup := st.MakePodGroup().Name("pg1").WorkloadRef("workload", "t-basic").Priority(100).BasicPolicy().Obj()

	preferredNodeAffinity := func(key, val string) *v1.NodeAffinity {
		return &v1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
				{
					Weight: 100,
					Preference: v1.NodeSelectorTerm{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      key,
								Operator: v1.NodeSelectorOpIn,
								Values:   []string{val},
							},
						},
					},
				},
			},
		}
	}

	mutP1 := st.MakePod().Name("mut-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()
	mutP2 := st.MakePod().Name("mut-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()
	mutP3 := st.MakePod().Name("mut-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()
	mutP4 := st.MakePod().Name("mut-p4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()
	mutP5 := st.MakePod().Name("mut-p5").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj()

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

	midPodGroup := st.MakePodGroup().Name("mid-pg").WorkloadRef("workload", "t-gang-2").
		Priority(50).MinCount(2).Obj()
	midPodGroupWithConstraint := st.MakePodGroup().Name("mid-pg").WorkloadRef("workload", "t-gang-2").
		Priority(50).MinCount(2).TopologyKey("topology.kubernetes.io/zone").Obj()

	otherP1 := st.MakePod().Name("other-p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP2 := st.MakePod().Name("other-p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()
	otherP3 := st.MakePod().Name("other-p3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName("pg2").Priority(100).Obj()

	tests := []struct {
		name                                  string
		enableTopologyAwareWorkloadScheduling []bool
		nodes                                 []*v1.Node
		steps                                 []stepsframework.Step
	}{
		{
			name: "gang schedules when pod group and resources are available",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup3,
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
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
			},
		},
		{
			name: "any available pod group members get scheduled",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup3,
				},
				{
					Name:       "Create initial subset of pods satisfying minCount",
					CreatePods: []*v1.Pod{p1, p2, p3},
				},
				{
					Name:                 "Verify initial gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:       "Create additional pod belonging to the gang",
					CreatePods: []*v1.Pod{p4},
				},
				{
					Name:                 "Verify all gang pods are scheduled successfully",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
			},
		},
		{
			name: "gang waits for quorum to start, then schedules",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup3,
				},
				{
					Name:       "Create subset of pods belonging to the gang",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                               "Verify pods are gated at PreEnqueue (no quorum)",
					WaitForPodsInUnschedulableEntities: []string{"p1", "p2"},
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
					Name:                                "Verify pods are waiting in incompletePodGroupPods (no PodGroup object)",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2", "p3"},
				},
				{
					Name:           "Create the PodGroup to unblock the pods",
					CreatePodGroup: gangPodGroup3,
				},
				{
					Name:                     "Verify pods become unschedulable due to resource blocker pod",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify PodGroup condition is set to Unschedulable",
					WaitForGroupsUnschedulable: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
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
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
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
					Name:           "Create the PodGroup to unblock pods",
					CreatePodGroup: gangPodGroup3,
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
					CreatePodGroup: gangPodGroup3,
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
					CreatePodGroup: gangPodGroup3,
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
			name: "pod group interleaving with an individual pod shouldn't deadlock nor livelock, and the individual pod gets scheduled",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup3,
				},
				{
					Name:       "Create pods from gang and individual pod",
					CreatePods: []*v1.Pod{blockerPod, p1, p2, p3},
				},
				{
					Name:                 "Verify the individual pod is scheduled",
					WaitForPodsScheduled: []string{"blocker"},
				},
				{
					Name:                     "Verify the gang becomes unschedulable",
					WaitForPodsUnschedulable: []string{"p1", "p2", "p3"},
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
					Name:                                "Verify pods are waiting in incompletePodGroupPods (no PodGroup object)",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2", "p3"},
				},
				{
					Name:           "Create the PodGroup to unblock pods",
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
			name:                                  "basic group schedules with workload-aware preemption",
			enableTopologyAwareWorkloadScheduling: []bool{false},
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
			name: "gang schedules with workload-aware preemption",
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
					Name:           "Create the PodGroup object",
					CreatePodGroup: gangPodGroup3,
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
					Name: "Verify PodGroup condition is set to Scheduled after preemption completes",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name:               "Verify preemption victims were removed",
					WaitForPodsRemoved: []string{"low-p1", "low-p2", "low-p3", "low-p4"},
				},
			},
		},
		{
			name: "gang schedules with partial workload-aware preemption",
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
			name:                                  "tas gang with constraint uses workload preemption",
			enableTopologyAwareWorkloadScheduling: []bool{true},
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
			name: "gang pods are unschedulable due to lack of quorum, then scheduled when minCount is decreased",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object with minCount=5",
					CreatePodGroup: gangPodGroup5,
				},
				{
					Name:              "Create 4 pods belonging to the gang (quorum is 5)",
					CreatePodsInOrder: []*v1.Pod{mutP1, mutP2, mutP3, mutP4},
				},
				{
					Name:                               "Verify gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
				},
				{
					Name:           "Update the PodGroup with decreased minCount=4",
					UpdatePodGroup: (&st.PodGroupWrapper{PodGroup: *gangPodGroup5.DeepCopy()}).MinCount(4).Obj(),
				},
				{
					Name:                 "Verify all gang pods are immediately re-queued and scheduled successfully",
					WaitForPodsScheduled: []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
				},
			},
		},
		{
			name: "gang pods are unschedulable due to lack of space, then scheduled when minCount is decreased",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object with minCount=5",
					CreatePodGroup: gangPodGroup5,
				},
				{
					Name:              "Create 5 pods belonging to the gang",
					CreatePodsInOrder: []*v1.Pod{mutP1, mutP2, mutP3, mutP4, mutP5},
				},
				{
					Name:                     "Verify gang pods are unschedulable",
					WaitForPodsUnschedulable: []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4", "mut-p5"},
				},
				{
					Name:           "Update the PodGroup with decreased minCount=4",
					UpdatePodGroup: (&st.PodGroupWrapper{PodGroup: *gangPodGroup5.DeepCopy()}).MinCount(4).Obj(),
				},
				{
					Name:                 "Verify 4 gang pods are immediately re-queued and scheduled successfully",
					WaitForPodsScheduled: []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
				},
				{
					Name:                     "Verify the last gang pod is unschedulable",
					WaitForPodsUnschedulable: []string{"mut-p5"},
				},
			},
		},
		{
			name: "gang pods are unschedulable due to lack of quorum, and remain unschedulable without triggering reschedule when minCount is increased",
			steps: []stepsframework.Step{
				{
					Name:           "Create the PodGroup object with minCount=5",
					CreatePodGroup: gangPodGroup5,
				},
				{
					Name:              "Create 4 pods belonging to the gang (quorum is 5)",
					CreatePodsInOrder: []*v1.Pod{mutP1, mutP2, mutP3, mutP4},
				},
				{
					Name:                               "Verify gang pods are gated at PreEnqueue",
					WaitForPodsInUnschedulableEntities: []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
				},
				{
					Name: "Verify scheduling attempts of all gang pods is 0",
					VerifyPodSchedulingAttempts: &stepsframework.VerifyPodsSchedulingAttempts{
						PodNames:     []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
						PodGroupName: "pg1",
						Attempts:     0,
					},
				},
				{
					Name:           "Update the PodGroup with increased minCount=6",
					UpdatePodGroup: (&st.PodGroupWrapper{PodGroup: *gangPodGroup5.DeepCopy()}).MinCount(6).Obj(),
				},
				{
					Name: "Verify scheduling attempts of all gang pods is still 0 (did not trigger reschedule/requeue)",
					VerifyPodSchedulingAttempts: &stepsframework.VerifyPodsSchedulingAttempts{
						PodNames:     []string{"mut-p1", "mut-p2", "mut-p3", "mut-p4"},
						PodGroupName: "pg1",
						Attempts:     0,
					},
				},
			},
		},
		{
			name: "gang with heterogeneous resource requests schedules across multiple nodes",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
				st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=3",
					CreatePodGroup: gangPodGroup3,
				},
				// Order matters (KEP-4671 "Algorithm Limitations"): a worker evaluated first would take node1
				// under LeastAllocated, and the 3-CPU leader would then fit nowhere.
				{
					Name: "Create one large leader pod and two smaller worker pods in order",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("leader").Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("worker1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("worker2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify all gang pods schedule",
					WaitForPodsScheduled: []string{"leader", "worker1", "worker2"},
				},
				{
					Name: "Verify PodGroup condition is Scheduled",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify leader lands on node1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"leader"},
						Nodes: sets.New("node1"),
					},
				},
				{
					Name: "Verify workers land on node2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"worker1", "worker2"},
						Nodes: sets.New("node2"),
					},
				},
			},
		},
		{
			name: "failed gang with partially fitting heterogeneous pods does not block standalone pod and schedules when capacity is added",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=3",
					CreatePodGroup: gangPodGroup3,
				},
				{
					Name: "Create two 2-CPU pods that fit and one 6-CPU pod that does not",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("small1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("small2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("large").Req(map[v1.ResourceName]string{v1.ResourceCPU: "6"}).
							Tolerations([]v1.Toleration{{Key: "dedicated", Value: "large", Effect: v1.TaintEffectNoSchedule}}).
							Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                     "Verify all gang pods become unschedulable",
					WaitForPodsUnschedulable: []string{"small1", "small2", "large"},
				},
				{
					Name: "Verify PodGroup condition is Unschedulable",
					WaitForGroupsUnschedulable: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Create standalone pod to verify small1 and small2 did not reserve node capacity",
					CreatePods: []*v1.Pod{
						st.MakePod().Name("standalone").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").ZeroTerminationGracePeriod().Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify standalone pod schedules",
					WaitForPodsScheduled: []string{"standalone"},
				},
				{
					Name:       "Delete standalone pod",
					DeletePods: []string{"standalone"},
				},
				{
					// node2 is tainted so LeastAllocated does not place small1 on node2 when the gang retries.
					Name: "Add a 6-CPU tainted node so the large gang member can fit",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "6"}).
							Taints([]v1.Taint{{Key: "dedicated", Value: "large", Effect: v1.TaintEffectNoSchedule}}).Obj(),
					},
				},
				{
					Name:                 "Verify all gang pods schedule after node2 is added",
					WaitForPodsScheduled: []string{"small1", "small2", "large"},
				},
				{
					Name: "Verify PodGroup condition transitions to Scheduled",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify small pods are assigned to node1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"small1", "small2"},
						Nodes: sets.New("node1"),
					},
				},
				{
					Name: "Verify large pod is assigned to node2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"large"},
						Nodes: sets.New("node2"),
					},
				},
			},
		},
		{
			name: "gang with minCount less than total schedules fitting heterogeneous subset past an unschedulable pod",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
			},
			steps: []stepsframework.Step{
				// Create pods before the PodGroup so the group popped from the queue by the
				// scheduling cycle has all pods in the sequence set by CreatePodsInOrder.
				// p3-large sits between p1 and p2 to show that evaluation continues past a rejected pod.
				// Evaluated first, it would saturate node1 and leave no room for other pods.
				{
					Name: "Create 2-CPU, 4-CPU, and 1-CPU pods in order",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p3-large").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:           "Create gang PodGroup with minCount=2",
					CreatePodGroup: gangPodGroup2,
				},
				{
					Name:                 "Verify the two fitting pods schedule to satisfy minCount=2",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:                     "Verify the 4-CPU pod is marked unschedulable",
					WaitForPodsUnschedulable: []string{"p3-large"},
				},
				{
					Name: "Verify PodGroup condition is Scheduled although p3-large is unschedulable",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify p1 and p2 are assigned to node1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2"},
						Nodes: sets.New("node1"),
					},
				},
				{
					Name: "Add a second 4-CPU node",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:                 "Verify p3-large schedules on the new node",
					WaitForPodsScheduled: []string{"p1", "p2", "p3-large"},
				},
				{
					Name: "Verify p3-large is assigned to node2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p3-large"},
						Nodes: sets.New("node2"),
					},
				},
			},
		},
		{
			name: "basic policy schedules fitting heterogeneous pods past an unschedulable pod in the middle",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Obj(),
			},
			steps: []stepsframework.Step{
				// Create pods before the PodGroup so the group popped from the queue by the
				// scheduling cycle has all pods in the sequence set by CreatePodsInOrder.
				{
					Name: "Create 1-CPU, 4-CPU (unfittable), and 2-CPU pods in order",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("p-small").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p-large").Req(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p-med").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:           "Create Basic PodGroup",
					CreatePodGroup: basicPodGroup,
				},
				{
					Name:                 "Verify p-small and p-med schedule on node1 despite p-large failing in between",
					WaitForPodsScheduled: []string{"p-small", "p-med"},
				},
				{
					Name:                     "Verify p-large becomes unschedulable",
					WaitForPodsUnschedulable: []string{"p-large"},
				},
				{
					Name: "Verify PodGroup condition is Scheduled although p-large is unschedulable",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify p-small and p-med are assigned to node1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p-small", "p-med"},
						Nodes: sets.New("node1"),
					},
				},
				{
					Name: "Add a 4-CPU node for p-large",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:                 "Verify p-large schedules after node2 is added",
					WaitForPodsScheduled: []string{"p-small", "p-med", "p-large"},
				},
				{
					Name: "Verify p-large is assigned to node2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p-large"},
						Nodes: sets.New("node2"),
					},
				},
			},
		},
		{
			name: "gang with distinct node selectors and tolerations per pod schedules each pod onto its matching node",
			nodes: []*v1.Node{
				st.MakeNode().Name("node-cpu").Label("pool", "cpu").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
				st.MakeNode().Name("node-gpu").Label("pool", "gpu").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).
					Taints([]v1.Taint{{Key: "gpu", Value: "true", Effect: v1.TaintEffectNoSchedule}}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=4",
					CreatePodGroup: gangPodGroup4,
				},
				{
					Name: "Create pods targeting cpu, gpu (with toleration) and mem pools, plus one unconstrained pod",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("p-cpu").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).NodeSelector(map[string]string{"pool": "cpu"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p-gpu").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).NodeSelector(map[string]string{"pool": "gpu"}).
							Tolerations([]v1.Toleration{{Key: "gpu", Value: "true", Effect: v1.TaintEffectNoSchedule}}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("p-mem").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).NodeSelector(map[string]string{"pool": "mem"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						// LeastAllocated would pick the mostly empty node-gpu; only its taint keeps p-any off it.
						st.MakePod().Name("p-any").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                     "Verify all gang pods are unschedulable while node-mem is missing",
					WaitForPodsUnschedulable: []string{"p-cpu", "p-gpu", "p-mem", "p-any"},
				},
				{
					Name: "Verify PodGroup condition is Unschedulable",
					WaitForGroupsUnschedulable: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Add node-mem matching p-mem's selector",
					CreateNodes: []*v1.Node{
						st.MakeNode().Name("node-mem").Label("pool", "mem").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
					},
				},
				{
					Name:                 "Verify all gang pods schedule",
					WaitForPodsScheduled: []string{"p-cpu", "p-gpu", "p-mem", "p-any"},
				},
				{
					Name: "Verify PodGroup condition transitions to Scheduled",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify p-cpu is assigned to node-cpu",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p-cpu"},
						Nodes: sets.New("node-cpu"),
					},
				},
				{
					Name: "Verify p-gpu is assigned to node-gpu",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p-gpu"},
						Nodes: sets.New("node-gpu"),
					},
				},
				{
					Name: "Verify p-mem and the unconstrained p-any are assigned to node-mem",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p-mem", "p-any"},
						Nodes: sets.New("node-mem"),
					},
				},
			},
		},
		{
			// Documents a known limitation of the sequential algorithm (KEP-4671, "Algorithm Limitations"):
			// flex prefers node2, so evaluating it first leaves no node for strict, although a placement exists.
			name: "flexible pod evaluated before constrained pod leaves the gang unschedulable",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
				st.MakeNode().Name("node2").Label("special", "true").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=2",
					CreatePodGroup: gangPodGroup2,
				},
				{
					Name: "Enqueue flexible pod (prefers special=true) before constrained pod (requires special=true)",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("flex").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
							NodeAffinity(preferredNodeAffinity("special", "true")).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("strict").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
							NodeSelector(map[string]string{"special": "true"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                     "Verify the gang is unschedulable",
					WaitForPodsUnschedulable: []string{"flex", "strict"},
				},
				{
					Name: "Verify PodGroup condition is Unschedulable",
					WaitForGroupsUnschedulable: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
			},
		},
		{
			name: "constrained pod evaluated before flexible pod schedules the gang",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
				st.MakeNode().Name("node2").Label("special", "true").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=2",
					CreatePodGroup: gangPodGroup2,
				},
				{
					Name: "Enqueue constrained pod (requires special=true) before flexible pod (prefers special=true)",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("strict").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
							NodeSelector(map[string]string{"special": "true"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("flex").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).
							NodeAffinity(preferredNodeAffinity("special", "true")).Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify the gang schedules",
					WaitForPodsScheduled: []string{"strict", "flex"},
				},
				{
					Name: "Verify PodGroup condition is Scheduled",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify strict lands on node2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"strict"},
						Nodes: sets.New("node2"),
					},
				},
				{
					Name: "Verify flex falls back to node1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"flex"},
						Nodes: sets.New("node1"),
					},
				},
			},
		},
		{
			// Documents a known limitation for intra-group dependencies (KEP-4671, "Algorithm Limitations"):
			// the worker is evaluated before its affinity target is assumed, so the gang can't be placed.
			name: "gang is unschedulable because worker is evaluated before its required PodAffinity target",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(v1.LabelHostname, "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=2",
					CreatePodGroup: gangPodGroup2,
				},
				{
					Name: "Enqueue worker (requiring PodAffinity to role=leader) before leader",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("worker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).
							PodAffinityIn("role", v1.LabelHostname, []string{"leader"}, st.PodAffinityWithRequiredReq).
							Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("leader").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).
							Label("role", "leader").Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                     "Verify the gang is unschedulable",
					WaitForPodsUnschedulable: []string{"worker", "leader"},
				},
				{
					Name: "Verify PodGroup condition is Unschedulable",
					WaitForGroupsUnschedulable: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
			},
		},
		{
			name: "gang is scheduled because required PodAffinity target is evaluated before the worker",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Label(v1.LabelHostname, "node1").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
				st.MakeNode().Name("node2").Label(v1.LabelHostname, "node2").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj(),
			},
			steps: []stepsframework.Step{
				{
					Name:           "Create gang PodGroup with minCount=2",
					CreatePodGroup: gangPodGroup2,
				},
				{
					Name: "Enqueue leader before worker (requiring PodAffinity to role=leader)",
					CreatePodsInOrder: []*v1.Pod{
						st.MakePod().Name("leader").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).
							Label("role", "leader").Container("image").PodGroupName("pg1").Priority(100).Obj(),
						st.MakePod().Name("worker").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).
							PodAffinityIn("role", v1.LabelHostname, []string{"leader"}, st.PodAffinityWithRequiredReq).
							Container("image").PodGroupName("pg1").Priority(100).Obj(),
					},
				},
				{
					Name:                 "Verify both pods schedule",
					WaitForPodsScheduled: []string{"leader", "worker"},
				},
				{
					Name: "Verify PodGroup condition is Scheduled",
					WaitForGroupsScheduled: &stepsframework.Groups{
						PodGroups: []string{"pg1"},
					},
				},
				{
					Name: "Verify both pods are co-located on one node",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"leader", "worker"},
						TopologyKey: v1.LabelHostname,
					},
				},
			},
		},
	}

	// CompositePodGroup requires TopologyAwareWorkloadScheduling, so (TAS off, CPG on) can't be enabled.
	for _, gates := range []struct{ tas, cpg bool }{{true, true}, {true, false}, {false, false}} {
		t.Run(fmt.Sprintf("TAS enabled: %v, CPG enabled: %v", gates.tas, gates.cpg), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: gates.tas,
				features.CompositePodGroup:               gates.cpg,
			})
			// The apiserver serves the PodGroup APIs depending on these gates, so share one per combination.
			sharedAPICtx := testutils.InitTestAPIServer(t, "podgroup-scheduling", nil)

			for _, tt := range tests {
				if len(tt.enableTopologyAwareWorkloadScheduling) > 0 && !slices.Contains(tt.enableTopologyAwareWorkloadScheduling, gates.tas) {
					continue
				}
				t.Run(tt.name, func(t *testing.T) {
					nsCtx := schedulerutils.WithNewNamespace(t, sharedAPICtx, "podgroup-scheduling")
					// Attribute scheduler logs to the test case rather than the feature-gate combination.
					tCtx := ktesting.Init(t)
					nsCtx.Ctx = tCtx
					testCtx := testutils.InitTestSchedulerWithOptions(t, nsCtx, 0,
						// disable backoff
						scheduler.WithPodMaxBackoffSeconds(0),
						scheduler.WithPodInitialBackoffSeconds(0))
					testutils.SyncSchedulerInformerFactory(testCtx)
					done := make(chan struct{})
					go func() {
						testCtx.Scheduler.Run(testCtx.SchedulerCtx)
						close(done)
					}()

					ns := testCtx.NS.Name
					tCtx.CleanupCtx(func(cleanupCtx ktesting.TContext) {
						testCtx.SchedulerCloseFn()
						<-done
						testCtx.InformerFactory.Shutdown()
						if testCtx.DynInformerFactory != nil {
							testCtx.DynInformerFactory.Shutdown()
						}
						cleanupPodGroupTestResources(t, cleanupCtx, testCtx.ClientSet, ns)
					})

					nodes := []*v1.Node{node}
					if len(tt.nodes) > 0 {
						nodes = tt.nodes
					}
					commonSteps := []stepsframework.Step{
						{
							Name:        "Create Nodes",
							CreateNodes: nodes,
						},
						{
							Name:            "Create workloads",
							CreateWorkloads: []*schedulingapi.Workload{workload, otherWorkload},
						},
					}

					if err := stepsframework.RunSteps(testCtx, t, ns, append(commonSteps, tt.steps...)); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
	}
}

// cleanupPodGroupTestResources deletes pods, podgroups, and workloads (which outlive their deleted namespace without a namespace controller) and cluster-scoped nodes.
func cleanupPodGroupTestResources(t *testing.T, ctx context.Context, cs clientset.Interface, ns string) {
	t.Helper()

	if err := cs.CoreV1().Pods(ns).DeleteCollection(ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		t.Fatalf("Failed to delete pods in namespace %s: %v", ns, err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, wait.ForeverTestTimeout, true,
		schedulerutils.PodsCleanedUp(ctx, cs, ns)); err != nil {
		t.Fatalf("Failed to wait for pods in namespace %s to be deleted: %v", ns, err)
	}

	// PodGroupProtection admission adds a finalizer and no controller removes it here.
	pgs, err := cs.SchedulingV1beta1().PodGroups(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list podgroups in namespace %s: %v", ns, err)
	}
	for i := range pgs.Items {
		if _, err := cs.SchedulingV1beta1().PodGroups(ns).Patch(ctx, pgs.Items[i].Name, types.MergePatchType, []byte(`{"metadata":{"finalizers":null}}`), metav1.PatchOptions{}); err != nil && !apierrors.IsNotFound(err) {
			t.Fatalf("Failed to remove finalizers from podgroup %s/%s: %v", ns, pgs.Items[i].Name, err)
		}
	}
	if err := cs.SchedulingV1beta1().PodGroups(ns).DeleteCollection(ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		t.Fatalf("Failed to delete podgroups in namespace %s: %v", ns, err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		pgs, err := cs.SchedulingV1beta1().PodGroups(ns).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(pgs.Items) == 0, nil
	}); err != nil {
		t.Fatalf("Failed to wait for podgroups in namespace %s to be deleted: %v", ns, err)
	}

	if err := cs.SchedulingV1beta1().Workloads(ns).DeleteCollection(ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		t.Fatalf("Failed to delete workloads in namespace %s: %v", ns, err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		workloads, err := cs.SchedulingV1beta1().Workloads(ns).List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(workloads.Items) == 0, nil
	}); err != nil {
		t.Fatalf("Failed to wait for workloads in namespace %s to be deleted: %v", ns, err)
	}

	if err := cs.CoreV1().Nodes().DeleteCollection(ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{}); err != nil {
		t.Fatalf("Failed to delete nodes: %v", err)
	}
	if err := wait.PollUntilContextTimeout(ctx, 50*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		nodes, err := cs.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		return len(nodes.Items) == 0, nil
	}); err != nil {
		t.Fatalf("Failed to wait for nodes to be deleted: %v", err)
	}
}

func TestWorkloadAwarePreemptionInvocation(t *testing.T) {
	for _, cpgEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("CPG enabled: %v", cpgEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: cpgEnabled,
				features.CompositePodGroup:               cpgEnabled,
			})

			node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

			workload := st.MakeWorkload().Name("workload").PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).Obj()
			pg := st.MakePodGroup().Namespace("default").Name("pg1").WorkloadRef("workload", "t1").
				DisruptionModeAll().Priority(100).MinCount(3).Obj()

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

			testCtx := schedulerutils.InitTestSchedulerWithNS(t, "wap-inv",
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
					schedulerutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Fatalf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
				}
			}

			// 2. Create workload
			if _, err := cs.SchedulingV1beta1().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create workload: %v", err)
			}

			// 3. Create PodGroup
			pg.Namespace = ns
			if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
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
		})
	}
}

// mockPostFilterPlugin is a custom PostFilter plugin that just counts invocations.
type mockPostFilterPlugin struct {
	lock  sync.Mutex
	count int
}

func (m *mockPostFilterPlugin) Name() string {
	return "MockPostFilter"
}

func (m *mockPostFilterPlugin) PostFilter(ctx context.Context, state framework.CycleState, pod *v1.Pod, filteredNodeStatusMap framework.NodeToStatusReader) (*framework.PostFilterResult, *framework.Status) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.count++
	return nil, framework.NewStatus(framework.Unschedulable)
}

func (m *mockPostFilterPlugin) getCount() int {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.count
}

func TestPostFilterNotCalled(t *testing.T) {
	for _, cpgEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("CPG enabled: %v", cpgEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: cpgEnabled,
				features.CompositePodGroup:               cpgEnabled,
			})

			node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "4"}).Obj()

			workload := st.MakeWorkload().Name("workload").PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(3).Obj()).Obj()
			pg := st.MakePodGroup().Namespace("default").Name("pg1").WorkloadRef("workload", "t1").
				DisruptionModeAll().Priority(100).MinCount(3).Obj()

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

			testCtx := schedulerutils.InitTestSchedulerWithNS(t, "post-filter-count",
				// Set high backoff times so that the scheduler does not retry scheduling before checking the count.
				scheduler.WithPodMaxBackoffSeconds(100),
				scheduler.WithPodInitialBackoffSeconds(100),
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
					schedulerutils.PodScheduled(cs, ns, p.Name)); err != nil {
					t.Fatalf("Failed to wait for pod %s to be scheduled: %v", p.Name, err)
				}
			}

			// 2. Create workload
			if _, err := cs.SchedulingV1beta1().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create workload: %v", err)
			}

			// 3. Create PodGroup
			pg.Namespace = ns
			if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create PodGroup: %v", err)
			}

			// 4. Create high priority pods
			for _, p := range highPods {
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}
			}

			// 5. Wait for high priority pod to be marked as unschedulable
			for _, p := range highPods {
				err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
					schedulerutils.PodUnschedulable(cs, ns, p.Name))
				if err != nil {
					t.Fatalf("failed to wait for pod %s to be unschedulable: %v", p.Name, err)
				}
			}

			if mockPlugin.getCount() != 0 {
				t.Fatalf("MockPostFilter was called %d times, expected 0", mockPlugin.getCount())
			}
		})
	}
}

// mockPodGroupPostFilterPlugin is a custom PodGroupPostFilter plugin that just counts invocations.
type mockPodGroupPostFilterPlugin struct {
	name  string
	lock  sync.Mutex
	count int
}

func (m *mockPodGroupPostFilterPlugin) Name() string {
	return m.name
}

func (m *mockPodGroupPostFilterPlugin) PodGroupPostFilter(ctx context.Context, state framework.PodGroupCycleState, pgInfo framework.PodGroupInfo, pgSchedulingFunc framework.PodGroupSchedulingFunc) (*framework.PodGroupPostFilterResult, *framework.Status) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.count++
	return &framework.PodGroupPostFilterResult{}, framework.NewStatus(framework.Unschedulable)
}

func (m *mockPodGroupPostFilterPlugin) getCount() int {
	m.lock.Lock()
	defer m.lock.Unlock()
	return m.count
}

func TestPodGroupPostFilterIteration(t *testing.T) {
	for _, cpgEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("CPG enabled: %v", cpgEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: cpgEnabled,
				features.CompositePodGroup:               cpgEnabled,
			})

			node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Obj()

			pg := st.MakePodGroup().Namespace("default").Name("pg1").DisruptionModeAll().Priority(100).MinCount(2).Obj()

			highPods := []*v1.Pod{
				st.MakePod().Namespace("default").Name("high-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
				st.MakePod().Namespace("default").Name("high-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg1").Priority(100).Obj(),
			}

			mockPlugin1 := &mockPodGroupPostFilterPlugin{name: "MockPlugin1"}
			mockPlugin2 := &mockPodGroupPostFilterPlugin{name: "MockPlugin2"}

			registry := frameworkruntime.Registry{
				"MockPlugin1": func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
					return mockPlugin1, nil
				},
				"MockPlugin2": func(ctx context.Context, obj runtime.Object, handle framework.Handle) (framework.Plugin, error) {
					return mockPlugin2, nil
				},
			}

			cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
				Profiles: []configv1.KubeSchedulerProfile{{
					SchedulerName: ptr.To(v1.DefaultSchedulerName),
					Plugins: &configv1.Plugins{
						MultiPoint: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: "GangScheduling"},
							},
						},
						PodGroupPostFilter: configv1.PluginSet{
							Enabled: []configv1.Plugin{
								{Name: "MockPlugin1"},
								{Name: "MockPlugin2"},
							},
							Disabled: []configv1.Plugin{
								{Name: "DefaultPreemption"},
							},
						},
					},
				}},
			})

			testCtx := schedulerutils.InitTestSchedulerWithNS(t, "pg-post-filter-iter",
				scheduler.WithPodMaxBackoffSeconds(100),
				scheduler.WithPodInitialBackoffSeconds(100),
				scheduler.WithFrameworkOutOfTreeRegistry(registry),
				scheduler.WithProfiles(cfg.Profiles...),
			)
			cs, ns := testCtx.ClientSet, testCtx.NS.Name

			if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create node: %v", err)
			}

			pg.Namespace = ns
			if _, err := cs.SchedulingV1beta1().PodGroups(ns).Create(testCtx.Ctx, pg, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Failed to create PodGroup: %v", err)
			}

			pgLister := testCtx.InformerFactory.Scheduling().V1beta1().PodGroups().Lister()
			err := wait.PollUntilContextTimeout(testCtx.Ctx, 10*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
				_, err := pgLister.PodGroups(ns).Get(pg.Name)
				return err == nil, nil
			})
			if err != nil {
				t.Fatalf("Failed to wait for PodGroup to be synced: %v", err)
			}

			for _, p := range highPods {
				p.Namespace = ns
				_, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("Failed to create pod %s: %v", p.Name, err)
				}

			}

			err = wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
				if mockPlugin1.getCount() == 1 && mockPlugin2.getCount() == 1 {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Errorf("Plugins were not called exactly once. MockPlugin1: %d, MockPlugin2: %d", mockPlugin1.getCount(), mockPlugin2.getCount())
			}
		})
	}
}
