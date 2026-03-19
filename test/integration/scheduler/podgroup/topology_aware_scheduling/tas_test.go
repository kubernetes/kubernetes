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

package topologyawarescheduling

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"

	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func makeGangPodGroup(podGroupName, topologyKey string, minCount int32) *schedulingapi.PodGroup {
	return st.MakePodGroup().Name(podGroupName).TemplateRef("t1", "workload").TopologyKey(topologyKey).MinCount(minCount).Obj()
}

func makeBasicPodGroup(podGroupName, topologyKey string) *schedulingapi.PodGroup {
	return st.MakePodGroup().Name(podGroupName).TemplateRef("t1", "workload").BasicPolicy().TopologyKey(topologyKey).Obj()
}

func makeNode(nodeName, rackLabel, zoneLabel string) *v1.Node {
	return st.MakeNode().Name(nodeName).Label("rack", rackLabel).Label("zone", zoneLabel).Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()
}

func makePod(podName, podGroupName string) *v1.Pod {
	return st.MakePod().Name(podName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").
		PodGroupName(podGroupName).Priority(100).ZeroTerminationGracePeriod().Obj()
}

// makeAssignedPod creates a pre-existing pod assigned to a node in the cluster
func makeAssignedPod(podName, nodeName, consumedCPU string) *v1.Pod {
	return st.MakePod().Name(podName).Node(nodeName).Req(map[v1.ResourceName]string{v1.ResourceCPU: consumedCPU}).Container("image").Priority(100).ZeroTerminationGracePeriod().Obj()
}

// makeLargePod creates a pod that consumes all resources of a single node
func makeLargePod(podName, podGroupName string) *v1.Pod {
	return st.MakePod().Name(podName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").
		PodGroupName(podGroupName).Priority(100).Obj()
}

// makeUnfittablePod creates a pod that requests more resources than available on a single node
func makeUnfittablePod(podName, podGroupName string) *v1.Pod {
	return st.MakePod().Name(podName).Req(map[v1.ResourceName]string{v1.ResourceCPU: "3"}).Container("image").
		PodGroupName(podGroupName).Priority(100).Obj()
}

// step represents a single step in a test scenario.
type step struct {
	name                      string
	createNodes               []*v1.Node
	createPodGroup            *schedulingapi.PodGroup
	createPods                []*v1.Pod
	deletePods                []string
	waitForPodsScheduled      []string
	verifyAssignments         *verifyAssignments
	verifyAssignedInOneDomain *verifyAssignedInOneDomain
	waitForPodsUnschedulable  []string
}

type verifyAssignments struct {
	pods  []string
	nodes sets.Set[string]
}

type verifyAssignedInOneDomain struct {
	pods        []string
	topologyKey string
}

type scenario struct {
	name  string
	steps []step
}

func TestTopologyAwareSchedulingWithGangPolicy(t *testing.T) {
	tests := []struct {
		name  string
		steps []step
	}{
		{
			name: "gang schedules on a single rack, when only one feasible rack is available",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks. Racks 1 and 2 can fit 3 pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
						makeNode("node5-rack2", "rack-2", "zone-1"),
						makeNode("node6-zone2", "rack-3", "zone-2"),
					},
				},
				{
					name: "Create an assigned pod in rack2, making rack2 unable to fit 3 additional pods",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node4-rack2", "2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name: "Verify all pods scheduled on rack1",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack1"),
					},
				},
			},
		},
		{
			name: "gang remains pending when no feasible resources are found",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-1"),
						makeNode("node4-zone2", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to the podgroup, requesting more resources in total than available in any rack",
					createPods: []*v1.Pod{
						makeLargePod("p1", "pg1"),
						makeLargePod("p2", "pg1"),
						makeLargePod("p3", "pg1"),
					},
				},
				{
					name:                     "Verify the entire gang becomes unschedulable",
					waitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "gang remains pending when resources are consumed by existing pods",
			steps: []step{
				{
					name: "Create nodes in a rack",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
					},
				},
				{
					name: "Create assigned pods in rack1, making rack1 unable to fit 3 additional pods",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node1-rack1", "2"),
						makeAssignedPod("existing2", "node3-rack1", "2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to the podgroup, requesting more resources in total than available in any rack",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                     "Verify the entire gang becomes unschedulable",
					waitForPodsUnschedulable: []string{"p1", "p2", "p3"},
				},
				{
					name:       "Delete a pod in rack1 to free up resources",
					deletePods: []string{"existing1"},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
			},
		},
		{
			name: "gang schedules on a single rack, choosing placement with the highest score in default placement scoring algorithm",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-2"),
						makeNode("node4-rack2", "rack-2", "zone-2"),
						makeNode("node5-rack2", "rack-2", "zone-2"),
						makeNode("node6-rack3", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to the podgroup. Pods won't fit on rack3, and will score higher on rack1 than rack2",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					// Scoring results: PodGroupPodsCount will score the same for rack1 and rack2,
					// and NodeResourcesFit with default strategy MostAllocated will score higher on rack1.
					name: "Verify all pods scheduled on rack1",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1"),
					},
				},
			},
		},
		{
			name: "gang schedules on a single rack, choosing placement with highest allocation percentage (default placement scoring algorithm) with pre-existing pods in cluster",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-2"),
						makeNode("node4-rack2", "rack-2", "zone-2"),
						makeNode("node5-rack2", "rack-2", "zone-2"),
						makeNode("node6-rack3", "rack-3", "zone-2"),
						makeNode("node7-rack3", "rack-3", "zone-2"),
						makeNode("node8-rack3", "rack-3", "zone-2"),
						makeNode("node9-rack4", "rack-4", "zone-2"),
					},
				},
				{
					name: "Create pre-existing nodes, which will be considered in the scoring algorithm",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node3-rack2", "2"),
						makeAssignedPod("existing2", "node8-rack3", "1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to the podgroup. Pods won't fit on rack4, and will score highest on rack2",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				// Each node has 2 CPUs, each of gang pods requests 1 CPU.
				// The PodGroupPods count scoring plugin will score racks 1-3 the same, as they will all fit the entire podgroup.
				// Allocation fractions in racks (for the default "most allocated" strategy the max allocation is picked):
				// - rack1: (0 + 3)/4 = 0.75
				// - rack2: (2 + 3)/6 = 0.83
				// - rack3: (1 + 3)/6 = 0.67
				// - rack4: unfeasible
				{
					name: "Verify all pods scheduled on rack2",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node3-rack2", "node4-rack2", "node5-rack2"),
					},
				},
			},
		},
		{
			name: "two gangs schedule consecutively, each on a separate rack",
			steps: []step{
				{
					name: "Create nodes in multiple racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
						makeNode("node5-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to gang pg1. Pods are large, pg1 will only fit on rack1",
					createPods: []*v1.Pod{
						makeLargePod("p1", "pg1"),
						makeLargePod("p2", "pg1"),
						makeLargePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang pg1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=2) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg2", "rack", 2),
				},
				{
					name: "Create all pods belonging to gang pg2",
					createPods: []*v1.Pod{
						makePod("p4", "pg2"),
						makePod("p5", "pg2"),
					},
				},
				{
					name:                 "Verify the entire gang pg2 is now scheduled",
					waitForPodsScheduled: []string{"p4", "p5"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack1 (the only one fitting them)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack1"),
					},
				},
				{
					name: "Verify all pods in pg2 scheduled on rack2 (the only one fitting them after pg1 is scheduled)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p4", "p5"},
						nodes: sets.New("node4-rack2", "node5-rack2"),
					},
				},
			},
		},
		{
			name: "two gangs schedule consecutively, both on the same rack",
			steps: []step{
				{
					name: "Create nodes in multiple racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=2) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 2),
				},
				{
					name: "Create all pods belonging to gang pg1",
					createPods: []*v1.Pod{
						makeLargePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg2", "rack", 3),
				},
				{
					name: "Create all pods belonging to gang pg2",
					createPods: []*v1.Pod{
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg2"),
					},
				},
				{
					name:                 "Verify the entire gang pg1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack1 (the only one fitting them)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack1"),
					},
				},
				{
					name:                 "Verify the entire gang pg2 is now scheduled",
					waitForPodsScheduled: []string{"p3", "p4", "p5"},
				},
				{
					name: "Verify all pods in pg2 scheduled also on rack1 (the only one fitting them)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p3", "p4", "p5"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack1"),
					},
				},
			},
		},
		{
			name: "two gangs schedule consecutively, different topology keys",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-1"),
						makeNode("node4-zone2", "rack-3", "zone-2"),
						makeNode("node5-zone2", "rack-3", "zone-2"),
						makeNode("node6-zone2", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to gang pg1",
					createPods: []*v1.Pod{
						makeLargePod("p1", "pg1"),
						makeLargePod("p2", "pg1"),
						makeLargePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang pg1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled in one zone",
					createPodGroup: makeGangPodGroup("pg2", "zone", 3),
				},
				{
					name: "Create all pods belonging to gang pg2",
					createPods: []*v1.Pod{
						makeLargePod("p4", "pg2"),
						makeLargePod("p5", "pg2"),
						makeLargePod("p6", "pg2"),
					},
				},
				{
					name:                 "Verify the entire gang pg2 is now scheduled",
					waitForPodsScheduled: []string{"p4", "p5", "p6"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack3 (the only one fitting them)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node4-zone2", "node5-zone2", "node6-zone2"),
					},
				},
				{
					name: "Verify all pods in pg2 scheduled in zone1 (the only one fitting them)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p4", "p5", "p6"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack2"),
					},
				},
			},
		},
		{
			name: "two gangs schedule consecutively, only one fits in the cluster",
			steps: []step{
				{
					name: "Create nodes in one rack",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create all pods belonging to gang pg1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang pg1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=2) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg2", "rack", 2),
				},
				{
					name: "Create all pods belonging to gang pg2",
					createPods: []*v1.Pod{
						makePod("p4", "pg2"),
						makePod("p5", "pg2"),
					},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack1",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1"),
					},
				},
				{
					name:                     "Verify the entire second gang becomes unschedulable due to insufficient resources",
					waitForPodsUnschedulable: []string{"p4", "p5"},
				},
			},
		},
		{
			name: "gang with minCount < pod count schedules on a single rack",
			steps: []step{
				{
					name: "Create nodes in two racks, each rack can fit 3 pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=2) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 2),
				},
				{
					name: "Create 3 pods (more than minCount) belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name: "Verify all pods scheduled on a single rack",
					verifyAssignedInOneDomain: &verifyAssignedInOneDomain{
						pods:        []string{"p1", "p2", "p3"},
						topologyKey: "rack",
					},
				},
			},
		},
		{
			// TODO: Add a test scenario where minCount < pod count and when entering the scheduling cycle, scheduler sees more than minCount pods in queue.
			name: "gang with minCount < pod count schedules on a single rack, choosing smaller rack",
			steps: []step{
				{
					name: "Create nodes in multiple racks, one can fit 4 pods, second can fit 6 pods, last is too small",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
						makeNode("node5-rack2", "rack-2", "zone-1"),
						makeNode("node6-rack3", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object (Gang with minCount=3) that should be scheduled on one rack",
					createPodGroup: makeGangPodGroup("pg1", "rack", 3),
				},
				{
					name: "Create 3 pods (=minCount) belonging to the gang",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name: "Create an additional pod (pod count > minCount) belonging to the gang",
					createPods: []*v1.Pod{
						makePod("p4", "pg1"),
					},
				},
				{
					name:                 "Verify the the new pods gets scheduled",
					waitForPodsScheduled: []string{"p4"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack2, which scored higher in the default placement scoring algorithm",
					// Scoring details:
					// At the time of scoring pg1 contained 3 pods. Racks 1 and 2 both fit all 3 pods, and were scored equally by PodGroupPodsCount.
					// NodeResourcesFit with the default mostAllocated strategy scored rack2 higher than rack1.
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3", "p4"},
						nodes: sets.New("node4-rack2", "node5-rack2"),
					},
				},
				{
					name: "Create one more pod belonging to the gang",
					createPods: []*v1.Pod{
						makePod("p5", "pg1"),
					},
				},
				{
					name:                     "Verify the last pod becomes unschedulable due to insufficient resources in the rack",
					waitForPodsUnschedulable: []string{"p5"},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runTestScenario(t, tt, true /* gangSchedulingEnabled */)
		})
	}
}

func TestTopologyAwareSchedulingWithBasicPolicy(t *testing.T) {
	tests := []scenario{
		{
			name: "basic podgroup schedules on the only rack in the cluster",
			steps: []step{
				{
					name: "Create nodes in only one rack",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire podgroup is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name: "Verify all pods scheduled on rack1",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1"),
					},
				},
			},
		},
		{
			name: "basic podgroup schedules all pods on a single rack, when there are multiple racks fitting in the cluster",
			steps: []step{
				{
					name: "Create nodes in multiple racks and zones, each rack enough to fit 3 pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
						makeNode("node5-zone2", "rack-3", "zone-2"),
						makeNode("node6-zone2", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					name: "Verify all pods scheduled the same rack",
					verifyAssignedInOneDomain: &verifyAssignedInOneDomain{
						pods:        []string{"p1", "p2", "p3"},
						topologyKey: "rack",
					},
				},
			},
		},
		{
			name: "basic podgroup cannot fit on a single rack, no pods get scheduled on a different rack",
			steps: []step{
				{
					name: "Create nodes in 2 racks, each rack enough to fit 2 pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup, more than fitting in a single rack",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify that 2 pods got scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					name:                     "Verify that the last pod becomes unschedulable due to insufficient resources in the rack",
					waitForPodsUnschedulable: []string{"p3"},
				},
				{
					name: "Verify both pods scheduled on the same rack",
					verifyAssignedInOneDomain: &verifyAssignedInOneDomain{
						pods:        []string{"p1", "p2"},
						topologyKey: "rack",
					},
				},
			},
		},
		{
			name: "basic podgroup does not schedule at all when no pods fit",
			steps: []step{
				{
					name: "Create nodes in 2 racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup, each pod not fitting any nodes in the cluster",
					createPods: []*v1.Pod{
						makeUnfittablePod("p1", "pg1"),
						makeUnfittablePod("p2", "pg1"),
					},
				},
				{
					name:                     "Verify no pods are scheduled due to insufficient resources in the rack",
					waitForPodsUnschedulable: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "basic podgroup schedules only some pods when others don't fit on any node",
			steps: []step{
				{
					name: "Create nodes in 2 racks, each rack enough to fit 2 regular pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup, some of them not fitting any nodes in the cluster",
					createPods: []*v1.Pod{
						makeUnfittablePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:                 "Verify that the smaller pod got scheduled",
					waitForPodsScheduled: []string{"p2"},
				},
				{
					name:                     "Verify that the big pod is not scheduled due to insufficient resources in the rack",
					waitForPodsUnschedulable: []string{"p1"},
				},
			},
		},
		{
			name: "basic podgroup schedules on a single rack, choosing placement with highest allocation percentage (default placement scoring algorithm)",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-2"),
						makeNode("node4-rack2", "rack-2", "zone-2"),
						makeNode("node5-rack2", "rack-2", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire gang is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				// Each node has 2 CPUs, each of gang pods requests 1 CPU.
				// Scores are first calculated when scheduling the first pod, without the knowledge how many pods would be in the podgroup.
				// PodGroupPodsCount will score the same for each rack (because both racks can fit the first pod).
				// NodeResourcesFit will compute allocation fractions in racks (for the default "most allocated" strategy the highest allocation is picked):
				// - rack1: 1/4 = 0.25
				// - rack2: 1/6 = 0.17
				{
					name: "Verify all pods scheduled on rack1",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node1-rack1", "node2-rack1"),
					},
				},
			},
		},
		{
			name: "basic podgroup schedules on a single rack, choosing placement with highest allocation percentage with pre-existing pods",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-2"),
						makeNode("node5-rack2", "rack-2", "zone-2"),
						makeNode("node6-rack2", "rack-2", "zone-2"),
						makeNode("node7-rack3", "rack-3", "zone-2"),
						makeNode("node8-rack3", "rack-3", "zone-2"),
						makeNode("node9-rack3", "rack-3", "zone-2"),
						makeNode("node10-rack3", "rack-3", "zone-2"),
					},
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node1-rack1", "1"),
						makeAssignedPod("existing2", "node5-rack2", "1"),
						makeAssignedPod("existing3", "node6-rack2", "1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:                 "Verify the entire podgroup is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				// Each node has 2 CPUs, each of podgroup pods requests 1 CPU.
				// Scores are first calculated when scheduling the first pod, without the knowledge how many pods would be in the podgroup.
				// PodGroupPodsCount will score the same for each rack (because both racks can fit the first pod).
				// NodeResourcesFit will compute allocation fractions in racks (for the default "most allocated" strategy the highest allocation is picked):
				// - rack1: 2/6 = 0.33
				// - rack2: 3/6 = 0.5
				// - rack3: 1/8 = 0.125
				{
					name: "Verify all pods scheduled on rack2",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2", "p3"},
						nodes: sets.New("node4-rack2", "node5-rack2", "node6-rack2"),
					},
				},
			},
		},
		{
			name: "basic podgroup schedules on a single rack, choosing best scoring placement that will not fit all podgroup",
			steps: []step{
				{
					name: "Create nodes in multiple racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack1", "rack-1", "zone-1"),
						makeNode("node5-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name: "Create pods in rack1, consuming 25% of its resources",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node1-rack1", "2"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				// Each node has 2 CPUs, podgroup pods request 1 or 2 CPUs.
				// Scores are first calculated when scheduling the first pod, without the knowledge how many pods would be in the podgroup.
				// PodGroupPodsCount will score the same for each rack (because both racks can fit the first pod).
				// NodeResourcesFit will compute allocation fractions in racks (for the default "most allocated" strategy the highest allocation is picked):
				// - rack1: 3/8 = 0.375
				// - rack2: 1/2 = 0.5
				{
					name:                 "Verify that the first two pods are now scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					name: "Verify pod scheduled on rack2",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node5-rack2"),
					},
				},
				{
					name:                     "Verify that third pod is not scheduled due to insufficient resources in rack2",
					waitForPodsUnschedulable: []string{"p3"},
				},
			},
		},
		{
			name: "two basic podgroups schedule consecutively, each on a single rack",
			steps: []step{
				{
					name: "Create nodes in multiple racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
						makeNode("node4-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg2", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg2",
					createPods: []*v1.Pod{
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					name:                 "Verify the entire podgroup pg1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				// Each node has 2 CPUs, podgroup pods request 1 or 2 CPUs.
				// Scores are first calculated when scheduling the first pod, without the knowledge how many pods would be in the podgroup.
				// PodGroupPodsCount will score the same for each rack (because both racks can fit the first pod).
				// Allocation fractions in racks (for the default "most allocated" strategy the highest allocation is picked):
				// - rack1: 1/6 = 0.17
				// - rack2: 1/2 = 0.5
				{
					name: "Verify all pods in pg1 scheduled on rack2 (which scored most allocation)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node4-rack2"),
					},
				},
				{
					name:                 "Verify the entire podgroup pg2 is now scheduled",
					waitForPodsScheduled: []string{"p3", "p4"},
				},
				{
					name: "Verify all pods in pg2 scheduled on rack1 (because rack2 is full)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p3", "p4"},
						nodes: sets.New("node1-rack1", "node2-rack1", "node3-rack1"),
					},
				},
			},
		},
		{
			name: "two basic podgroups schedule consecutively on the only rack",
			steps: []step{
				{
					name: "Create nodes in one rack",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack1", "rack-1", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg2", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg2",
					createPods: []*v1.Pod{
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg2"),
					},
				},
				{
					name:                 "Verify all pods scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5"},
				},
			},
		},
		{
			name: "two basic podgroups schedule consecutively, different topology keys",
			steps: []step{
				{
					name: "Create nodes in multiple zones and racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-zone2", "rack-2", "zone-2"),
						makeNode("node3-zone2", "rack-2", "zone-2"),
						makeNode("node4-zone2", "rack-3", "zone-2"),
						makeNode("node5-zone2", "rack-3", "zone-2"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled in one zone",
					createPodGroup: makeBasicPodGroup("pg2", "zone"),
				},
				{
					name: "Create all pods belonging to podgroup pg2. Large pods, each consumes CPU of an entire node",
					createPods: []*v1.Pod{
						makeLargePod("p3", "pg2"),
						makeLargePod("p4", "pg2"),
						makeLargePod("p5", "pg2"),
					},
				},
				{
					name:                 "Verify all pods scheduled",
					waitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack1 (the smallest rack, scoring highest allocation fraction)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node1-rack1"),
					},
				},
				{
					name: "Verify all pods in pg2 scheduled in zone2 (because zone1 is full)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p3", "p4", "p5"},
						nodes: sets.New("node2-zone2", "node3-zone2", "node4-zone2", "node5-zone2"),
					},
				},
			},
		},
		{
			name: "two basic podgroups schedule consecutively, one does not fit in the assigned rack",
			steps: []step{
				{
					name: "Create nodes in multiple racks",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack1", "rack-1", "zone-1"),
						makeNode("node3-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg2", "rack"),
				},
				{
					name: "Create all pods belonging to podgroup pg2",
					createPods: []*v1.Pod{
						makePod("p4", "pg2"),
						makePod("p5", "pg2"),
						makePod("p6", "pg2"),
					},
				},
				// Each node has 2 CPUs, each of gang pods requests 1 CPU.
				// Scores are first calculated when scheduling the first pod, without the knowledge how many pods would be in the podgroup.
				// PodGroupPodsCount will score the same for each rack (because both racks can fit the first pod).
				// Allocation fractions in racks (for the default "most allocated" strategy the highest allocation is picked):
				// - rack1: 1/4 = 0.25
				// - rack2: 1/2 = 0.5
				{
					name:                 "Verify part of podgroup p1 is now scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					name: "Verify all pods in pg1 scheduled on rack2 (which scored higher allocation)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node3-rack2"),
					},
				},
				{
					name:                     "Verify the last pod becomes unschedulable due to insufficient resources",
					waitForPodsUnschedulable: []string{"p3"},
				},
				{
					name:                 "Verify the entire podgroup pg2 is now scheduled",
					waitForPodsScheduled: []string{"p4", "p5", "p6"},
				},
				{
					name: "Verify pods in pg2 scheduled on rack1 (because rack2 is full)",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p4", "p5", "p6"},
						nodes: sets.New("node1-rack1", "node2-rack1"),
					},
				},
			},
		},
		{
			name: "basic podgroup continues to schedule pods when more resources become available",
			steps: []step{
				{
					name: "Create nodes in 2 racks, each rack enough to fit 2 pods",
					createNodes: []*v1.Node{
						makeNode("node1-rack1", "rack-1", "zone-1"),
						makeNode("node2-rack2", "rack-2", "zone-1"),
					},
				},
				{
					name: "Create blocker pod in rack1",
					createPods: []*v1.Pod{
						makeAssignedPod("existing1", "node1-rack1", "1"),
					},
				},
				{
					name:           "Create the PodGroup object that should be scheduled on one rack",
					createPodGroup: makeBasicPodGroup("pg1", "rack"),
				},
				{
					name: "Create all pods belonging to the podgroup, more than fitting in rack1",
					createPods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
					},
				},
				{
					name:                 "Verify that 1 pod got scheduled",
					waitForPodsScheduled: []string{"p1"},
				},
				{
					name: "Verify that p1 got scheduled on rack1, which scored higher in mostAllocated strategy",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1"},
						nodes: sets.New("node1-rack1"),
					},
				},
				{
					name:                     "Verify that the last pod becomes unschedulable due to insufficient resources in the rack",
					waitForPodsUnschedulable: []string{"p2"},
				},
				{
					name:       "Remove the blocker pod",
					deletePods: []string{"existing1"},
				},
				{
					name:                 "Verify that both pods got scheduled",
					waitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					name: "Verify that both are on the same rack",
					verifyAssignments: &verifyAssignments{
						pods:  []string{"p1", "p2"},
						nodes: sets.New("node1-rack1"),
					},
				},
			},
		},
	}

	for _, gangSchedulingEnabled := range []bool{true, false} {
		for _, tt := range tests {
			t.Run(fmt.Sprintf("%s (GangScheduling enabled: %v)", tt.name, gangSchedulingEnabled), func(t *testing.T) {
				runTestScenario(t, tt, gangSchedulingEnabled)
			})
		}
	}
}

func runTestScenario(t *testing.T, tt scenario, gangSchedulingEnabled bool) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.GangScheduling:                  gangSchedulingEnabled,
		features.TopologyAwareWorkloadScheduling: true,
	})

	// TODO: Add tests for at least one non-default scoring strategy.
	testCtx := testutils.InitTestSchedulerWithNS(t, "tas",
		// Disable backoff - it doesn't impact the end scheduling result.
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0))
	cs, ns := testCtx.ClientSet, testCtx.NS.Name

	workload := st.MakeWorkload().Name("workload").Namespace(ns).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(1).Obj()).
		Obj()
	if _, err := cs.SchedulingV1alpha2().Workloads(ns).Create(testCtx.Ctx, workload, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create Workload: %v", err)
	}

	for i, step := range tt.steps {
		t.Logf("Executing step %d: %s", i, step.name)
		switch {
		case step.createNodes != nil:
			for _, node := range step.createNodes {
				n := node.DeepCopy()
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, n, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Step %d: Failed to create node %s: %v", i, n.Name, err)
				}
			}
		case step.createPods != nil:
			for _, pod := range step.createPods {
				p := pod.DeepCopy()
				p.Namespace = ns
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, p, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Step %d: Failed to create pod %s: %v", i, p.Name, err)
				}
			}
		case step.createPodGroup != nil:
			w := step.createPodGroup.DeepCopy()
			w.Namespace = ns
			if _, err := cs.SchedulingV1alpha2().PodGroups(ns).Create(testCtx.Ctx, w, metav1.CreateOptions{}); err != nil {
				t.Fatalf("Step %d: Failed to create pod group %s: %v", i, w.Name, err)
			}
			// Ensure all next steps will see this pod group.
			err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
				func(_ context.Context) (bool, error) {
					_, err := testCtx.InformerFactory.Scheduling().V1alpha2().PodGroups().Lister().PodGroups(ns).Get(w.Name)
					if err != nil {
						if apierrors.IsNotFound(err) {
							return false, nil
						}
						return false, err
					}
					return true, nil
				},
			)
			if err != nil {
				t.Fatalf("Step %d: Failed to wait for pod group %s to be discoverable by scheduler: %v", i, w.Name, err)
			}
		case step.deletePods != nil:
			for _, podName := range step.deletePods {
				if err := cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, podName, metav1.DeleteOptions{}); err != nil {
					t.Fatalf("Step %d: Failed to delete pod %s: %v", i, podName, err)
				}
				// Ensure all next steps will not see the deleted pod.
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
					func(_ context.Context) (bool, error) {
						_, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
						if err != nil {
							if apierrors.IsNotFound(err) {
								return true, nil
							}
							return false, err
						}
						return false, nil
					},
				)
				if err != nil {
					t.Fatalf("Step %d: Failed to wait for pod %s to be no longer visible in scheduler: %v", i, podName, err)
				}
			}
		case step.verifyAssignments != nil:
			for _, podName := range step.verifyAssignments.pods {
				assignedPod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Step %d: Failed to retrieve assigned pod %s: %v", i, podName, err)
				}
				nodeName := assignedPod.Spec.NodeName
				if nodeName == "" {
					t.Errorf("Step %d: Pod %s is not assigned", i, podName)
				}
				if !step.verifyAssignments.nodes.Has(nodeName) {
					t.Errorf("Step %d: Wanted pod %s scheduled on node within %v but got assignment to %s. Error %v", i, podName, step.verifyAssignments.nodes, nodeName, err)
				}
			}
		case step.verifyAssignedInOneDomain != nil:
			expectedDomain := ""
			for _, podName := range step.verifyAssignedInOneDomain.pods {
				assignedPod, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, podName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Step %d: Failed to retrieve assigned pod %s: %v", i, podName, err)
				}
				nodeName := assignedPod.Spec.NodeName
				if nodeName == "" {
					t.Errorf("Step %d: Pod %s is not assigned", i, podName)
				}
				node, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, nodeName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("Step %d: Failed to retrieve node %s: %v", i, nodeName, err)
				}
				domain := node.Labels[step.verifyAssignedInOneDomain.topologyKey]
				if domain == "" {
					t.Fatalf("Step %d: Invalid domain value \"\" in node %s", i, nodeName)
				}

				if expectedDomain == "" {
					expectedDomain = domain
				} else if expectedDomain != domain {
					t.Errorf("Step %d: Pod %s assigned to a different domain than other pods in the podgroup. Expected %s but got %s", i, podName, expectedDomain, domain)
				}
			}
		case step.waitForPodsUnschedulable != nil:
			for _, podName := range step.waitForPodsUnschedulable {
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
					testutils.PodUnschedulable(cs, ns, podName))
				if err != nil {
					t.Fatalf("Step %d: Failed to wait for pod %s to be unschedulable: %v", i, podName, err)
				}
			}
		case step.waitForPodsScheduled != nil:
			for _, podName := range step.waitForPodsScheduled {
				err := wait.PollUntilContextTimeout(testCtx.Ctx, 100*time.Millisecond, wait.ForeverTestTimeout, false,
					testutils.PodScheduled(cs, ns, podName))
				if err != nil {
					t.Fatalf("Step %d: Failed to wait for pod %s to be scheduled: %v", i, podName, err)
				}
			}
		}
	}
}
