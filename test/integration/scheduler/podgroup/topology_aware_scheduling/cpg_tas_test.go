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
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func makeNodeWithLabels(nodeName string, labels map[string]string) *v1.Node {
	node := st.MakeNode().Name(nodeName).Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"})
	for k, v := range labels {
		node.Label(k, v)
	}
	return node.Obj()
}

func addPreferredNode(pod *v1.Pod, preferredNode string) *v1.Pod {
	pod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{PreferredDuringSchedulingIgnoredDuringExecution: []v1.PreferredSchedulingTerm{
			{Weight: 100, Preference: v1.NodeSelectorTerm{MatchFields: []v1.NodeSelectorRequirement{{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{preferredNode}}}}},
		}},
	}
	return pod
}

func addTaint(node *v1.Node, key string) *v1.Node {
	node.Spec.Taints = append(node.Spec.Taints, v1.Taint{Key: key, Effect: v1.TaintEffectNoSchedule})
	return node
}

func addToleration(pod *v1.Pod, key string) *v1.Pod {
	pod.Spec.Tolerations = append(pod.Spec.Tolerations, v1.Toleration{Key: key})
	return pod
}

func makeAssignedGroupPod(podName, podGroupName, nodeName, consumedCPU string) *v1.Pod {
	return st.MakePod().Name(podName).PodGroupName(podGroupName).Node(nodeName).Req(map[v1.ResourceName]string{v1.ResourceCPU: consumedCPU}).Container("image").Priority(100).ZeroTerminationGracePeriod().Obj()
}

func makeGangPodGroupWithParent(podGroupName, parentCPGName, topologyKey string, minCount int32) *schedulingapi.PodGroup {
	pg := st.MakePodGroup().Name(podGroupName).WorkloadRef("pg", "workload").MinCount(minCount).Priority(100).ParentCompositePodGroup(parentCPGName)
	if topologyKey != "" {
		pg.TopologyKey(topologyKey)
	}
	return pg.Obj()
}

func makeBasicPodGroupWithParent(podGroupName, parentCPGName, topologyKey string) *schedulingapi.PodGroup {
	pg := st.MakePodGroup().Name(podGroupName).WorkloadRef("pg", "workload").BasicPolicy().Priority(100).ParentCompositePodGroup(parentCPGName)
	if topologyKey != "" {
		pg.TopologyKey(topologyKey)
	}
	return pg.Obj()
}

func makeGangCompositePodGroup(cpgName, parentCPGName, topologyKey string, minGroupCount int32) *schedulingv1alpha3.CompositePodGroup {
	cpg := st.MakeCompositePodGroup().Name(cpgName).WorkloadRef("cpg", "workload").MinGroupCount(minGroupCount).Priority(100)
	if parentCPGName != "" {
		cpg.ParentCompositePodGroup(parentCPGName)
	}
	if topologyKey != "" {
		cpg.TopologyKey(topologyKey)
	}
	return cpg.Obj()
}

func makeBasicCompositePodGroup(cpgName, parentCPGName, topologyKey string) *schedulingv1alpha3.CompositePodGroup {
	cpg := st.MakeCompositePodGroup().Name(cpgName).WorkloadRef("cpg", "workload").BasicPolicy().Priority(100)
	if parentCPGName != "" {
		cpg.ParentCompositePodGroup(parentCPGName)
	}
	if topologyKey != "" {
		cpg.TopologyKey(topologyKey)
	}
	return cpg.Obj()
}

func TestCPGTopologyAwareScheduling(t *testing.T) {
	tests := []scenario{
		{
			name: "parent CPG has topology constraints, children do not; schedules on a single rack",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks, each rack with 4 CPU available",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create a pod on rack-2, taking up 2 out of 4 CPUs available on rack-2",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing", "node4-z1-r2", "2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2, each pod requiring 1 CPU",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify all pods across both children scheduled on rack1 due to parent CPG topology constraint",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, children do not; preexisting pod belonging to the hierarchy determines the topology",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks, each rack with 4 CPU available",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create an assigned pod from pg1 in rack-2",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing", "pg1", "node4-z1-r2", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create the remaining pods belonging to pg1 and pg2, each pod requiring 1 CPU",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify all pods across both children scheduled on rack-2 due to preexisting pod",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node3-z1-r2", "node4-z1-r2"),
					},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, children do not; remains pending when no single rack can fit all children",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks. Both rack-1 and rack-2 can fit at most 2 pods",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Assign a pod on both rack-1 and rack-2, such that no single rack can fit 2 pods",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing1", "node1-z1-r1", "1"),
						makeAssignedPod("existing2", "node2-z1-r2", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Basic policy, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg1", "cpg-root", ""),
				},
				{
					Name:           "Create child PodGroup pg2 (Basic policy, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg2", "cpg-root", ""),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2 (total 2 pods)",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
					},
				},
				{
					Name:                     "Verify all pods become unschedulable because parent requires all 2 pods on a single rack",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "parent CPG and child PGs both have topology constraints (multi-level constraints: zone and rack)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r1", "rack-1", "zone-2"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name: "Create an assigned pod in zone-2 rack-1 making zone-2 able to fit at most 3 pods",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing-z2", "node3-z2-r1", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify assignments are in zone-1 matching cpg-level topology constraints",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r2"),
					},
				},
				{
					Name: "Verify pg1 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pg2 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "parent CPG and only one child PG has topology constraints; the other child uses parent's topology",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						// In order for the whole group to fit, pg1 must choose node1.
						// To prevent other pgs from taking its place, we add a taint.
						addTaint(makeNode("node1-z1-r1", "rack-1", "zone-1"), "taint"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z1-r3", "rack-3", "zone-1"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
						makeNode("node5-z2-r2", "rack-2", "zone-2"),
					},
				},
				{
					Name: "Create an assigned pod in zone-2 rack-2 making zone-2 able to fit at most 3 pods and therefore ineligible for cpg-root",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing-z2", "node5-z2-r2", "1"),
					},
				},
				{
					Name: "Create assigned pods in zone-1 rack-2 and rack-3 making rack-2 and rack-3 able to fit at most 1 pod each and therefore ineligible for pg1",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing-z1-r2", "node2-z1-r2", "1"),
						makeAssignedPod("existing-z1-r3", "node3-z1-r3", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						// Since we want pg1 pods to schedule on the tainted node, we need to add a toleration.
						addToleration(makePod("p1", "pg1"), "taint"),
						addToleration(makePod("p2", "pg1"), "taint"),
						// Preferred node is in zone-2, which cannot fit both groups and is therefore ineligible.
						addPreferredNode(makePod("p3", "pg2"), "node4-z2-r1"),
						addPreferredNode(makePod("p4", "pg2"), "node4-z2-r1"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify all pods are in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pg1 pods are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "parent CPG and all child PGs have topology constraints, preexisting pod group pods constrain the available topology domains",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Assign pg1 pod to zone-1 rack-1",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing-z1", "pg1", "node1-z1-r1", "1"),
					},
				},
				{
					Name: "Create remaining unscheduled pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name: "Verify pg1 assignments are in rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1"},
						Nodes: sets.New("node1-z1-r1"),
					},
				},
				{
					Name: "Verify pg2 assignments are in rack-2 of zone-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p2", "p3"},
						Nodes: sets.New("node2-z1-r2", "node3-z1-r2"),
					},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, preexisting pod group pods are in conflicting domains across pod groups, fails to schedule any pod",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z2-r1", "rack-1", "zone-2"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Assign pg1 pod to zone-1 and pg2 pod to zone-2",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing-z1", "pg1", "node1-z1-r1", "1"),
						makeAssignedGroupPod("existing-z2", "pg2", "node3-z2-r1", "1"),
					},
				},
				{
					Name: "Create remaining unscheduled pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
					},
				},
				{
					Name:                       "Verify all pods get scheduling error due to preexisting conflicting zone domains",
					WaitForPodsSchedulingError: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: all levels have topology constraints (root=zone, sub=block, leaf=rack)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones, blocks, and racks",
					CreateNodes: []*v1.Node{
						makeNodeWithLabels("node1-z1-g1-r1", map[string]string{"zone": "zone-1", "block": "block-1", "rack": "rack-1"}),
						makeNodeWithLabels("node2-z1-g1-r2", map[string]string{"zone": "zone-1", "block": "block-1", "rack": "rack-2"}),
						makeNodeWithLabels("node3-z1-g2-r1", map[string]string{"zone": "zone-1", "block": "block-2", "rack": "rack-1"}),
						makeNodeWithLabels("node4-z1-g2-r2", map[string]string{"zone": "zone-1", "block": "block-2", "rack": "rack-2"}),
						makeNodeWithLabels("node5-z2-g1-r1", map[string]string{"zone": "zone-2", "block": "block-1", "rack": "rack-1"}),
						makeNodeWithLabels("node6-z2-g1-r1", map[string]string{"zone": "zone-2", "block": "block-1", "rack": "rack-1"}),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=block, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "block", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=block, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "block", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same block",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "block",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same block",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "block",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg3 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg4 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: mid-level without topology constraints (root=zone, sub=none, leaf=rack)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Basic without topology constraints, Parent=cpg-root)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-sub1", "cpg-root", ""),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Basic without topology constraints, Parent=cpg-root)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-sub2", "cpg-root", ""),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to cpg-root",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods across all pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: top level without topology constraint (root=none, sub=zone, leaf=rack)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r3", "rack-3", "zone-2"),
						makeNode("node4-z2-r4", "rack-4", "zone-2"),
						makeNode("node5-z2-r5", "rack-5", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object without topology constraint (Basic policy)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-root", "", ""),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=zone, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=zone, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg3 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg4 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: bottom level without topology constraint (root=zone, sub=rack, leaf=none)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks. Zone-1 fits 4 pods per rack across 2 racks. Zone-2 fits 2 pods per rack across 1 rack after assigned pod",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z2-r3", "rack-3", "zone-2"),
						makeNode("node6-z2-r3", "rack-3", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Basic policy, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "rack", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: preexisting pod in leaf PodGroup determines topology for intermediate and root CPGs",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across two zones, each zone with 8 CPUs across two racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
						makeNode("node6-z2-r1", "rack-1", "zone-2"),
						makeNode("node7-z2-r2", "rack-2", "zone-2"),
						makeNode("node8-z2-r2", "rack-2", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "rack", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "", 2),
				},
				{
					Name: "Assign a preexisting pg1 pod to rack-1 in zone-2 to anchor cpg-sub1 to rack-1 and cpg-root to zone-2",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing-pg1", "pg1", "node5-z2-r1", "1"),
					},
				},
				{
					Name: "Create remaining unscheduled pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
						makePod("p4", "pg3"),
						makePod("p5", "pg3"),
						makePod("p6", "pg4"),
						makePod("p7", "pg4"),
					},
				},
				{
					Name:                 "Verify all newly created pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7"},
				},
				{
					Name: "Verify all pods across both sub-CPGs are scheduled in zone-2 due to cpg-root topology constraint",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7"},
						Nodes: sets.New("node5-z2-r1", "node6-z2-r1", "node7-z2-r2", "node8-z2-r2"),
					},
				},
				{
					Name: "Verify remaining cpg-sub1 pods are scheduled in rack-1 of zone-2 due to preexisting pod",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node5-z2-r1", "node6-z2-r1"),
					},
				},
				{
					Name: "Verify cpg-sub2 pods are scheduled in rack-2 of zone-2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p4", "p5", "p6", "p7"},
						Nodes: sets.New("node7-z2-r2", "node8-z2-r2"),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCPGTestScenario(t, tt)
		})
	}
}

func TestCPGTopologyAwareSchedulingWorkloadAwarePreemption(t *testing.T) {
	tests := []scenario{
		{
			name: "parent CPG has topology constraints, children do not; schedules on a single rack after preempting lower priority pods",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks, each rack with 4 CPU available",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create low-priority pods on rack-1 and a pod on rack-2, making both racks unable to fit 4 CPU without preemption",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low1-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low2-z1-r1", "node2-z1-r1", "2", 10),
						makeAssignedPod("existing", "node4-z1-r2", "2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2, each pod requiring 1 CPU",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify low-priority pods on rack-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low1-z1-r1", "low2-z1-r1"},
				},
				{
					Name: "Verify all pods across both children scheduled on rack1 due to parent CPG topology constraint after preemption",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, children do not; schedules on a single rack after preempting lower priority pods, then cannot schedule additional pods",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks, each rack with 4 CPU available",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create low-priority pods on rack-1 and a pod on rack-2, making both racks unable to fit 4 CPU without preemption, new pod does not schedule on any node",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low1-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low2-z1-r1", "node2-z1-r1", "2", 10),
						makeAssignedPod("existing", "node4-z1-r2", "2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2, each pod requiring 1 CPU",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify low-priority pods on rack-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low1-z1-r1", "low2-z1-r1"},
				},
				{
					Name: "Verify all pods across both children scheduled on rack1 due to parent CPG topology constraint after preemption",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
				{
					Name: "Add a new pod to the hierarchy",
					CreatePods: []*v1.Pod{
						makePod("p5", "pg1"),
					},
				},
				// There is a space on rack2 available (2CPU), but pod cannot be scheduled because of parent PG topology
				{
					Name:                     "Verify the new pod is unschedulable",
					WaitForPodsUnschedulable: []string{"p5"},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, children do not; preexisting pod belonging to the hierarchy determines topology and lower priority pods are preempted",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks, each rack with 4 CPU available",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create an assigned pod from pg1 in rack-2, and low-priority pods making rack-2 unable to fit 3 remaining CPUs without preemption",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing", "pg1", "node4-z1-r2", "1"),
						makeAssignedPodWithPriority("low1-z1-r2", "node3-z1-r2", "2", 10),
						makeAssignedPodWithPriority("low2-z1-r2", "node4-z1-r2", "1", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create the remaining pods belonging to pg1 and pg2, each pod requiring 1 CPU",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:               "Verify low-priority pods on rack-2 are removed via preemption",
					WaitForPodsRemoved: []string{"low1-z1-r2", "low2-z1-r2"},
				},
				{
					Name: "Verify all pods across both children scheduled on rack-2 due to preexisting pod",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node3-z1-r2", "node4-z1-r2"),
					},
				},
			},
		},
		{
			name: "parent CPG has topology constraints, children do not; schedules on a rack after preempting lower priority pods",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple racks. Both rack-1 and rack-2 can fit at most 2 pods",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Assign a low-priority pod on rack-1 and a high-priority pod on rack-2",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low1-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPod("existing2", "node2-z1-r2", "2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Basic policy, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg1", "cpg-root", ""),
				},
				{
					Name:           "Create child PodGroup pg2 (Basic policy, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg2", "cpg-root", ""),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2 (total 2 pods)",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					Name:               "Verify low-priority pod on rack-1 is removed via preemption",
					WaitForPodsRemoved: []string{"low1-z1-r1"},
				},
				{
					Name: "Verify all pods scheduled on rack-1 after preemption",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2"},
						Nodes: sets.New("node1-z1-r1"),
					},
				},
			},
		},
		{
			name: "parent CPG and child PGs both have topology constraints (multi-level constraints: zone and rack) with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r1", "rack-1", "zone-2"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name: "Create low-priority pods in zone-1 and high-priority assigned pod in zone-2",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-r2", "node2-z1-r2", "2", 10),
						makeAssignedPod("existing-z2", "node3-z2-r1", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify low-priority pods in zone-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1", "low-z1-r2"},
				},
				{
					Name: "Verify assignments are in zone-1 matching cpg-level topology constraints after preemption",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r2"),
					},
				},
				{
					Name: "Verify pg1 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pg2 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "parent CPG and child PGs both have topology constraints (multi-level constraints: zone and rack), preemption does not help for subsequent",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z1-r3", "rack-3", "zone-1"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name: "Create low-priority pods in zone-2 and rack-3",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r3", "node3-z1-r3", "2", 10),
						makeAssignedPodWithPriority("low-z2-r1", "node4-z2-r1", "2", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2",
					// Those should fit on zone-1, without preemption
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify pg1 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pg2 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-root", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg3",
					CreatePods: []*v1.Pod{
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
					},
				},
				{
					Name:                 "Verify all pods in the pg3 are scheduled",
					WaitForPodsScheduled: []string{"p5", "p6"},
				},
				{
					Name:               "Verify low-priority pods in rack-3 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r3"},
				},
				{
					Name: "Verify pg3 assignments are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
					},
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-root", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg4",
					CreatePods: []*v1.Pod{
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name: "Verify all pods in the pg4 are unschedulable",
					// Even though preemption can free up space in zone-2/rack-1, this PG cannot be scheduled there.
					WaitForPodsUnschedulable: []string{"p7", "p8"},
				},
			},
		},
		{
			name: "parent CPG and only one child PG has topology constraints; the other child uses parent's topology with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						addTaint(makeNode("node1-z1-r1", "rack-1", "zone-1"), "taint"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z1-r3", "rack-3", "zone-1"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
						makeNode("node5-z2-r2", "rack-2", "zone-2"),
					},
				},
				{
					Name: "Create assigned pods in zone-2 and zone-1, including low-priority pod on node1-z1-r1",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPod("existing-z2", "node5-z2-r2", "1"),
						makeAssignedPod("existing-z1-r2", "node2-z1-r2", "1"),
						makeAssignedPod("existing-z1-r3", "node3-z1-r3", "1"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						addToleration(makePod("p1", "pg1"), "taint"),
						addToleration(makePod("p2", "pg1"), "taint"),
						addPreferredNode(makePod("p3", "pg2"), "node4-z2-r1"),
						addPreferredNode(makePod("p4", "pg2"), "node4-z2-r1"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify low-priority pod on node1-z1-r1 is removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1"},
				},
				{
					Name: "Verify all pods are in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pg1 pods are in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "parent CPG and all child PGs have topology constraints, preexisting pod group pods constrain available topology domains with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z2-r1", "rack-1", "zone-2"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "rack", 2),
				},
				{
					Name: "Assign pg1 pod to zone-1 rack-1, and create low-priority pods blocking remaining capacity",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing-z1", "pg1", "node1-z1-r1", "1"),
						makeAssignedPodWithPriority("low-z1-r1", "node1-z1-r1", "1", 10),
						makeAssignedPodWithPriority("low-z1-r2", "node2-z1-r2", "2", 10),
					},
				},
				{
					Name: "Create remaining unscheduled pods belonging to pg1 and pg2",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:               "Verify low-priority pods are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1", "low-z1-r2"},
				},
				{
					Name: "Verify pg1 assignments are in rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1"},
						Nodes: sets.New("node1-z1-r1"),
					},
				},
				{
					Name: "Verify pg2 assignments are in rack-2 of zone-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p2", "p3"},
						Nodes: sets.New("node2-z1-r2", "node3-z1-r2"),
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: all levels have topology constraints (root=zone, sub=block, leaf=rack) with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones, blocks, and racks",
					CreateNodes: []*v1.Node{
						makeNodeWithLabels("node1-z1-g1-r1", map[string]string{"zone": "zone-1", "block": "block-1", "rack": "rack-1"}),
						makeNodeWithLabels("node2-z1-g1-r2", map[string]string{"zone": "zone-1", "block": "block-1", "rack": "rack-2"}),
						makeNodeWithLabels("node3-z1-g2-r1", map[string]string{"zone": "zone-1", "block": "block-2", "rack": "rack-1"}),
						makeNodeWithLabels("node4-z1-g2-r2", map[string]string{"zone": "zone-1", "block": "block-2", "rack": "rack-2"}),
						makeNodeWithLabels("node5-z2-g1-r1", map[string]string{"zone": "zone-2", "block": "block-1", "rack": "rack-1"}),
						makeNodeWithLabels("node6-z2-g1-r1", map[string]string{"zone": "zone-2", "block": "block-1", "rack": "rack-1"}),
					},
				},
				{
					Name: "Create low-priority pods filling zone-1 nodes",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-1", "node1-z1-g1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-2", "node2-z1-g1-r2", "2", 10),
						makeAssignedPodWithPriority("low-z1-3", "node3-z1-g2-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-4", "node4-z1-g2-r2", "2", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=block, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "block", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=block, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "block", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name:               "Verify low-priority pods in zone-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-1", "low-z1-2", "low-z1-3", "low-z1-4"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same block",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "block",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same block",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "block",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg3 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg4 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: mid-level without topology constraints (root=zone, sub=none, leaf=rack) with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r1", "rack-1", "zone-2"),
					},
				},
				{
					Name: "Create low-priority pods filling zone-1 nodes",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-r2", "node2-z1-r2", "2", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Basic without topology constraints, Parent=cpg-root)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-sub1", "cpg-root", ""),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Basic without topology constraints, Parent=cpg-root)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-sub2", "cpg-root", ""),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to cpg-root",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods across all pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:               "Verify low-priority pods in zone-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1", "low-z1-r2"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: top level without topology constraint (root=none, sub=zone, leaf=rack) with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r2", "rack-2", "zone-1"),
						makeNode("node3-z2-r3", "rack-3", "zone-2"),
						makeNode("node4-z2-r4", "rack-4", "zone-2"),
					},
				},
				{
					Name: "Create low-priority pods filling nodes in zone-1 and zone-2",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-r2", "node2-z1-r2", "2", 10),
						makeAssignedPodWithPriority("low-z2-r3", "node3-z2-r3", "2", 10),
						makeAssignedPodWithPriority("low-z2-r4", "node4-z2-r4", "2", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object without topology constraint (Basic policy)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-root", "", ""),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=zone, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=zone, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "zone", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, TopologyKey=rack, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "rack", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name:               "Verify low-priority pods are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1", "low-z1-r2", "low-z2-r3", "low-z2-r4"},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify pods in pg1 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg2 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg3 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify pods in pg4 are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: bottom level without topology constraint (root=zone, sub=rack, leaf=none) with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z2-r3", "rack-3", "zone-2"),
						makeNode("node6-z2-r3", "rack-3", "zone-2"),
					},
				},
				{
					Name: "Create low-priority pods filling zone-1 nodes",
					CreatePods: []*v1.Pod{
						makeAssignedPodWithPriority("low-z1-r1-1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-r1-2", "node2-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low-z1-r2-1", "node3-z1-r2", "2", 10),
						makeAssignedPodWithPriority("low-z1-r2-2", "node4-z1-r2", "2", 10),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Basic policy, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "rack", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "", 2),
				},
				{
					Name: "Create all pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
						makePod("p7", "pg4"),
						makePod("p8", "pg4"),
					},
				},
				{
					Name:                 "Verify all pods across all four child PGs are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
				},
				{
					Name:               "Verify low-priority pods in zone-1 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z1-r1-1", "low-z1-r1-2", "low-z1-r2-1", "low-z1-r2-2"},
				},
				{
					Name: "Verify all pods are assigned in the same zone",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"},
						TopologyKey: "zone",
					},
				},
				{
					Name: "Verify cpg-sub1 pods are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1", "p2", "p3", "p4"},
						TopologyKey: "rack",
					},
				},
				{
					Name: "Verify cpg-sub2 pods are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6", "p7", "p8"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "3-level CPG hierarchy: preexisting pod in leaf PodGroup determines topology for intermediate and root CPGs with preemption",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across two zones, each zone with 8 CPUs across two racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
						makeNode("node6-z2-r1", "rack-1", "zone-2"),
						makeNode("node7-z2-r2", "rack-2", "zone-2"),
						makeNode("node8-z2-r2", "rack-2", "zone-2"),
					},
				},
				{
					Name:                    "Create the root CompositePodGroup object (Gang with minGroupCount=2, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "rack", 2),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub2 (Gang with minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub2", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang with minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-sub2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg4 (Gang with minCount=2, without topology constraints, Parent=cpg-sub2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-sub2", "", 2),
				},
				{
					Name: "Assign a preexisting pg1 pod to rack-1 in zone-2, and low-priority pods on rack-1 and rack-2 in zone-2",
					CreatePods: []*v1.Pod{
						makeAssignedGroupPod("existing-pg1", "pg1", "node5-z2-r1", "1"),
						makeAssignedPodWithPriority("low-z2-r1", "node6-z2-r1", "2", 10),
						makeAssignedPodWithPriority("low-z2-r2", "node7-z2-r2", "2", 10),
					},
				},
				{
					Name: "Create remaining unscheduled pods belonging to pg1, pg2, pg3, and pg4",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg2"),
						makePod("p3", "pg2"),
						makePod("p4", "pg3"),
						makePod("p5", "pg3"),
						makePod("p6", "pg4"),
						makePod("p7", "pg4"),
					},
				},
				{
					Name:                 "Verify all newly created pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7"},
				},
				{
					Name:               "Verify low-priority pods in zone-2 are removed via preemption",
					WaitForPodsRemoved: []string{"low-z2-r1", "low-z2-r2"},
				},
				{
					Name: "Verify all pods across both sub-CPGs are scheduled in zone-2 due to cpg-root topology constraint",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4", "p5", "p6", "p7"},
						Nodes: sets.New("node5-z2-r1", "node6-z2-r1", "node7-z2-r2", "node8-z2-r2"),
					},
				},
				{
					Name: "Verify remaining cpg-sub1 pods are scheduled in rack-1 of zone-2 due to preexisting pod",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node5-z2-r1", "node6-z2-r1"),
					},
				},
				{
					Name: "Verify cpg-sub2 pods are scheduled in rack-2 of zone-2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p4", "p5", "p6", "p7"},
						Nodes: sets.New("node7-z2-r2", "node8-z2-r2"),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runCPGTestScenario(t, tt)
		})
	}
}

func runCPGTestScenario(t *testing.T, tt scenario) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.CompositePodGroup:               true,
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	testCtx := testutils.InitTestSchedulerWithNS(t, "cpg-tas",
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0))
	ns := testCtx.NS.Name

	if err := stepsframework.RunSteps(testCtx, t, ns, tt.steps); err != nil {
		t.Fatal(err)
	}
}
