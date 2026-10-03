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
	pg := st.MakePodGroup().Name(podGroupName).WorkloadRef("workload", "pg").MinCount(minCount).Priority(100).ParentCompositePodGroup(parentCPGName)
	if topologyKey != "" {
		pg.TopologyKey(topologyKey)
	}
	return pg.Obj()
}

func makeBasicPodGroupWithParent(podGroupName, parentCPGName, topologyKey string) *schedulingapi.PodGroup {
	pg := st.MakePodGroup().Name(podGroupName).WorkloadRef("workload", "pg").BasicPolicy().Priority(100).ParentCompositePodGroup(parentCPGName)
	if topologyKey != "" {
		pg.TopologyKey(topologyKey)
	}
	return pg.Obj()
}

func makeGangCompositePodGroup(cpgName, parentCPGName, topologyKey string, minGroupCount int32) *schedulingv1alpha3.CompositePodGroup {
	cpg := st.MakeCompositePodGroup().Name(cpgName).WorkloadRef("workload", "cpg").MinGroupCount(minGroupCount).Priority(100)
	if parentCPGName != "" {
		cpg.ParentCompositePodGroup(parentCPGName)
	}
	if topologyKey != "" {
		cpg.TopologyKey(topologyKey)
	}
	return cpg.Obj()
}

func makeBasicCompositePodGroup(cpgName, parentCPGName, topologyKey string) *schedulingv1alpha3.CompositePodGroup {
	cpg := st.MakeCompositePodGroup().Name(cpgName).WorkloadRef("workload", "cpg").BasicPolicy().Priority(100)
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
		{
			name: "CPG schedules on a single rack, choosing placement with highest allocation percentage (NodeResourcesFit placement scoring)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in two racks, each rack with 4 CPU capacity",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create a preexisting pod consuming 1 CPU on rack-1 leaving 3 CPU free",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing", "node1-z1-r1", "1"),
					},
				},
				{
					Name:                    "Create root CompositePodGroup (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang minCount=1, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 1),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create pods for pg1 (1 pod) and pg2 (2 pods), requiring 3 CPU total",
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
					Name: "Verify all pods scheduled on rack-1 due to highest allocation percentage scoring",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "CPG schedules on a single rack, choosing placement that accommodates more descendant pods (PodGroupPodsCount placement scoring)",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in two racks, each rack with 4 CPU capacity",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Create a preexisting pod consuming 1 CPU on rack-1 leaving only 3 CPU free",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing", "node1-z1-r1", "1"),
					},
				},
				{
					Name:           "Create child PodGroup pg1 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name: "Create 4 pods across pg1 and pg2 requiring 4 CPU total",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
					},
				},
				{
					Name:                    "Create root CompositePodGroup (Gang minGroupCount=1, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 1),
				},
				{
					Name:                 "Verify all 4 pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name: "Verify all pods scheduled on rack-2 which fits all 4 pods",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node3-z1-r2", "node4-z1-r2"),
					},
				},
			},
		},
		{
			name: "CPG with minGroupCount < total children schedules satisfying subset on a single rack",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in two racks, each rack with 4 CPU capacity",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name: "Occupy rack-2 completely with preexisting pods",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing1", "node3-z1-r2", "2"),
						makeAssignedPod("existing2", "node4-z1-r2", "2"),
					},
				},
				{
					Name:                    "Create root CompositePodGroup (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 2),
				},
				{
					Name:           "Create child PodGroup pg3 (Gang minCount=2, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-root", "", 2),
				},
				{
					Name: "Create pods for pg1, pg2, and pg3 (6 pods total)",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
						makePod("p4", "pg2"),
						makePod("p5", "pg3"),
						makePod("p6", "pg3"),
					},
				},
				{
					Name:                 "Verify pods for pg1 and pg2 are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3", "p4"},
				},
				{
					Name:                     "Verify pods for pg3 remain unschedulable",
					WaitForPodsUnschedulable: []string{"p5", "p6"},
				},
				{
					Name: "Verify scheduled pods are on rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "basic CPG picks the higher-scoring rack and stays there instead of spilling to a rack that fits",
			steps: []stepsframework.Step{
				{
					Name: "Create two nodes in rack-1 and one in rack-2 (2 CPU each)",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name:       "Fill node1-z1-r1 with a preexisting pod",
					CreatePods: []*v1.Pod{makeAssignedPod("existing", "node1-z1-r1", "2")},
				},
				{
					Name:                 "Verify the preexisting pod is bound",
					WaitForPodsScheduled: []string{"existing"},
				},
				{
					Name:                    "Create root CompositePodGroup (Basic policy, TopologyKey=rack)",
					CreateCompositePodGroup: makeBasicCompositePodGroup("cpg-root", "", "rack"),
				},
				{
					Name:           "Create child PodGroup pg1 (Basic policy, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg1", "cpg-root", ""),
				},
				{
					Name:           "Create child PodGroup pg2 (Basic policy, Parent=cpg-root)",
					CreatePodGroup: makeBasicPodGroupWithParent("pg2", "cpg-root", ""),
				},
				{
					// One pod first so the rack choice is a clean scoring decision: rack-1
					// (2+1)/4 = 0.75 beats rack-2 1/2 = 0.5 under MostAllocated, and
					// PodGroupPodsCount ties. With all pods at once both racks score 1.0.
					Name:       "Create p1 (pg1)",
					CreatePods: []*v1.Pod{makePod("p1", "pg1")},
				},
				{
					Name:                 "Verify p1 is scheduled",
					WaitForPodsScheduled: []string{"p1"},
				},
				{
					Name: "Verify p1 landed on rack-1, the higher-scoring rack",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
				{
					Name:       "Create p2 (pg1)",
					CreatePods: []*v1.Pod{makePod("p2", "pg1")},
				},
				{
					Name:                 "Verify p2 is scheduled",
					WaitForPodsScheduled: []string{"p2"},
				},
				{
					Name:       "Create p3 and p4 (pg2)",
					CreatePods: []*v1.Pod{makePod("p3", "pg2"), makePod("p4", "pg2")},
				},
				{
					// rack-2 has room for both, so only the CPG's rack constraint keeps them pending.
					Name:                     "Verify pg2 pods stay unschedulable instead of spilling to rack-2",
					WaitForPodsUnschedulable: []string{"p3", "p4"},
				},
				{
					Name:       "Delete the preexisting pod to free up rack-1",
					DeletePods: []string{"existing"},
				},
				{
					Name:                 "Verify pg2 pods are now scheduled",
					WaitForPodsScheduled: []string{"p3", "p4"},
				},
				{
					Name: "Verify all pods are on rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "two CPG hierarchies schedule consecutively on the same rack although another rack fits",
			steps: []stepsframework.Step{
				{
					// rack-1 is smaller, so MostAllocated prefers it for as long as it fits.
					Name: "Create 2 nodes in rack-1 and 3 in rack-2 (2 CPU each)",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name:                    "Create first root CompositePodGroup (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root1", "", "rack", 2),
				},
				{
					Name:           "Create pg1-1 (Gang minCount=1, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-1", "cpg-root1", "", 1),
				},
				{
					Name:           "Create pg1-2 (Gang minCount=1, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-2", "cpg-root1", "", 1),
				},
				{
					Name:       "Create pods of the first hierarchy (2 CPU total)",
					CreatePods: []*v1.Pod{makePod("p1", "pg1-1"), makePod("p2", "pg1-2")},
				},
				{
					Name:                 "Verify the first hierarchy is scheduled",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
				{
					// rack-1 2/4 = 0.5 vs rack-2 2/6 = 0.33.
					Name: "Verify the first hierarchy is on rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
				{
					Name:                    "Create second root CompositePodGroup (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root2", "", "rack", 2),
				},
				{
					Name:           "Create pg2-1 (Gang minCount=1, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-1", "cpg-root2", "", 1),
				},
				{
					Name:           "Create pg2-2 (Gang minCount=1, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-2", "cpg-root2", "", 1),
				},
				{
					Name:       "Create pods of the second hierarchy (2 CPU total)",
					CreatePods: []*v1.Pod{makePod("p3", "pg2-1"), makePod("p4", "pg2-2")},
				},
				{
					Name:                 "Verify the second hierarchy is scheduled",
					WaitForPodsScheduled: []string{"p3", "p4"},
				},
				{
					// rack-1 (2+2)/4 = 1.0 vs rack-2 2/6 = 0.33: sharing wins although rack-2 fits.
					Name: "Verify the second hierarchy joined the first on rack-1",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p3", "p4"},
						Nodes: sets.New("node1-z1-r1", "node2-z1-r1"),
					},
				},
			},
		},
		{
			name: "two CPG hierarchies schedule consecutively, each on a separate rack",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in two racks, each rack with 4 CPU capacity",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
					},
				},
				{
					Name:                    "Create first CompositePodGroup cpg-root1 (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root1", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1-1 (Gang minCount=2, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-1", "cpg-root1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg1-2 (Gang minCount=2, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-2", "cpg-root1", "", 2),
				},
				{
					Name: "Create pods for cpg-root1",
					CreatePods: []*v1.Pod{
						makePod("p1-1", "pg1-1"),
						makePod("p1-2", "pg1-1"),
						makePod("p1-3", "pg1-2"),
						makePod("p1-4", "pg1-2"),
					},
				},
				{
					Name:                 "Verify all pods for cpg-root1 are scheduled",
					WaitForPodsScheduled: []string{"p1-1", "p1-2", "p1-3", "p1-4"},
				},
				{
					Name: "Verify all pods for cpg-root1 are assigned in a single rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p1-1", "p1-2", "p1-3", "p1-4"},
						TopologyKey: "rack",
					},
				},
				{
					Name:                    "Create second CompositePodGroup cpg-root2 (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root2", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2-1 (Gang minCount=2, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-1", "cpg-root2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2-2 (Gang minCount=2, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-2", "cpg-root2", "", 2),
				},
				{
					Name: "Create pods for cpg-root2",
					CreatePods: []*v1.Pod{
						makePod("p2-1", "pg2-1"),
						makePod("p2-2", "pg2-1"),
						makePod("p2-3", "pg2-2"),
						makePod("p2-4", "pg2-2"),
					},
				},
				{
					Name:                 "Verify all pods for cpg-root2 are scheduled",
					WaitForPodsScheduled: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
				{
					Name: "Verify all pods for cpg-root2 are assigned in a single rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p2-1", "p2-2", "p2-3", "p2-4"},
						TopologyKey: "rack",
					},
				},
			},
		},
		{
			name: "two CPG hierarchies schedule consecutively, second remains pending when no additional rack is available",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in a single rack with 4 CPU capacity",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
					},
				},
				{
					Name:                    "Create first CompositePodGroup cpg-root1 (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root1", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1-1 (Gang minCount=2, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-1", "cpg-root1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg1-2 (Gang minCount=2, Parent=cpg-root1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1-2", "cpg-root1", "", 2),
				},
				{
					Name: "Create pods for cpg-root1",
					CreatePods: []*v1.Pod{
						makePod("p1-1", "pg1-1"),
						makePod("p1-2", "pg1-1"),
						makePod("p1-3", "pg1-2"),
						makePod("p1-4", "pg1-2"),
					},
				},
				{
					Name:                 "Verify all pods for cpg-root1 are scheduled",
					WaitForPodsScheduled: []string{"p1-1", "p1-2", "p1-3", "p1-4"},
				},
				{
					Name:                    "Create second CompositePodGroup cpg-root2 (Gang minGroupCount=2, TopologyKey=rack)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root2", "", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg2-1 (Gang minCount=2, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-1", "cpg-root2", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2-2 (Gang minCount=2, Parent=cpg-root2)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2-2", "cpg-root2", "", 2),
				},
				{
					Name: "Create pods for cpg-root2",
					CreatePods: []*v1.Pod{
						makePod("p2-1", "pg2-1"),
						makePod("p2-2", "pg2-1"),
						makePod("p2-3", "pg2-2"),
						makePod("p2-4", "pg2-2"),
					},
				},
				{
					Name:                     "Verify pods for cpg-root2 remain unschedulable",
					WaitForPodsUnschedulable: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
			},
		},
		{
			name: "CPG hierarchy with mixed sub-CPG and direct child PGs and different topology constraints",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes across zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z1-r2", "rack-2", "zone-1"),
						makeNode("node4-z1-r2", "rack-2", "zone-1"),
						makeNode("node5-z2-r1", "rack-1", "zone-2"),
						makeNode("node6-z2-r2", "rack-2", "zone-2"),
					},
				},
				{
					Name:                    "Create root CompositePodGroup (Gang minGroupCount=3, TopologyKey=zone)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-root", "", "zone", 3),
				},
				{
					Name:                    "Create sub CompositePodGroup cpg-sub1 (Gang minGroupCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreateCompositePodGroup: makeGangCompositePodGroup("cpg-sub1", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create child PodGroup pg1 (Gang minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg1", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create child PodGroup pg2 (Gang minCount=2, without topology constraints, Parent=cpg-sub1)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-sub1", "", 2),
				},
				{
					Name:           "Create direct child PodGroup pg3 (Gang minCount=2, TopologyKey=rack, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg3", "cpg-root", "rack", 2),
				},
				{
					Name:           "Create direct child PodGroup pg4 (Gang minCount=2, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg4", "cpg-root", "", 2),
				},
				{
					Name: "Create all pods across pg1, pg2, pg3, and pg4 (8 pods total)",
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
					Name:                 "Verify all 8 pods across all child groups are scheduled",
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
					Name: "Verify pg3 pods are assigned in the same rack",
					VerifyAssignedInOneDomain: &stepsframework.VerifyAssignedInOneDomain{
						Pods:        []string{"p5", "p6"},
						TopologyKey: "rack",
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
			// Every feasible rack can be freed by evicting a single priority-10 pod, so neither victim
			// count nor victim priority can explain the choice; only placement scoring can. Pod group
			// preemption removes all potential victims, schedules the hierarchy once (scoring included)
			// and then reprieves whatever still fits, so the scores are computed without low-priority pods.
			name: "parent CPG has topology constraints, children do not; preemption chooses the rack with the highest allocation percentage among several feasible racks",
			steps: []stepsframework.Step{
				{
					Name: "Create nodes in multiple zones and racks",
					CreateNodes: []*v1.Node{
						makeNode("node1-z1-r1", "rack-1", "zone-1"),
						makeNode("node2-z1-r1", "rack-1", "zone-1"),
						makeNode("node3-z2-r2", "rack-2", "zone-2"),
						makeNode("node4-z2-r2", "rack-2", "zone-2"),
						makeNode("node5-z2-r2", "rack-2", "zone-2"),
						makeNode("node6-z2-r3", "rack-3", "zone-2"),
						makeNode("node7-z2-r3", "rack-3", "zone-2"),
						makeNode("node8-z2-r3", "rack-3", "zone-2"),
						makeNode("node9-z2-r4", "rack-4", "zone-2"),
					},
				},
				{
					Name: "Create non-preemptible pods and low-priority pods, leaving no rack able to fit 3 CPU without preemption",
					CreatePods: []*v1.Pod{
						makeAssignedPod("existing1", "node3-z2-r2", "2"),
						makeAssignedPod("existing2", "node8-z2-r3", "1"),
						makeAssignedPodWithPriority("low1-z1-r1", "node1-z1-r1", "2", 10),
						makeAssignedPodWithPriority("low1-z2-r2", "node4-z2-r2", "2", 10),
						makeAssignedPodWithPriority("low1-z2-r3", "node6-z2-r3", "2", 10),
						makeAssignedPodWithPriority("low2-z2-r3", "node8-z2-r3", "1", 10),
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
					Name:           "Create child PodGroup pg2 (Gang with minCount=1, without topology constraints, Parent=cpg-root)",
					CreatePodGroup: makeGangPodGroupWithParent("pg2", "cpg-root", "", 1),
				},
				{
					Name: "Create all pods belonging to pg1 and pg2 (3 CPU in total)",
					CreatePods: []*v1.Pod{
						makePod("p1", "pg1"),
						makePod("p2", "pg1"),
						makePod("p3", "pg2"),
					},
				},
				{
					Name:                 "Verify all pods in the composite group are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2", "p3"},
				},
				{
					Name:               "Verify the low-priority pod on rack-2 is removed via preemption",
					WaitForPodsRemoved: []string{"low1-z2-r2"},
				},
				// Allocation fractions with all potential victims removed (MostAllocated placement scoring):
				// - rack-1: (0 + 3)/4 = 0.75
				// - rack-2: (2 + 3)/6 = 0.833
				// - rack-3: (1 + 3)/6 = 0.667
				// - rack-4: infeasible (only 2 CPUs in total)
				// PodGroupPodsCount scores all feasible racks the same, since all 3 pods fit in each.
				{
					Name: "Verify all pods are scheduled on rack-2",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"p1", "p2", "p3"},
						Nodes: sets.New("node3-z2-r2", "node4-z2-r2", "node5-z2-r2"),
					},
				},
				{
					Name: "Verify the low-priority pods on the other racks are not evicted",
					VerifyAssignments: &stepsframework.VerifyAssignments{
						Pods:  []string{"low1-z1-r1", "low1-z2-r3", "low2-z2-r3"},
						Nodes: sets.New("node1-z1-r1", "node6-z2-r3", "node8-z2-r3"),
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
