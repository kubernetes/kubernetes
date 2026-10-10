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

package podgroup

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

func TestCPGQueueing(t *testing.T) {
	node := st.MakeNode().Name("node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Obj()

	pg1 := st.MakePodGroup().Name("pg1").WorkloadRef("w1", "pg-t").MinCount(2).ParentCompositePodGroup("cpg-mid").Obj()
	cpgMid := st.MakeCompositePodGroup().Name("cpg-mid").WorkloadRef("w1", "cpg-mid-t").MinGroupCount(1).ParentCompositePodGroup("cpg-root").Obj()
	cpgRoot := st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("w1", "cpg-t").MinGroupCount(1).Obj()

	p1 := st.MakePod().Name("p1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Obj()
	p2 := st.MakePod().Name("p2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg1").Obj()

	cpgRoot2 := st.MakeCompositePodGroup().Name("cpg-root2").WorkloadRef("w2", "cpg-t2").MinGroupCount(2).Obj()
	pg2_1 := st.MakePodGroup().Name("pg2-1").WorkloadRef("w2", "pg-t2-1").MinCount(2).ParentCompositePodGroup("cpg-root2").Obj()
	pg2_2 := st.MakePodGroup().Name("pg2-2").WorkloadRef("w2", "pg-t2-2").MinCount(2).ParentCompositePodGroup("cpg-root2").Obj()

	p2_1 := st.MakePod().Name("p2-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-1").Obj()
	p2_2 := st.MakePod().Name("p2-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-1").Obj()
	p2_3 := st.MakePod().Name("p2-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-2").Obj()
	p2_4 := st.MakePod().Name("p2-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-2").Obj()

	// cpgMid3 is created with MinGroupCount=2, but only one child PodGroup (pg3_1) is intentionally created
	// so the CPG remains unschedulable until MinGroupCount is reduced to 1 in the test.
	cpgRoot3 := st.MakeCompositePodGroup().Name("cpg-root3").WorkloadRef("w3", "cpg-t3").MinGroupCount(1).Obj()
	cpgMid3 := st.MakeCompositePodGroup().Name("cpg-mid3").WorkloadRef("w3", "cpg-mid3-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root3").Obj()
	pg3_1 := st.MakePodGroup().Name("pg3-1").WorkloadRef("w3", "pg-t3-1").MinCount(2).ParentCompositePodGroup("cpg-mid3").Obj()

	p3_1 := st.MakePod().Name("p3-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg3-1").Obj()
	p3_2 := st.MakePod().Name("p3-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg3-1").Obj()

	cpgRoot4 := st.MakeCompositePodGroup().Name("cpg-root4").WorkloadRef("w4", "cpg-t4").MinGroupCount(3).Obj()
	pg4_1 := st.MakePodGroup().Name("pg4-1").WorkloadRef("w4", "pg-t4-1").MinCount(1).ParentCompositePodGroup("cpg-root4").Obj()
	pg4_2 := st.MakePodGroup().Name("pg4-2").WorkloadRef("w4", "pg-t4-2").MinCount(1).ParentCompositePodGroup("cpg-root4").Obj()

	p4_1 := st.MakePod().Name("p4-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg4-1").Obj()
	p4_2 := st.MakePod().Name("p4-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg4-2").Obj()

	// Cousin PodGroups in different CPG branches under same root CPG:
	cpgRootCousin := st.MakeCompositePodGroup().Name("cpg-root-cousin").WorkloadRef("w-cousin", "cpg-root-t").MinGroupCount(2).Obj()
	cpgBranch1 := st.MakeCompositePodGroup().Name("cpg-branch1").WorkloadRef("w-cousin", "cpg-b1-t").MinGroupCount(1).ParentCompositePodGroup("cpg-root-cousin").Obj()
	cpgBranch2 := st.MakeCompositePodGroup().Name("cpg-branch2").WorkloadRef("w-cousin", "cpg-b2-t").MinGroupCount(1).ParentCompositePodGroup("cpg-root-cousin").Obj()
	pgBranch1 := st.MakePodGroup().Name("pg-b1").WorkloadRef("w-cousin", "pg-b1-t").MinCount(1).ParentCompositePodGroup("cpg-branch1").Obj()
	pgBranch2 := st.MakePodGroup().Name("pg-b2").WorkloadRef("w-cousin", "pg-b2-t").MinCount(2).ParentCompositePodGroup("cpg-branch2").Obj()

	pB1 := st.MakePod().Name("p-b1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-b1").Obj()
	pB2 := st.MakePod().Name("p-b2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-b2").Obj()

	// CPG schedulable initially with an unready subtree:
	cpgRootPartialTree := st.MakeCompositePodGroup().Name("cpg-root-ptree").WorkloadRef("w-ptree", "cpg-root-t").MinGroupCount(1).Obj()
	pgMainSched := st.MakePodGroup().Name("pg-main-ptree").WorkloadRef("w-ptree", "pg-main-t").MinCount(1).ParentCompositePodGroup("cpg-root-ptree").Obj()
	cpgSubUnready := st.MakeCompositePodGroup().Name("cpg-sub-ptree").WorkloadRef("w-ptree", "cpg-sub-t").MinGroupCount(2).ParentCompositePodGroup("cpg-root-ptree").Obj()
	pgSubUnready1 := st.MakePodGroup().Name("pg-sub-ptree-1").WorkloadRef("w-ptree", "pg-sub1-t").MinCount(1).ParentCompositePodGroup("cpg-sub-ptree").Obj()
	pgSubUnready2 := st.MakePodGroup().Name("pg-sub-ptree-2").WorkloadRef("w-ptree", "pg-sub2-t").MinCount(2).ParentCompositePodGroup("cpg-sub-ptree").Obj()

	pMainSched := st.MakePod().Name("p-main-ptree").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-main-ptree").Obj()
	pSubUnready1 := st.MakePod().Name("p-sub-ptree-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-sub-ptree-1").Obj()
	pSubUnready2 := st.MakePod().Name("p-sub-ptree-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg-sub-ptree-2").Obj()

	taintedNode := st.MakeNode().Name("tainted-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Taints([]v1.Taint{{Key: "dedicated", Value: "special", Effect: v1.TaintEffectNoSchedule}}).Obj()

	blockingPod := st.MakePod().Name("blocking-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Container("image").Obj()

	cpgRoot5 := st.MakeCompositePodGroup().Name("cpg-root5").WorkloadRef("w5", "cpg-t5").MinGroupCount(3).Obj()
	pg5_1 := st.MakePodGroup().Name("pg5-1").WorkloadRef("w5", "pg-t5-1").MinCount(2).ParentCompositePodGroup("cpg-root5").Obj()
	pg5_2 := st.MakePodGroup().Name("pg5-2").WorkloadRef("w5", "pg-t5-2").MinCount(2).ParentCompositePodGroup("cpg-root5").Obj()
	pg5_3 := st.MakePodGroup().Name("pg5-3").WorkloadRef("w5", "pg-t5-3").MinCount(2).ParentCompositePodGroup("cpg-root5").Obj()
	pg5_4 := st.MakePodGroup().Name("pg5-4").WorkloadRef("w5", "pg-t5-4").MinCount(2).ParentCompositePodGroup("cpg-root5").Obj()
	p5_1 := st.MakePod().Name("p5-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-1").Obj()
	p5_2 := st.MakePod().Name("p5-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-1").Obj()
	p5_3 := st.MakePod().Name("p5-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-2").Obj()
	p5_4 := st.MakePod().Name("p5-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-2").Obj()
	p5_5 := st.MakePod().Name("p5-5").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-3").Obj()
	p5_6 := st.MakePod().Name("p5-6").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-3").Obj()
	p5_7 := st.MakePod().Name("p5-7").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-4").Obj()
	p5_8 := st.MakePod().Name("p5-8").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg5-4").Obj()

	// cpgRootCyclic forms a dependency cycle with cpgMid (cpg-root -> cpg-mid -> cpg-root) to test hierarchy loop detection.
	cpgRootCyclic := st.MakeCompositePodGroup().Name("cpg-root").WorkloadRef("w1", "cpg-t").MinGroupCount(1).ParentCompositePodGroup("cpg-mid").Obj()

	tests := []struct {
		name  string
		steps []stepsframework.Step
	}{
		{
			name: "Incomplete CPG tree buffers pods and missing root CPG wakes them up",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:           "Create PodGroup",
					CreatePodGroup: pg1,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:       "Create member pods",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                                "Verify pods are unschedulable due to incomplete CPG hierarchy",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2"},
				},
				{
					Name:                    "Create the missing root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                 "Verify pods get scheduled successfully after root CPG is added",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Node add triggers queueing hint for complete CPG tree from unschedulableQ",
			steps: []stepsframework.Step{
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:           "Create PodGroup",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create member pods while there are no nodes",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                     "Verify pods are unschedulable (no nodes) and move to unschedulableQ",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
				{
					Name:        "Add a node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                 "Verify pods get scheduled successfully after node is added, moving tree from unschedulableQ to activeQ",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "PodGroup add triggers queueing hint for unschedulable CPG root",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 2 groups",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:           "Create first PodGroup",
					CreatePodGroup: pg2_1,
				},
				{
					Name:       "Create all pods",
					CreatePods: []*v1.Pod{p2_1, p2_2, p2_3, p2_4},
				},
				{
					Name:                               "Verify first group pods are in unschedulableEntities due to root needing 2 groups",
					WaitForPodsInUnschedulableEntities: []string{"p2-1", "p2-2"},
				},
				{
					Name:                                "Verify second group pods are in incomplete entities due to missing PodGroup",
					WaitForPodsInIncompletePodGroupPods: []string{"p2-3", "p2-4"},
				},
				{
					Name:           "Create second PodGroup, waking up root CPG",
					CreatePodGroup: pg2_2,
				},
				{
					Name:                 "Verify all pods get scheduled after root becomes schedulable",
					WaitForPodsScheduled: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
			},
		},
		{
			name: "Pod add triggers queueing hint for unschedulable CPG root",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 2 groups",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:           "Create first PodGroup",
					CreatePodGroup: pg2_1,
				},
				{
					Name:           "Create second PodGroup",
					CreatePodGroup: pg2_2,
				},
				{
					Name:       "Create 3 out of 4 pods",
					CreatePods: []*v1.Pod{p2_1, p2_2, p2_3},
				},
				{
					Name:                               "Verify all 3 pods are in unschedulableEntities because root is missing one pod in pg2-2",
					WaitForPodsInUnschedulableEntities: []string{"p2-1", "p2-2", "p2-3"},
				},
				{
					Name:       "Create final pod, completing the tree",
					CreatePods: []*v1.Pod{p2_4},
				},
				{
					Name:                 "Verify all pods get scheduled after root becomes schedulable",
					WaitForPodsScheduled: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
			},
		},
		{
			name: "PodGroup update (reduce minCount) triggers queueing hint for unschedulable CPG tree",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:           "Create PodGroup with MinCount=2",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create 1 out of 2 pods",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                               "Verify pod is in unschedulableEntities because PodGroup requires 2 pods",
					WaitForPodsInUnschedulableEntities: []string{"p1"},
				},
				{
					Name:           "Update PodGroup to MinCount=1",
					UpdatePodGroup: st.MakePodGroup().Name("pg1").WorkloadRef("w1", "pg-t").MinCount(1).ParentCompositePodGroup("cpg-mid").Obj(),
				},
				{
					Name:                 "Verify pod gets scheduled after PodGroup MinCount is reduced",
					WaitForPodsScheduled: []string{"p1"},
				},
			},
		},
		{
			name: "CPG update (reduce minGroupCount) triggers queueing hint for unschedulable CPG tree",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 2 groups",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:           "Create first PodGroup",
					CreatePodGroup: pg2_1,
				},
				{
					Name:       "Create all pods of first PodGroup",
					CreatePods: []*v1.Pod{p2_1, p2_2},
				},
				{
					Name:                               "Verify first group pods are in unschedulableEntities due to root needing 2 groups",
					WaitForPodsInUnschedulableEntities: []string{"p2-1", "p2-2"},
				},
				{
					Name:                    "Update root CPG to MinGroupCount=1",
					UpdateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root2").WorkloadRef("w2", "cpg-t2").MinGroupCount(1).Obj(),
				},
				{
					Name:                 "Verify pods get scheduled after root CPG MinGroupCount is reduced",
					WaitForPodsScheduled: []string{"p2-1", "p2-2"},
				},
			},
		},
		{
			name: "Intermediate CPG update (reduce minGroupCount) triggers queueing hint for unschedulable CPG tree",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 1 group",
					CreateCompositePodGroup: cpgRoot3,
				},
				{
					Name:                    "Create intermediate CPG requiring 2 groups",
					CreateCompositePodGroup: cpgMid3,
				},
				{
					Name:           "Create first PodGroup under intermediate CPG",
					CreatePodGroup: pg3_1,
				},
				{
					Name:       "Create all pods of first PodGroup",
					CreatePods: []*v1.Pod{p3_1, p3_2},
				},
				{
					Name:                               "Verify pods are in unschedulableEntities due to intermediate CPG needing 2 groups",
					WaitForPodsInUnschedulableEntities: []string{"p3-1", "p3-2"},
				},
				{
					Name:                    "Update intermediate CPG to MinGroupCount=1",
					UpdateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-mid3").WorkloadRef("w3", "cpg-mid3-t").MinGroupCount(1).ParentCompositePodGroup("cpg-root3").Obj(),
				},
				{
					Name:                 "Verify pods get scheduled after intermediate CPG MinGroupCount is reduced",
					WaitForPodsScheduled: []string{"p3-1", "p3-2"},
				},
			},
		},
		{
			name: "CPG update (reduce minGroupCount) unblocks multiple sibling PodGroups simultaneously",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 3 groups",
					CreateCompositePodGroup: cpgRoot4,
				},
				{
					Name:           "Create first sibling PodGroup",
					CreatePodGroup: pg4_1,
				},
				{
					Name:           "Create second sibling PodGroup",
					CreatePodGroup: pg4_2,
				},
				{
					Name:       "Create pods for both sibling PodGroups",
					CreatePods: []*v1.Pod{p4_1, p4_2},
				},
				{
					Name:                               "Verify both sibling pods are in unschedulableEntities due to root needing 3 groups",
					WaitForPodsInUnschedulableEntities: []string{"p4-1", "p4-2"},
				},
				{
					Name:                    "Update root CPG to MinGroupCount=2, satisfying both sibling groups",
					UpdateCompositePodGroup: st.MakeCompositePodGroup().Name("cpg-root4").WorkloadRef("w4", "cpg-t4").MinGroupCount(2).Obj(),
				},
				{
					Name:                 "Verify all sibling pods get scheduled after root CPG MinGroupCount is reduced",
					WaitForPodsScheduled: []string{"p4-1", "p4-2"},
				},
			},
		},
		{
			name: "PodGroup update (reduce minCount) in child branch unblocks entire CPG tree",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 2 branches",
					CreateCompositePodGroup: cpgRootCousin,
				},
				{
					Name:                    "Create first branch CPG",
					CreateCompositePodGroup: cpgBranch1,
				},
				{
					Name:                    "Create second branch CPG",
					CreateCompositePodGroup: cpgBranch2,
				},
				{
					Name:           "Create PodGroup in first branch",
					CreatePodGroup: pgBranch1,
				},
				{
					Name:           "Create PodGroup in second branch",
					CreatePodGroup: pgBranch2,
				},
				{
					Name:       "Create 1 pod in first branch and 1 pod in second branch (which needs 2)",
					CreatePods: []*v1.Pod{pB1, pB2},
				},
				{
					Name:                               "Verify both cousin pods are in unschedulableEntities because branch2 is missing 1 pod",
					WaitForPodsInUnschedulableEntities: []string{"p-b1", "p-b2"},
				},
				{
					Name:           "Update branch2 PodGroup to MinCount=1",
					UpdatePodGroup: st.MakePodGroup().Name("pg-b2").WorkloadRef("w-cousin", "pg-b2-t").MinCount(1).ParentCompositePodGroup("cpg-branch2").Obj(),
				},
				{
					Name:                 "Verify all cousin pods get scheduled after branch2 MinCount is reduced",
					WaitForPodsScheduled: []string{"p-b1", "p-b2"},
				},
			},
		},
		{
			name: "CPG schedules ready branch while unready subtree waits, then reducing child minCount unblocks and schedules the subtree",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 1 group (schedulable via main branch)",
					CreateCompositePodGroup: cpgRootPartialTree,
				},
				{
					Name:           "Create main ready PodGroup",
					CreatePodGroup: pgMainSched,
				},
				{
					Name:                    "Create subtree CPG requiring 2 child groups",
					CreateCompositePodGroup: cpgSubUnready,
				},
				{
					Name:           "Create first child PodGroup in subtree",
					CreatePodGroup: pgSubUnready1,
				},
				{
					Name:           "Create second child PodGroup in subtree",
					CreatePodGroup: pgSubUnready2,
				},
				{
					Name:       "Create pods across all groups (1 for main, 1 for sub1, 1 for sub2 which needs 2)",
					CreatePods: []*v1.Pod{pMainSched, pSubUnready1, pSubUnready2},
				},
				{
					Name:                 "Verify main pod gets scheduled because root CPG minGroupCount=1 is satisfied",
					WaitForPodsScheduled: []string{"p-main-ptree"},
				},
				{
					Name:                     "Verify subtree pods remain unschedulable because subtree minGroupCount=2 is not met",
					WaitForPodsUnschedulable: []string{"p-sub-ptree-1", "p-sub-ptree-2"},
				},
				{
					Name:           "Update second child PodGroup in subtree to MinCount=1",
					UpdatePodGroup: st.MakePodGroup().Name("pg-sub-ptree-2").WorkloadRef("w-ptree", "pg-sub2-t").MinCount(1).ParentCompositePodGroup("cpg-sub-ptree").Obj(),
				},
				{
					Name:                 "Verify subtree pods get scheduled once the subtree minGroupCount is satisfied",
					WaitForPodsScheduled: []string{"p-sub-ptree-1", "p-sub-ptree-2"},
				},
			},
		},
		{
			name: "Node update (taint removal) triggers queueing hint for complete CPG tree from unschedulableQ",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial tainted node",
					CreateNodes: []*v1.Node{taintedNode},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:           "Create PodGroup",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create member pods while node is tainted",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                     "Verify pods are unschedulable (taint mismatch) and move to unschedulableQ",
					WaitForPodsUnschedulable: []string{"p1", "p2"},
				},
				{
					Name: "Remove taint from node",
					UpdateNode: &stepsframework.UpdateNode{
						NodeName: "tainted-node",
						ModifyFn: func(n *v1.Node) {
							n.Spec.Taints = nil
						},
					},
				},
				{
					Name:                 "Verify pods get scheduled successfully after taint is removed",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Intermediate CPG creation triggers queueing hint for incomplete CPG hierarchy",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:           "Create leaf PodGroup referencing missing intermediate CPG",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create member pods",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                                "Verify pods are buffered in incompletePodGroupPods due to missing intermediate CPG",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2"},
				},
				{
					Name:                    "Create the missing intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:                 "Verify pods get scheduled successfully after intermediate CPG completes the tree",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Pod deletion triggers queueing hint for complete CPG tree from unschedulableQ",
			steps: []stepsframework.Step{
				{
					Name:        "Create initial node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:       "Create blocking pod consuming entire node capacity",
					CreatePods: []*v1.Pod{blockingPod},
				},
				{
					Name:                 "Wait for blocking pod to be scheduled",
					WaitForPodsScheduled: []string{"blocking-pod"},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:           "Create first leaf PodGroup",
					CreatePodGroup: pg2_1,
				},
				{
					Name:           "Create second leaf PodGroup",
					CreatePodGroup: pg2_2,
				},
				{
					Name:       "Create member pods for CPG tree",
					CreatePods: []*v1.Pod{p2_1, p2_2, p2_3, p2_4},
				},
				{
					Name:                     "Verify CPG member pods are unschedulable due to saturated node",
					WaitForPodsUnschedulable: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
				{
					Name:       "Delete blocking pod to free node capacity",
					DeletePods: []string{"blocking-pod"},
				},
				{
					Name:                 "Verify CPG member pods are scheduled after blocking pod deletion",
					WaitForPodsScheduled: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
			},
		},
		{
			name: "Intermediate CPG deletion transitions child pods to incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:        "Create node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:           "Create leaf PodGroup with MinCount=2",
					CreatePodGroup: pg1,
				},
				{
					// The pod has to be held back by the PodGroup quorum (PreEnqueue). A pod group
					// rejected by Filter plugins (or by missing nodes) is requeued to activeQ/backoffQ
					// and never parks in unschedulableEntities.
					Name:       "Create 1 out of 2 pods",
					CreatePods: []*v1.Pod{p1},
				},
				{
					Name:                               "Verify pod is in unschedulableEntities because PodGroup requires 2 pods",
					WaitForPodsInUnschedulableEntities: []string{"p1"},
				},
				{
					Name:                    "Delete intermediate CPG",
					DeleteCompositePodGroup: "cpg-mid",
				},
				{
					Name:                                "Verify pod transitions to incompletePodGroupPods due to broken hierarchy",
					WaitForPodsInIncompletePodGroupPods: []string{"p1"},
				},
				{
					Name:                    "Re-create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:                               "Verify pod moves back to unschedulableEntities",
					WaitForPodsInUnschedulableEntities: []string{"p1"},
				},
				{
					Name:       "Create the second pod to satisfy the PodGroup quorum",
					CreatePods: []*v1.Pod{p2},
				},
				{
					Name:                 "Verify pods are scheduled",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
		{
			name: "Root CPG deletion transitions descendant pods from unschedulableQ to incompletePodGroupPods",
			steps: []stepsframework.Step{
				{
					Name:        "Create node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 2 child groups",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:           "Create first leaf PodGroup",
					CreatePodGroup: pg2_1,
				},
				{
					Name:       "Create pods for first leaf group",
					CreatePods: []*v1.Pod{p2_1, p2_2},
				},
				{
					Name:                               "Verify pods are in unschedulableEntities because root CPG requires 2 groups",
					WaitForPodsInUnschedulableEntities: []string{"p2-1", "p2-2"},
				},
				{
					Name:                    "Delete root CPG",
					DeleteCompositePodGroup: "cpg-root2",
				},
				{
					Name:                                "Verify pods transition to incompletePodGroupPods after root CPG deletion",
					WaitForPodsInIncompletePodGroupPods: []string{"p2-1", "p2-2"},
				},
				{
					Name:                    "Re-create root CPG",
					CreateCompositePodGroup: cpgRoot2,
				},
				{
					Name:                               "Verify pods transition back to unschedulableEntities as tree is restored",
					WaitForPodsInUnschedulableEntities: []string{"p2-1", "p2-2"},
				},
				{
					Name:           "Create second leaf PodGroup under root CPG",
					CreatePodGroup: pg2_2,
				},
				{
					Name:       "Create pods for second leaf group, satisfying quorum",
					CreatePods: []*v1.Pod{p2_3, p2_4},
				},
				{
					Name:                 "Verify all pods are scheduled after quorum is satisfied",
					WaitForPodsScheduled: []string{"p2-1", "p2-2", "p2-3", "p2-4"},
				},
			},
		},
		{
			name: "Leaf PodGroup deletion in multi-branch CPG preserves sibling subtree in unschedulableQ",
			steps: []stepsframework.Step{
				{
					Name:        "Create node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create root CPG requiring 3 child groups",
					CreateCompositePodGroup: cpgRoot5,
				},
				{
					Name:           "Create first leaf PodGroup",
					CreatePodGroup: pg5_1,
				},
				{
					Name:           "Create second leaf PodGroup",
					CreatePodGroup: pg5_2,
				},
				{
					Name:       "Create pods for first two leaf groups",
					CreatePods: []*v1.Pod{p5_1, p5_2, p5_3, p5_4},
				},
				{
					Name:                               "Verify all pods are in unschedulableEntities because root CPG requires 3 groups",
					WaitForPodsInUnschedulableEntities: []string{"p5-1", "p5-2", "p5-3", "p5-4"},
				},
				{
					Name:           "Delete first leaf PodGroup",
					DeletePodGroup: "pg5-1",
				},
				{
					Name:                                "Verify deleted group pods transition to incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p5-1", "p5-2"},
				},
				{
					Name:                               "Verify sibling group pods remain in unschedulableEntities",
					WaitForPodsInUnschedulableEntities: []string{"p5-3", "p5-4"},
				},
				{
					Name:           "Create third leaf PodGroup",
					CreatePodGroup: pg5_3,
				},
				{
					Name:       "Create pods for third leaf group",
					CreatePods: []*v1.Pod{p5_5, p5_6},
				},
				{
					Name:           "Create fourth leaf PodGroup",
					CreatePodGroup: pg5_4,
				},
				{
					Name:       "Create pods for fourth leaf group, completing 3 groups under root CPG",
					CreatePods: []*v1.Pod{p5_7, p5_8},
				},
				{
					Name:                 "Verify remaining groups are scheduled after quorum of 3 is satisfied",
					WaitForPodsScheduled: []string{"p5-3", "p5-4", "p5-5", "p5-6", "p5-7", "p5-8"},
				},
				{
					Name:                                "Verify pods of the deleted PodGroup are still in incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p5-1", "p5-2"},
				},
			},
		},
		{
			name: "Cyclic CPG hierarchy loop protection buffers pods in incompletePodGroupPods until cycle is broken",
			steps: []stepsframework.Step{
				{
					Name:        "Create node",
					CreateNodes: []*v1.Node{node},
				},
				{
					Name:                    "Create CPG mid referencing root CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:                    "Create root CPG referencing mid CPG (forming cyclic dependency)",
					CreateCompositePodGroup: cpgRootCyclic,
				},
				{
					Name:           "Create leaf PodGroup referencing mid CPG",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create member pods",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                                "Verify pods are held in incompletePodGroupPods due to detected cyclic hierarchy",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2"},
				},
				{
					Name:                    "Delete cyclic root CPG",
					DeleteCompositePodGroup: "cpg-root",
				},
				{
					Name:                    "Re-create root CPG without parent, establishing it as legitimate root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                 "Verify pods are scheduled after cycle is broken and hierarchy is resolved",
					WaitForPodsScheduled: []string{"p1", "p2"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
				features.GenericWorkload:                 true,
				features.TopologyAwareWorkloadScheduling: true,
				features.CompositePodGroup:               true,
			})

			testCtx := testutils.InitTestSchedulerWithOptions(
				t,
				testutils.InitTestAPIServer(t, "podgroup-queueing", nil),
				0,
				scheduler.WithPodInitialBackoffSeconds(0),
				scheduler.WithPodMaxBackoffSeconds(0),
			)
			testutils.SyncSchedulerInformerFactory(testCtx)
			go testCtx.Scheduler.Run(testCtx.SchedulerCtx)
			ns := testCtx.NS.Name

			if err := stepsframework.RunSteps(testCtx, t, ns, tt.steps); err != nil {
				t.Fatal(err)
			}
		})
	}
}
