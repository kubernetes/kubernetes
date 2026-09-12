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

	taintedNode := st.MakeNode().Name("tainted-node").Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Taints([]v1.Taint{{Key: "dedicated", Value: "special", Effect: v1.TaintEffectNoSchedule}}).Obj()

	p2_1 := st.MakePod().Name("p2-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-1").Obj()
	p2_2 := st.MakePod().Name("p2-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-1").Obj()
	p2_3 := st.MakePod().Name("p2-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-2").Obj()
	p2_4 := st.MakePod().Name("p2-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Container("image").PodGroupName("pg2-2").Obj()

	blockingPod := st.MakePod().Name("blocking-pod").Req(map[v1.ResourceName]string{v1.ResourceCPU: "8"}).Container("image").Obj()

	cpgRoot3 := st.MakeCompositePodGroup().Name("cpg-root3").WorkloadRef("w3", "cpg-t3").MinGroupCount(3).Obj()
	pg3_1 := st.MakePodGroup().Name("pg3-1").WorkloadRef("w3", "pg-t3-1").MinCount(2).ParentCompositePodGroup("cpg-root3").Obj()
	pg3_2 := st.MakePodGroup().Name("pg3-2").WorkloadRef("w3", "pg-t3-2").MinCount(2).ParentCompositePodGroup("cpg-root3").Obj()
	pg3_3 := st.MakePodGroup().Name("pg3-3").WorkloadRef("w3", "pg-t3-3").MinCount(2).ParentCompositePodGroup("cpg-root3").Obj()
	pg3_4 := st.MakePodGroup().Name("pg3-4").WorkloadRef("w3", "pg-t3-4").MinCount(2).ParentCompositePodGroup("cpg-root3").Obj()
	p3_1 := st.MakePod().Name("p3-1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-1").Obj()
	p3_2 := st.MakePod().Name("p3-2").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-1").Obj()
	p3_3 := st.MakePod().Name("p3-3").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-2").Obj()
	p3_4 := st.MakePod().Name("p3-4").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-2").Obj()
	p3_5 := st.MakePod().Name("p3-5").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-3").Obj()
	p3_6 := st.MakePod().Name("p3-6").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-3").Obj()
	p3_7 := st.MakePod().Name("p3-7").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-4").Obj()
	p3_8 := st.MakePod().Name("p3-8").Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).Container("image").PodGroupName("pg3-4").Obj()

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
					Name:                    "Create root CPG",
					CreateCompositePodGroup: cpgRoot,
				},
				{
					Name:                    "Create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:           "Create leaf PodGroup",
					CreatePodGroup: pg1,
				},
				{
					Name:       "Create member pods",
					CreatePods: []*v1.Pod{p1, p2},
				},
				{
					Name:                 "Verify pods are in activeQ",
					WaitForPodsInActiveQ: []string{"p1", "p2"},
				},
				{
					Name:                    "Delete intermediate CPG",
					DeleteCompositePodGroup: "cpg-mid",
				},
				{
					Name:                                "Verify pods transition to incompletePodGroupPods due to broken hierarchy",
					WaitForPodsInIncompletePodGroupPods: []string{"p1", "p2"},
				},
				{
					Name:                    "Re-create intermediate CPG",
					CreateCompositePodGroup: cpgMid,
				},
				{
					Name:                 "Verify pods transition back to activeQ",
					WaitForPodsInActiveQ: []string{"p1", "p2"},
				},
				{
					Name:        "Create node to allow scheduling",
					CreateNodes: []*v1.Node{node},
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
					CreateCompositePodGroup: cpgRoot3,
				},
				{
					Name:           "Create first leaf PodGroup",
					CreatePodGroup: pg3_1,
				},
				{
					Name:           "Create second leaf PodGroup",
					CreatePodGroup: pg3_2,
				},
				{
					Name:       "Create pods for first two leaf groups",
					CreatePods: []*v1.Pod{p3_1, p3_2, p3_3, p3_4},
				},
				{
					Name:                               "Verify all pods are in unschedulableEntities because root CPG requires 3 groups",
					WaitForPodsInUnschedulableEntities: []string{"p3-1", "p3-2", "p3-3", "p3-4"},
				},
				{
					Name:           "Delete first leaf PodGroup",
					DeletePodGroup: "pg3-1",
				},
				{
					Name:                                "Verify deleted group pods transition to incompletePodGroupPods",
					WaitForPodsInIncompletePodGroupPods: []string{"p3-1", "p3-2"},
				},
				{
					Name:                               "Verify sibling group pods remain in unschedulableEntities",
					WaitForPodsInUnschedulableEntities: []string{"p3-3", "p3-4"},
				},
				{
					Name:           "Create third leaf PodGroup",
					CreatePodGroup: pg3_3,
				},
				{
					Name:       "Create pods for third leaf group",
					CreatePods: []*v1.Pod{p3_5, p3_6},
				},
				{
					Name:           "Create fourth leaf PodGroup",
					CreatePodGroup: pg3_4,
				},
				{
					Name:       "Create pods for fourth leaf group, completing 3 groups under root CPG",
					CreatePods: []*v1.Pod{p3_7, p3_8},
				},
				{
					Name:                 "Verify remaining groups are scheduled after quorum of 3 is satisfied",
					WaitForPodsScheduled: []string{"p3-3", "p3-4", "p3-5", "p3-6", "p3-7", "p3-8"},
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
