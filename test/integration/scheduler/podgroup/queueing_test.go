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
					Name:                            "Verify pods are unschedulable due to incomplete CPG hierarchy",
					WaitForPodsInIncompleteEntities: []string{"p1", "p2"},
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
					Name:                            "Verify second group pods are in incomplete entities due to missing PodGroup",
					WaitForPodsInIncompleteEntities: []string{"p2-3", "p2-4"},
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
