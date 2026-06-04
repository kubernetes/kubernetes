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
	"testing"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeschedulerconfigv1 "k8s.io/kube-scheduler/config/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	stepsframework "k8s.io/kubernetes/test/integration/scheduler/podgroup/stepsframework"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/ptr"
)

const nodePoolPlacementName = "NodePoolPlacement"

// nodePoolPlacement is a test PlacementGeneratePlugin that restricts a PodGroup to nodes whose
// "pool" label matches a fixed value. Running it alongside the in-tree TopologyPlacement plugin
// exercises the framework's merging of multiple PlacementGenerate plugins.
type nodePoolPlacement struct {
	pool string
}

func (p *nodePoolPlacement) Name() string { return nodePoolPlacementName }

func (p *nodePoolPlacement) GeneratePlacements(_ context.Context, _ fwk.PodGroupCycleState, _ fwk.PodGroupInfo, parent *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	var nodes []fwk.NodeInfo
	for _, n := range parent.Nodes {
		if n.Node().Labels["pool"] == p.pool {
			nodes = append(nodes, n)
		}
	}
	if len(nodes) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no nodes in pool "+p.pool)
	}
	return &fwk.GeneratePlacementsResult{Placements: []*fwk.Placement{{Name: "pool-" + p.pool, Nodes: nodes}}}, nil
}

func newNodePoolPlacement(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
	return &nodePoolPlacement{pool: "a"}, nil
}

func makePoolNode(nodeName, rackLabel, poolLabel string) *v1.Node {
	return st.MakeNode().Name(nodeName).Label("rack", rackLabel).Label("pool", poolLabel).
		Capacity(map[v1.ResourceName]string{v1.ResourceCPU: "2"}).Obj()
}

// TestTopologyAwareSchedulingWithMultiplePlacementPlugins verifies that when two PlacementGenerate
// plugins are configured, the scheduler merges their placements by node intersection: a gang must
// satisfy both the rack topology constraint (TopologyPlacement) and the pool constraint
// (nodePoolPlacement). rack-1 has 3 nodes but only 2 in pool "a", so it cannot fit a gang of 3
// large pods; rack-2 has 3 pool "a" nodes and is the only feasible placement.
func TestTopologyAwareSchedulingWithMultiplePlacementPlugins(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.GangScheduling:                  true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	cfg := configtesting.V1ToInternalWithDefaults(t, kubeschedulerconfigv1.KubeSchedulerConfiguration{
		Profiles: []kubeschedulerconfigv1.KubeSchedulerProfile{{
			SchedulerName: ptr.To(v1.DefaultSchedulerName),
			Plugins: &kubeschedulerconfigv1.Plugins{
				PlacementGenerate: kubeschedulerconfigv1.PluginSet{
					Enabled: []kubeschedulerconfigv1.Plugin{{Name: nodePoolPlacementName}},
				},
			},
		}},
	})
	registry := frameworkruntime.Registry{nodePoolPlacementName: newNodePoolPlacement}

	testCtx := testutils.InitTestSchedulerWithOptions(
		t, testutils.InitTestAPIServer(t, "tas-merge", nil), 0,
		scheduler.WithProfiles(cfg.Profiles...),
		scheduler.WithFrameworkOutOfTreeRegistry(registry),
		scheduler.WithPodMaxBackoffSeconds(0),
		scheduler.WithPodInitialBackoffSeconds(0),
	)
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.SchedulerCtx)

	ns := testCtx.NS.Name
	workload := st.MakeWorkload().Name("workload").Namespace(ns).
		PodGroupTemplate(st.MakePodGroupTemplate().Name("t1").MinCount(1).Obj()).Obj()

	steps := []stepsframework.Step{
		{
			Name:            "Creating workload",
			CreateWorkloads: []*schedulingapi.Workload{workload},
		},
		{
			Name: "Create racks. rack-1 has only 2 pool-a nodes, rack-2 has 3 pool-a nodes",
			CreateNodes: []*v1.Node{
				makePoolNode("node1-rack1", "rack-1", "a"),
				makePoolNode("node2-rack1", "rack-1", "a"),
				makePoolNode("node3-rack1", "rack-1", "b"),
				makePoolNode("node4-rack2", "rack-2", "a"),
				makePoolNode("node5-rack2", "rack-2", "a"),
				makePoolNode("node6-rack2", "rack-2", "a"),
			},
		},
		{
			Name:           "Create the gang PodGroup (minCount=3) keyed on rack",
			CreatePodGroup: makeGangPodGroup("pg1", "rack", 3),
		},
		{
			Name: "Create all pods belonging to the podgroup. Each pod consumes a whole node",
			CreatePods: []*v1.Pod{
				makeLargePod("p1", "pg1"),
				makeLargePod("p2", "pg1"),
				makeLargePod("p3", "pg1"),
			},
		},
		{
			Name:                 "Verify the entire gang is scheduled",
			WaitForPodsScheduled: []string{"p1", "p2", "p3"},
		},
		{
			Name: "Verify pods landed on rack-2, the only rack whose pool-a nodes fit the gang",
			VerifyAssignments: &stepsframework.VerifyAssignments{
				Pods:  []string{"p1", "p2", "p3"},
				Nodes: sets.New("node4-rack2", "node5-rack2", "node6-rack2"),
			},
		},
	}
	if err := stepsframework.RunSteps(testCtx, t, ns, steps); err != nil {
		t.Fatal(err)
	}
}
