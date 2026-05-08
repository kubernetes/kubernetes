package runtime

import (
	"context"
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	cache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/mockdra"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type FakeTopologyPlugin struct{}

func (f *FakeTopologyPlugin) Name() string {
	return "FakeTopology"
}

func (f *FakeTopologyPlugin) GeneratePlacements(ctx context.Context, state fwk.PodGroupCycleState, podGroup fwk.PodGroupInfo, parentPlacement *fwk.Placement) (*fwk.GeneratePlacementsResult, *fwk.Status) {
	// Fake topology plugin groups nodes by their "topology.kubernetes.io/zone" label.
	zoneMap := make(map[string][]fwk.NodeInfo)
	for _, node := range parentPlacement.Nodes {
		zone := node.Node().Labels["topology.kubernetes.io/zone"]
		if zone != "" {
			zoneMap[zone] = append(zoneMap[zone], node)
		}
	}

	var placements []*fwk.Placement
	for zone, nodes := range zoneMap {
		placements = append(placements, &fwk.Placement{
			Name:  zone,
			Nodes: nodes,
		})
	}

	// Sort placements by name for deterministic test results
	sort.Slice(placements, func(i, j int) bool {
		return placements[i].Name < placements[j].Name
	})

	return &fwk.GeneratePlacementsResult{Placements: placements}, nil
}

func TestMockDRAAndTopologyMerging(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Setup nodes with both DRA labels and Topology (zone) labels to test a full cross-product intersection.
	nodeResources := []*v1.Node{
		st.MakeNode().Name("node1").Label("topology.kubernetes.io/zone", "zone-1").Label("dra-group", "group-A").Obj(),
		st.MakeNode().Name("node2").Label("topology.kubernetes.io/zone", "zone-1").Label("dra-group", "group-B").Obj(),
		st.MakeNode().Name("node3").Label("topology.kubernetes.io/zone", "zone-1").Label("dra-group", "group-C").Obj(),
		st.MakeNode().Name("node4").Label("topology.kubernetes.io/zone", "zone-2").Label("dra-group", "group-A").Obj(),
		st.MakeNode().Name("node5").Label("topology.kubernetes.io/zone", "zone-2").Label("dra-group", "group-B").Obj(),
		st.MakeNode().Name("node6").Label("topology.kubernetes.io/zone", "zone-2").Label("dra-group", "group-C").Obj(),
		st.MakeNode().Name("node7").Label("topology.kubernetes.io/zone", "zone-3").Label("dra-group", "group-A").Obj(),
		st.MakeNode().Name("node8").Label("topology.kubernetes.io/zone", "zone-3").Label("dra-group", "group-B").Obj(),
		st.MakeNode().Name("node9").Label("topology.kubernetes.io/zone", "zone-3").Label("dra-group", "group-C").Obj(),
		st.MakeNode().Name("node10").Label("topology.kubernetes.io/zone", "zone-1").Label("dra-group", "group-A").Obj(), // Extra node for dra-a + zone-1
	}
	nodesInCluster := make([]fwk.NodeInfo, len(nodeResources))
	for i, node := range nodeResources {
		nodesInCluster[i] = framework.NewNodeInfo()
		nodesInCluster[i].SetNode(node)
	}

	r := make(Registry)
	r.Register(mockdra.Name, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return mockdra.New(ctx, obj, fh, feature.Features{})
	})
	r.Register("FakeTopology", func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return &FakeTopologyPlugin{}, nil
	})

	profile := config.KubeSchedulerProfile{
		Plugins: &config.Plugins{
			PlacementGenerate: config.PluginSet{
				Enabled: []config.Plugin{
					{Name: mockdra.Name},
					{Name: "FakeTopology"},
				},
			},
		},
	}

	fw, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
	if err != nil {
		t.Fatalf("Unexpected error during calling NewFramework, got %v", err)
	}

	state := framework.NewCycleState()
	result, status := fw.RunPlacementGeneratePlugins(ctx, state, nil, nodesInCluster)
	if !status.IsSuccess() {
		t.Fatalf("Expected success, got %v", status)
	}

	// We expect 9 resulting placements because the cross-product of 3 DRA groups and 3 Zones
	// creates 9 valid intersections.
	if len(result) != 9 {
		for _, r := range result {
			t.Logf("Found placement: %s", r.Name)
		}
		t.Fatalf("Expected 9 merged placements, got %d", len(result))
	}

	// Sort the resulting merged placements for easier deterministic validation.
	sort.Slice(result, func(i, j int) bool {
		return result[i].Name < result[j].Name
	})

	expectedPlacements := []struct {
		Name          string
		ExpectedNodes []string
		ExpectedDRA   mockdra.AllocationData
	}{
		{
			Name:          "dra-a-zone-1", // node1 and node10 both have group-A and zone-1
			ExpectedNodes: []string{"node1", "node10"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-a-123", Model: "A100", MemoryGB: 80},
		},
		{
			Name:          "dra-a-zone-2", // node4
			ExpectedNodes: []string{"node4"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-a-123", Model: "A100", MemoryGB: 80},
		},
		{
			Name:          "dra-a-zone-3", // node7
			ExpectedNodes: []string{"node7"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-a-123", Model: "A100", MemoryGB: 80},
		},
		{
			Name:          "dra-b-zone-1", // node2
			ExpectedNodes: []string{"node2"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-b-456", Model: "H100", MemoryGB: 80},
		},
		{
			Name:          "dra-b-zone-2", // node5
			ExpectedNodes: []string{"node5"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-b-456", Model: "H100", MemoryGB: 80},
		},
		{
			Name:          "dra-b-zone-3", // node8
			ExpectedNodes: []string{"node8"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-b-456", Model: "H100", MemoryGB: 80},
		},
		{
			Name:          "dra-c-zone-1", // node3
			ExpectedNodes: []string{"node3"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-c-789", Model: "B200", MemoryGB: 192},
		},
		{
			Name:          "dra-c-zone-2", // node6
			ExpectedNodes: []string{"node6"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-c-789", Model: "B200", MemoryGB: 192},
		},
		{
			Name:          "dra-c-zone-3", // node9
			ExpectedNodes: []string{"node9"},
			ExpectedDRA:   mockdra.AllocationData{DeviceID: "gpu-c-789", Model: "B200", MemoryGB: 192},
		},
	}

	for i, expected := range expectedPlacements {
		p := result[i]
		if p.Name != expected.Name {
			t.Errorf("Expected placement name %s, got %s", expected.Name, p.Name)
		}

		var gotNodeNames []string
		for _, n := range p.Nodes {
			gotNodeNames = append(gotNodeNames, n.Node().Name)
		}
		if diff := cmp.Diff(expected.ExpectedNodes, gotNodeNames); diff != "" {
			t.Errorf("Unexpected nodes in placement %s (-want,+got):\n%s", p.Name, diff)
		}

		pState := state.GetPlacementCycleStateForName(p.Name)
		if pState == nil {
			t.Errorf("Expected PlacementCycleState for %s to be non-nil", p.Name)
			continue
		}

		val, err := pState.Read(mockdra.StateKey)
		if err != nil {
			t.Errorf("Expected PlacementCycleState to have mockdra state for %s, got error: %v", p.Name, err)
			continue
		}

		allocData, ok := val.(*mockdra.AllocationData)
		if !ok {
			t.Errorf("Expected AllocationData for %s, got %T", p.Name, val)
			continue
		}

		if diff := cmp.Diff(expected.ExpectedDRA, *allocData); diff != "" {
			t.Errorf("Unexpected AllocationData for %s (-want,+got):\n%s", p.Name, diff)
		}
	}
}

func TestMockDRAAndTopologyMerging_EmptyPlacement(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)

	// Setup nodes with Topology (zone) labels, but NO DRA labels.
	// This will cause MockDRA to generate 0 placements.
	nodeResources := []*v1.Node{
		st.MakeNode().Name("node1").Label("topology.kubernetes.io/zone", "zone-1").Obj(),
		st.MakeNode().Name("node2").Label("topology.kubernetes.io/zone", "zone-2").Obj(),
	}
	nodesInCluster := make([]fwk.NodeInfo, len(nodeResources))
	for i, node := range nodeResources {
		nodesInCluster[i] = framework.NewNodeInfo()
		nodesInCluster[i].SetNode(node)
	}

	r := make(Registry)
	r.Register(mockdra.Name, func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return mockdra.New(ctx, obj, fh, feature.Features{})
	})
	r.Register("FakeTopology", func(ctx context.Context, obj runtime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return &FakeTopologyPlugin{}, nil
	})

	profile := config.KubeSchedulerProfile{
		Plugins: &config.Plugins{
			PlacementGenerate: config.PluginSet{
				Enabled: []config.Plugin{
					{Name: mockdra.Name},
					{Name: "FakeTopology"},
				},
			},
		},
	}

	fw, err := newFrameworkWithQueueSortAndBind(ctx, r, profile, WithSnapshotSharedLister(cache.NewEmptySnapshot()))
	if err != nil {
		t.Fatalf("Unexpected error during calling NewFramework, got %v", err)
	}

	result, status := fw.RunPlacementGeneratePlugins(ctx, framework.NewCycleState(), nil, nodesInCluster)
	
	// We expect the framework to return an Unschedulable status because MockDRA returned 0 placements.
	if status.IsSuccess() {
		t.Fatalf("Expected Unschedulable status, got Success")
	}

	if status.Code() != fwk.Unschedulable {
		t.Errorf("Expected Unschedulable code, got %v", status.Code())
	}

	if result != nil {
		t.Errorf("Expected result to be nil, got %v placements", len(result))
	}
}
