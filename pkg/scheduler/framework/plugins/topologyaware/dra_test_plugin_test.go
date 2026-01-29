package topologyaware

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apimachineryruntime "k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestGeneratePlacements(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Label("hostname", "node1").Label("topology-block", "b0").Obj()
	node2 := st.MakeNode().Name("node2").Label("hostname", "node2").Label("topology-block", "b0").Obj()
	node3 := st.MakeNode().Name("node3").Label("hostname", "node3").Label("topology-block", "b1").Obj()
	node4 := st.MakeNode().Name("node4").Label("hostname", "node4").Label("topology-block", "b1").Obj()
	class1 := st.MakeDeviceClass("block.topo.example.com").
		Selector(`device.driver == "topo.example.com" && device.attributes["topo.example.com"].topoLevel == "block"`).
		Obj()
	classNode := st.MakeDeviceClass("node.topo.example.com").
		Selector(`device.driver == "topo.example.com" && device.attributes["topo.example.com"].topoLevel == "node"`).
		Obj()

	// Slice on Node1
	sliceBlock1 := st.MakeResourceSlice("b0", "topo.example.com").
		NodeSelector(map[string]string{"topology-block": "b0"}).
		ResourceSliceCount(2).
		Pool("pool1").
		Device("block-0", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"uid":       {StringValue: ptr.To("block-0-uid")},
			"topoLevel": {StringValue: ptr.To("block")},
		}).
		Obj()

	// Slice on Node2
	sliceBlock2 := st.MakeResourceSlice("b1", "topo.example.com").
		NodeSelector(map[string]string{"topology-block": "b1"}).
		ResourceSliceCount(2).
		Pool("pool1").
		Device("block-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"uid":       {StringValue: ptr.To("block-1-uid")},
			"topoLevel": {StringValue: ptr.To("block")},
		}).
		Obj()

	// Slice specific to Node1 for combination test
	sliceNode1 := st.MakeResourceSlice("node1", "topo.example.com").
		NodeSelector(map[string]string{"hostname": "node1"}).
		Pool("pool2").
		ResourceSliceCount(1).
		Device("node-dev-1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"uid":       {StringValue: ptr.To("node-dev-1-uid")},
			"topoLevel": {StringValue: ptr.To("node")},
		}).
		Obj()

	// Claim to be on the same block
	claimBlock := st.MakeResourceClaim().
		Name("claim-block").
		Namespace("default").
		UID("claim-block-uid").
		RequestWithNameCount("req1", "block.topo.example.com", 1).
		Obj()

	// Claim seeking device on block 1 (valid on node1, node2) - distinct from claimBlock to allow composition
	claimBlock1 := st.MakeResourceClaim().
		Name("claim-block1").
		Namespace("default").
		UID("claim-block1-uid").
		RequestExact("req1", "block.topo.example.com", 1, `device.attributes["topo.example.com"]["uid"] == "block-0-uid"`).
		Obj()

	// Claim seeking device on node1 only
	claimNode1 := st.MakeResourceClaim().
		Name("claim-node1").
		Namespace("default").
		UID("claim-node1-uid").
		RequestExact("req1", "node.topo.example.com", 1, `device.attributes["topo.example.com"]["uid"] == "node-dev-1-uid"`).
		Obj()

	claimUnsatisfiable := st.MakeResourceClaim().
		Name("claim-unsatable").
		Namespace("default").
		UID("claim-unsatable-uid").
		RequestWithNameCount("req1", "block.topo.example.com", 1000).
		Obj()

	nodeList := []*v1.Node{node1, node2, node3, node4}
	classes := []*resourceapi.DeviceClass{class1, classNode}
	slices := []*resourceapi.ResourceSlice{sliceBlock1, sliceBlock2, sliceNode1}
	claims := []*resourceapi.ResourceClaim{claimBlock, claimBlock1, claimNode1, claimUnsatisfiable}
	allNodesParent := &fwk.ParentPlacement{
		Placement: fwk.Placement{
			NodeSelector: &v1.NodeSelector{},
		},
		PlacementNodes: nodeList,
	}

	tests := []struct {
		name               string
		podGroup           *fwk.PodGroupInfo
		parentPlacements   []*fwk.ParentPlacement
		expectedPlacements int
		verify             func(t *testing.T, placements []*fwk.Placement)
	}{
		{
			name: "1 pod in a pod group, requesting claim, expected 2 placements (b0 and b1)",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-block", ResourceClaimName: ptr.To("claim-block")}).Obj(),
				},
			},
			parentPlacements:   []*fwk.ParentPlacement{allNodesParent},
			expectedPlacements: 2,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				// We expect 2 placements:
				// 1. Matches node1/node2 (rack1), allocation rack-1
				// 2. Matches node3/node4 (rack2), allocation rack-2
				// Placements come in arbitrary order, so we accept either if valid.
				block0Found := false
				block1Found := false
				for _, p := range placements {
					// Check allocation
					if len(p.DRAAllocations) != 1 {
						t.Errorf("Placement has %d allocations, expected 1", len(p.DRAAllocations))
						continue
					}
					alloc := p.DRAAllocations[0]
					if alloc.ResourceClaimName != "claim-block" {
						t.Errorf("Unexpected claim name: %s", alloc.ResourceClaimName)
					}

					deviceName := alloc.Allocation.Devices.Results[0].Device
					switch deviceName {
					case "block-0":
						block0Found = true
						if !matches(p.NodeSelector, node1) || !matches(p.NodeSelector, node2) {
							t.Errorf("Placement for block-0 should match node1 and node2, got selector: %v", p.NodeSelector)
						}
						if matches(p.NodeSelector, node3) || matches(p.NodeSelector, node4) {
							t.Errorf("Placement for block-0 should NOT match node3 or node4")
						}
					case "block-1":
						block1Found = true
						if !matches(p.NodeSelector, node3) || !matches(p.NodeSelector, node4) {
							t.Errorf("Placement for block-1 should match node3 and node4, got selector: %v", p.NodeSelector)
						}
						if matches(p.NodeSelector, node1) || matches(p.NodeSelector, node2) {
							t.Errorf("Placement for block-1 should NOT match node1 or node2")
						}
					default:
						t.Errorf("Unexpected device: %s", deviceName)
					}
				}
				if !block0Found || !block1Found {
					t.Errorf("Did not find both block-0 and block-1 placements")
				}
			},
		},
		{
			name: "2 pods in a pod group, requesting the same claim, expected 2 placements (rack 1 and rack 2)",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-block", ResourceClaimName: ptr.To("claim-block")}).Obj(),
					st.MakePod().Name("pod2").Namespace("default").UID("pod2-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-block", ResourceClaimName: ptr.To("claim-block")}).Obj(),
				},
			},
			parentPlacements:   []*fwk.ParentPlacement{allNodesParent},
			expectedPlacements: 2,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				// Same verification as above because they share the same claim
				block0Found := false
				block1Found := false
				for _, p := range placements {
					if len(p.DRAAllocations) != 1 {
						t.Errorf("Placement has %d allocations, expected 1", len(p.DRAAllocations))
						continue
					}
					deviceName := p.DRAAllocations[0].Allocation.Devices.Results[0].Device
					switch deviceName {
					case "block-0":
						block0Found = true
						if !matches(p.NodeSelector, node1) || !matches(p.NodeSelector, node2) {
							t.Errorf("Placement for block-0 should match node1 and node2")
						}
					case "block-1":
						block1Found = true
						if !matches(p.NodeSelector, node3) || !matches(p.NodeSelector, node4) {
							t.Errorf("Placement for block-1 should match node3 and node4")
						}
					}
				}
				if !block0Found || !block1Found {
					t.Errorf("Did not find both block-0 and block-1 placements")
				}
			},
		},
		{
			name: "1 pod in a pod group, requesting no claim, expected 1 placement (node selector, all nodes)",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").Obj(),
				},
			},
			parentPlacements:   []*fwk.ParentPlacement{allNodesParent},
			expectedPlacements: 1,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				p := placements[0]
				if len(p.DRAAllocations) != 0 {
					t.Errorf("Expected 0 allocations, got %d", len(p.DRAAllocations))
				}
				// Selector should be nil or empty or match everything
				if p.NodeSelector != nil && len(p.NodeSelector.NodeSelectorTerms) > 0 {
					// We verify it matches all nodes
					if !matches(p.NodeSelector, node1) || !matches(p.NodeSelector, node2) || !matches(p.NodeSelector, node3) || !matches(p.NodeSelector, node4) {
						t.Errorf("Placement should match all nodes, got selector: %v", p.NodeSelector)
					}
				}
			},
		},
		{
			name: "1 pod in a pod group requesting claim that cannot be allocated, expected 0 placements",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-unsatable", ResourceClaimName: ptr.To("claim-unsatable")}).Obj(),
				},
			},
			expectedPlacements: 0,
			verify:             func(t *testing.T, placements []*fwk.Placement) {},
		},
		{
			name: "1 pod requesting two claims with intersecting topologies (Rack+Node), expected 1 placement on Node1",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod-comb").Namespace("default").UID("pod-comb-uid").
						PodResourceClaims(
							v1.PodResourceClaim{Name: "claim-block1", ResourceClaimName: ptr.To("claim-block1")},
							v1.PodResourceClaim{Name: "claim-node1", ResourceClaimName: ptr.To("claim-node1")},
						).Obj(),
				},
			},
			parentPlacements:   []*fwk.ParentPlacement{allNodesParent},
			expectedPlacements: 1,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				if len(placements) == 0 {
					return
				}
				p := placements[0]
				// Must match node1
				if !matches(p.NodeSelector, node1) {
					t.Errorf("Placement should match node1")
				}
				// Must NOT match node2 (which is in the same rack, but doesn't satisfy claim-comb-node)
				if matches(p.NodeSelector, node2) {
					t.Errorf("Placement should NOT match node2")
				}
				if len(p.DRAAllocations) != 2 {
					t.Errorf("Expected 2 DRAAllocations, got %d", len(p.DRAAllocations))
				}
			},
		},
		{
			name: "no claims, 1 parent placement, expected 1 placement (parent returned)",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").Obj(),
				},
			},
			parentPlacements: []*fwk.ParentPlacement{
				{
					Placement: fwk.Placement{
						NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "hostname", Operator: v1.NodeSelectorOpIn, Values: []string{"node1"}}}}}},
					},
					PlacementNodes: []*v1.Node{node1},
				},
			},
			expectedPlacements: 1,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				p := placements[0]
				if len(p.DRAAllocations) != 0 {
					t.Errorf("Expected 0 allocations, got %d", len(p.DRAAllocations))
				}
				if !matches(p.NodeSelector, node1) {
					t.Errorf("Placement should match node1")
				}
				if matches(p.NodeSelector, node2) {
					t.Errorf("Placement should NOT match node2")
				}
			},
		},
		{
			name: "1 claim, 1 parent placement (node1), expected 1 placement (on node1)",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-block", ResourceClaimName: ptr.To("claim-block")}).Obj(),
				},
			},
			parentPlacements: []*fwk.ParentPlacement{
				{
					Placement: fwk.Placement{
						NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "hostname", Operator: v1.NodeSelectorOpIn, Values: []string{"node1"}}}}}},
					},
					PlacementNodes: []*v1.Node{node1},
				},
			},
			expectedPlacements: 1,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				p := placements[0]
				if len(p.DRAAllocations) != 1 {
					t.Errorf("Expected 1 allocation, got %d", len(p.DRAAllocations))
				}
				if !matches(p.NodeSelector, node1) {
					t.Errorf("Placement should match node1")
				}
				if matches(p.NodeSelector, node2) {
					t.Errorf("Placement should NOT match node2 (filtered by parent)")
				}
			},
		},
		{
			name: "1 claim, 2 parent placements (node1, node2), expected 2 placements",
			podGroup: &fwk.PodGroupInfo{
				UnscheduledPods: []*v1.Pod{
					st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
						PodResourceClaims(v1.PodResourceClaim{Name: "claim-block", ResourceClaimName: ptr.To("claim-block")}).Obj(),
				},
			},
			parentPlacements: []*fwk.ParentPlacement{
				{
					Placement: fwk.Placement{
						NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "hostname", Operator: v1.NodeSelectorOpIn, Values: []string{"node1"}}}}}},
					},
					PlacementNodes: []*v1.Node{node1},
				},
				{
					Placement: fwk.Placement{
						NodeSelector: &v1.NodeSelector{NodeSelectorTerms: []v1.NodeSelectorTerm{{MatchExpressions: []v1.NodeSelectorRequirement{{Key: "hostname", Operator: v1.NodeSelectorOpIn, Values: []string{"node2"}}}}}},
					},
					PlacementNodes: []*v1.Node{node2},
				},
			},
			expectedPlacements: 2,
			verify: func(t *testing.T, placements []*fwk.Placement) {
				// We expect one placement matching node1 and one matching node2
				node1Found := false
				node2Found := false
				for _, p := range placements {
					if matches(p.NodeSelector, node1) && !matches(p.NodeSelector, node2) {
						node1Found = true
					}
					if matches(p.NodeSelector, node2) && !matches(p.NodeSelector, node1) {
						node2Found = true
					}
				}
				if !node1Found || !node2Found {
					t.Errorf("Expected placements for both node1 and node2")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			draPlugin, tCtx := setupDRAPlugin(t, nodeList, classes, slices, claims)
			ctx := tCtx

			var state fwk.CycleState = framework.NewCycleState()
			placements, status := draPlugin.GeneratePlacements(ctx, &state, tt.podGroup, tt.parentPlacements)
			if !status.IsSuccess() {
				t.Fatalf("GeneratePlacements failed: %v", status)
			}

			if len(placements) != tt.expectedPlacements {
				t.Errorf("Expected %d placements, got %d", tt.expectedPlacements, len(placements))
				for i, p := range placements {
					t.Logf("Placement %d: NodeSelector=%v", i, p.NodeSelector)
				}
			}
			if tt.verify != nil {
				tt.verify(t, placements)
			}
		})
	}
}

func setupDRAPlugin(t *testing.T, nodes []*v1.Node, classes []*resourceapi.DeviceClass, slices []*resourceapi.ResourceSlice, claims []*resourceapi.ResourceClaim) (*DRATestPlugin, ktesting.TContext) {
	tCtx := ktesting.Init(t)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, true)
	ctx := context.Background()

	objs := []apimachineryruntime.Object{}
	for _, n := range nodes {
		objs = append(objs, n)
	}
	for _, c := range classes {
		objs = append(objs, c)
	}
	for _, s := range slices {
		objs = append(objs, s)
	}
	for _, c := range claims {
		objs = append(objs, c)
	}

	client := fake.NewSimpleClientset(objs...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	for _, n := range nodes {
		informerFactory.Core().V1().Nodes().Informer().GetIndexer().Add(n)
	}
	for _, c := range classes {
		informerFactory.Resource().V1().DeviceClasses().Informer().GetIndexer().Add(c)
	}
	for _, s := range slices {
		informerFactory.Resource().V1().ResourceSlices().Informer().GetIndexer().Add(s)
	}
	for _, c := range claims {
		informerFactory.Resource().V1().ResourceClaims().Informer().GetIndexer().Add(c)
	}

	sharedLister := &fakeSharedLister{nodes: nodes}

	claimsCache := assumecache.NewAssumeCache(tCtx.Logger(), informerFactory.Resource().V1().ResourceClaims().Informer(), "resource claim", "", nil)
	registeredHandler := claimsCache.AddEventHandler(cache.ResourceEventHandlerFuncs{})

	resourceSliceTrackerOpts := resourceslicetracker.Options{
		EnableDeviceTaintRules: true,
		SliceInformer:          informerFactory.Resource().V1().ResourceSlices(),
		TaintInformer:          informerFactory.Resource().V1alpha3().DeviceTaintRules(),
		ClassInformer:          informerFactory.Resource().V1().DeviceClasses(),
		KubeClient:             client,
	}
	resourceSliceTracker, err := resourceslicetracker.StartTracker(tCtx, resourceSliceTrackerOpts)
	require.NoError(tCtx, err, "couldn't start resource slice tracker")

	draManager := dynamicresources.NewDRAManager(ctx, claimsCache, resourceSliceTracker, informerFactory)

	fh, err := frameworkruntime.NewFramework(context.Background(), nil, nil,
		frameworkruntime.WithSnapshotSharedLister(sharedLister),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithSharedDRAManager(draManager))
	if err != nil {
		t.Fatalf("Failed to create framework: %v", err)
	}

	pl, err := New(context.Background(), nil, fh)
	if err != nil {
		t.Fatalf("Failed to create plugin: %v", err)
	}

	informerFactory.Start(tCtx.Done())
	tCtx.Cleanup(func() {
		// Need to cancel before waiting for the shutdown.
		tCtx.Cancel("test is done")
		// Now we can wait for all goroutines to stop.
		informerFactory.Shutdown()
	})

	informerFactory.WaitForCacheSync(tCtx.Done())
	cache.WaitForNamedCacheSyncWithContext(tCtx, registeredHandler.HasSynced, resourceSliceTracker.HasSynced)

	return pl.(*DRATestPlugin), tCtx
}

type fakeSharedLister struct {
	nodes []*v1.Node
}

func (f *fakeSharedLister) NodeInfos() fwk.NodeInfoLister {
	return tf.NodeInfoLister(tf.BuildNodeInfos(f.nodes))
}

func (f *fakeSharedLister) StorageInfos() fwk.StorageInfoLister {
	return nil
}

func matches(selector *v1.NodeSelector, node *v1.Node) bool {
	if selector == nil {
		return true
	}
	ns, err := nodeaffinity.NewNodeSelector(selector)
	if err != nil {
		return false
	}
	return ns.Match(node)
}

func TestAssumeRevertPlacement(t *testing.T) {
	node1 := st.MakeNode().Name("node1").Label("hostname", "node1").Obj()
	class1 := st.MakeDeviceClass("class1").
		Selector(`true`).
		Obj()
	slice1 := st.MakeResourceSlice("slice1", "driver1").
		NodeSelector(map[string]string{"hostname": "node1"}).
		Pool("pool1").
		ResourceSliceCount(1).
		Device("dev1", map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"uid": {StringValue: ptr.To("dev1")},
		}).
		Obj()

	claim1 := st.MakeResourceClaim().
		Name("claim1").
		Namespace("default").
		UID("claim1-uid").
		RequestWithNameCount("req1", "class1", 1).
		Obj()

	draPlugin, tCtx := setupDRAPlugin(t, []*v1.Node{node1}, []*resourceapi.DeviceClass{class1}, []*resourceapi.ResourceSlice{slice1}, []*resourceapi.ResourceClaim{claim1})
	ctx := tCtx

	pod := st.MakePod().Name("pod1").Namespace("default").UID("pod1-uid").
		PodResourceClaims(v1.PodResourceClaim{Name: "claim1", ResourceClaimName: ptr.To("claim1")}).Obj()
	podGroup := &fwk.PodGroupInfo{UnscheduledPods: []*v1.Pod{pod}}

	// Generate placement to get a valid allocation
	var state fwk.CycleState = framework.NewCycleState()
	parentPlacements := []*fwk.ParentPlacement{{Placement: fwk.Placement{NodeSelector: &v1.NodeSelector{}}, PlacementNodes: []*v1.Node{node1}}}
	placements, status := draPlugin.GeneratePlacements(ctx, &state, podGroup, parentPlacements)
	require.True(t, status.IsSuccess())
	require.Len(t, placements, 1)
	placement := placements[0]

	// 1. Check initially not allocated
	if draPlugin.draManager.ResourceClaims().ClaimHasPendingAllocation(claim1.UID) {
		t.Fatal("Claim should NOT have pending allocation initially")
	}

	// 2. AssumePlacement
	status = draPlugin.AssumePlacement(ctx, &state, podGroup, placement)
	require.True(t, status.IsSuccess())

	// 3. Check allocated
	if !draPlugin.draManager.ResourceClaims().ClaimHasPendingAllocation(claim1.UID) {
		t.Fatal("Claim SHOULD have pending allocation after AssumePlacement")
	}

	// 4. RevertPlacement
	status = draPlugin.RevertPlacement(ctx, &state, podGroup, placement)
	require.True(t, status.IsSuccess())

	// 5. Check not allocated
	if draPlugin.draManager.ResourceClaims().ClaimHasPendingAllocation(claim1.UID) {
		t.Fatal("Claim should NOT have pending allocation after RevertPlacement")
	}
}
