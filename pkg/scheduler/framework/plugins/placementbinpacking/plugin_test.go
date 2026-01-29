package placementbinpacking

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestPlacementBinPackingArgsValidation(t *testing.T) {
	testCases := []struct {
		name        string
		features    feature.Features
		args        *config.PlacementBinPackingArgs
		expectedErr bool
	}{
		{
			name:     "Succeeds if feature is enabled and strategy is set",
			features: feature.Features{EnableWorkloadSchedulingCycle: true},
			args: &config.PlacementBinPackingArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: config.LeastAllocated,
					Resources: []config.ResourceSpec{
						{Name: "cpu", Weight: 1},
						{Name: "memory", Weight: 1},
					},
				},
			},
			expectedErr: false,
		},
		{
			name:     "Fails to validate if feature is not enabled",
			features: feature.Features{EnableWorkloadSchedulingCycle: false},
			args: &config.PlacementBinPackingArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: config.LeastAllocated,
					Resources: []config.ResourceSpec{
						{Name: "cpu", Weight: 1},
						{Name: "memory", Weight: 1},
					},
				},
			},
			expectedErr: true,
		},
		{
			name:        "Fails to validate if scoring strategy is unset",
			features:    feature.Features{EnableWorkloadSchedulingCycle: true},
			args:        &config.PlacementBinPackingArgs{},
			expectedErr: true,
		},
		{
			name:     "Fails to validate if scoring strategy is unknown",
			features: feature.Features{EnableWorkloadSchedulingCycle: true},
			args: &config.PlacementBinPackingArgs{
				ScoringStrategy: &config.ScoringStrategy{
					Type: "foo",
					Resources: []config.ResourceSpec{
						{Name: "cpu", Weight: 1},
						{Name: "memory", Weight: 1},
					},
				},
			},
			expectedErr: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			_, err := New(tCtx, tc.args, nil, tc.features)
			if (err != nil) != tc.expectedErr {
				t.Fatalf("Unexpected error, want error %v, got %v", tc.expectedErr, err)
			}
		})
	}

}

func TestPlacementBinPackingScore(t *testing.T) {
	testCases := []struct {
		name                string
		nodes               []*v1.Node
		placementNodeNames  []string
		podGroup            []*v1.Pod
		podGroupAssignments map[types.UID]string
		preExistingPods     []*v1.Pod
		expectedScore       int64
		strategy            *config.ScoringStrategy
	}{
		{
			name: "Computes score based on total requests",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
				st.MakeNode().Name("node2").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
			},
			placementNodeNames: []string{"node1", "node2"},
			podGroup: []*v1.Pod{
				st.MakePod().UID("foo").Req(cpuAndMemory("1000m", "0")).Obj(),
				st.MakePod().UID("bar").Req(cpuAndMemory("2000m", "2000")).Obj(),
			},
			podGroupAssignments: map[types.UID]string{
				"foo": "node1",
				"bar": "node2",
			},
			expectedScore: 25,
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "Does not include nodes from outside of placement",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
				st.MakeNode().Name("node2").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
				st.MakeNode().Name("node3").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
			},
			placementNodeNames: []string{"node1", "node2"},
			podGroup: []*v1.Pod{
				st.MakePod().UID("foo").Req(cpuAndMemory("1000m", "0")).Obj(),
				st.MakePod().UID("bar").Req(cpuAndMemory("2000m", "2000")).Obj(),
			},
			podGroupAssignments: map[types.UID]string{
				"foo": "node1",
				"bar": "node2",
			},
			expectedScore: 25,
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "Includes pre-existing pods from outside of pod group",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
				st.MakeNode().Name("node2").Capacity(cpuAndMemory("5000m", "5000")).Obj(),
			},
			placementNodeNames: []string{"node1", "node2"},
			podGroup: []*v1.Pod{
				st.MakePod().UID("foo").Req(cpuAndMemory("1000m", "0")).Obj(),
				st.MakePod().UID("bar").Req(cpuAndMemory("2000m", "2000")).Obj(),
			},
			podGroupAssignments: map[types.UID]string{
				"foo": "node1",
				"bar": "node2",
			},
			preExistingPods: []*v1.Pod{
				st.MakePod().Node("node1").UID("baz").Req(cpuAndMemory("1000m", "0")).Obj(),
			},
			expectedScore: 30,
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
		},
		{
			name: "Includes scalar resources if any pod in pod group has nonzero requests",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(cpuAndMemoryAndGpu("5000m", "5000", "5")).Obj(),
				st.MakeNode().Name("node2").Capacity(cpuAndMemoryAndGpu("5000m", "5000", "5")).Obj(),
			},
			placementNodeNames: []string{"node1", "node2"},
			podGroup: []*v1.Pod{
				st.MakePod().UID("foo").Req(cpuAndMemoryAndGpu("1000m", "0", "0")).Obj(),
				st.MakePod().UID("bar").Req(cpuAndMemoryAndGpu("1000m", "2000", "1")).Obj(),
			},
			podGroupAssignments: map[types.UID]string{
				"foo": "node1",
				"bar": "node2",
			},
			preExistingPods: []*v1.Pod{
				st.MakePod().Node("node1").UID("baz").Req(cpuAndMemoryAndGpu("1000m", "0", "0")).Obj(),
			},
			expectedScore: 20,
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
					{Name: "nvidia.com/gpu", Weight: 1},
				},
			},
		},
		{
			name: "Does not include scalar resources if all pods in pod group have zero requests",
			nodes: []*v1.Node{
				st.MakeNode().Name("node1").Capacity(cpuAndMemoryAndGpu("5000m", "5000", "5")).Obj(),
				st.MakeNode().Name("node2").Capacity(cpuAndMemoryAndGpu("5000m", "5000", "5")).Obj(),
			},
			placementNodeNames: []string{"node1", "node2"},
			podGroup: []*v1.Pod{
				st.MakePod().UID("foo").Req(cpuAndMemoryAndGpu("1000m", "0", "0")).Obj(),
				st.MakePod().UID("bar").Req(cpuAndMemoryAndGpu("1000m", "2000", "0")).Obj(),
			},
			podGroupAssignments: map[types.UID]string{
				"foo": "node1",
				"bar": "node2",
			},
			preExistingPods: []*v1.Pod{
				st.MakePod().Node("node1").UID("baz").Req(cpuAndMemoryAndGpu("1000m", "0", "1")).Obj(),
			},
			expectedScore: 25,
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
					{Name: "nvidia.com/gpu", Weight: 1},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			snapshot := cache.NewSnapshot(tc.preExistingPods, tc.nodes)
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			plugin, err := New(tCtx, &config.PlacementBinPackingArgs{ScoringStrategy: tc.strategy},
				fh, feature.Features{EnableWorkloadSchedulingCycle: true})

			if err != nil {
				t.Fatal(err)
			}

			placementNodes := make([]*v1.Node, 0, len(tc.placementNodeNames))
			for _, name := range tc.placementNodeNames {
				nodeInfo, err := snapshot.NodeInfos().Get(name)
				if err != nil {
					t.Fatal(err)
				}
				placementNodes = append(placementNodes, nodeInfo.Node())
			}

			score, status := plugin.ScorePlacement(tCtx, framework.NewCycleState(), &fwk.PodGroupInfo{UnscheduledPods: tc.podGroup}, &fwk.ParentPlacement{
				PlacementNodes: placementNodes,
			}, &fwk.PodGroupAssignments{
				UnscheduledPodsToNodes: tc.podGroupAssignments,
			})

			if !status.IsSuccess() {
				t.Fatal(status.AsError())
			}
			if score != tc.expectedScore {
				t.Fatalf("Unexpected score, want %v got %v", tc.expectedScore, score)
			}
		})
	}
}

func cpuAndMemory(cpuReq, memoryReq string) map[v1.ResourceName]string {
	return map[v1.ResourceName]string{v1.ResourceCPU: cpuReq, v1.ResourceMemory: memoryReq}
}
func cpuAndMemoryAndGpu(cpuReq, memoryReq, gpuReq string) map[v1.ResourceName]string {
	return map[v1.ResourceName]string{v1.ResourceCPU: cpuReq, v1.ResourceMemory: memoryReq, "nvidia.com/gpu": gpuReq}
}
