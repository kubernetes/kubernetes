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

package noderesources

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
			features: feature.Features{EnableTopologyAwareWorkloadScheduling: true},
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
			features: feature.Features{EnableTopologyAwareWorkloadScheduling: false},
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
			features:    feature.Features{EnableTopologyAwareWorkloadScheduling: true},
			args:        &config.PlacementBinPackingArgs{},
			expectedErr: true,
		},
		{
			name:     "Fails to validate if scoring strategy is unknown",
			features: feature.Features{EnableTopologyAwareWorkloadScheduling: true},
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
			_, err := NewPlacementBinPacking(tCtx, tc.args, nil, tc.features)
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
		strategy            *config.ScoringStrategy
		expectedScore       int64
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
			// CPU: (1000m + 2000m) / (5000m + 5000m) = 0.3
			// Memory: (0 + 2000) / (5000 + 5000) = 0.2
			// Score: MaxNodeScore * (0.3 + 0.2) / 2 = 25
			expectedScore: 25,
		},
		{
			name: "Computes score using weights",
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 4},
					{Name: "memory", Weight: 1},
				},
			},
			// CPU: (1000m + 2000m) / (5000m + 5000m) = 0.3
			// Memory: (0 + 2000) / (5000 + 5000) = 0.2
			// Score: MaxNodeScore * (4 * 0.3 + 0.2) / 5 = 28
			expectedScore: 28,
		},
		{
			name: "Handles multiple pods per node",
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
				"bar": "node1",
			},
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 4},
					{Name: "memory", Weight: 1},
				},
			},
			// CPU: (1000m + 2000m) / (5000m + 5000m) = 0.3
			// Memory: (0 + 2000) / (5000 + 5000) = 0.2
			// Score: MaxNodeScore * (4 * 0.3 + 0.2) / 5 = 28
			expectedScore: 28,
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
			expectedScore: 25,
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
			// CPU: (1000m + 2000m + 1000m) / (5000m + 5000m) = 0.4
			// Memory: (0 + 2000 + 0) / (5000 + 5000) = 0.2
			// Score: MaxNodeScore * (0.3 + 0.2) / 2 = 30
			expectedScore: 30,
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
					{Name: "nvidia.com/gpu", Weight: 1},
				},
			},
			// CPU: (1000m + 1000m + 1000m) / (5000m + 5000m) = 0.3
			// Memory: (0 + 2000 + 0) / (5000 + 5000) = 0.2
			// GPU: (0 + 1 + 0) / (5 + 5) = 0.1
			// Score: MaxNodeScore * (0.3 + 0.2 + 0.1) / 3 = 20
			expectedScore: 20,
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
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
					{Name: "nvidia.com/gpu", Weight: 1},
				},
			},
			// CPU: (1000m + 1000m + 1000m) / (5000m + 5000m) = 0.3
			// Memory: (0 + 2000 + 0) / (5000 + 5000) = 0.2
			// GPU: not included as it isn't requested by the pods
			// Score: MaxNodeScore * (0.3 + 0.2) / 2 = 25
			expectedScore: 25,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			snapshot := cache.NewSnapshot(tc.preExistingPods, tc.nodes)
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			plugin, err := NewPlacementBinPacking(tCtx, &config.PlacementBinPackingArgs{ScoringStrategy: tc.strategy},
				fh, feature.Features{EnableTopologyAwareWorkloadScheduling: true})

			if err != nil {
				t.Fatal(err)
			}

			placementNodes := make([]fwk.NodeInfo, 0, len(tc.placementNodeNames))
			for _, name := range tc.placementNodeNames {
				nodeInfo, err := snapshot.NodeInfos().Get(name)
				if err != nil {
					t.Fatal(err)
				}
				placementNodes = append(placementNodes, nodeInfo)
			}
			proposedAssignments := make(map[*v1.Pod]string)
			for _, pod := range tc.podGroup {
				if nodeName, ok := tc.podGroupAssignments[pod.UID]; ok {
					proposedAssignments[pod] = nodeName
				}
			}
			podGroupInfo := &framework.PodGroupInfo{
				UnscheduledPods: tc.podGroup,
			}
			podGroupAssignments := &fwk.PodGroupAssignments{
				Placement: &fwk.Placement{
					Nodes: placementNodes,
				},
				ProposedAssignments: proposedAssignments,
			}

			score, status := plugin.ScorePlacement(tCtx, framework.NewCycleState(), podGroupInfo, podGroupAssignments)

			if !status.IsSuccess() {
				t.Fatalf("ScorePlacement failed: %v", status.AsError())
			}
			if score != tc.expectedScore {
				t.Fatalf("Unexpected score, want %v got %v", tc.expectedScore, score)
			}
		})
	}
}

func TestPlacementBinPackingScore_Strategies(t *testing.T) {
	nodes := []*v1.Node{
		st.MakeNode().Name("node1").Capacity(cpuAndMemory("100m", "100")).Obj(),
		st.MakeNode().Name("node2").Capacity(cpuAndMemory("200m", "200")).Obj(),
		st.MakeNode().Name("node3").Capacity(cpuAndMemory("300m", "300")).Obj(),
	}
	podGroup := []*v1.Pod{
		st.MakePod().UID("foo").Req(cpuAndMemory("100m", "0")).Obj(),
		st.MakePod().UID("bar").Req(cpuAndMemory("200m", "60")).Obj(),
	}
	podGroupAssignments := map[*v1.Pod]string{
		podGroup[0]: "node1",
		podGroup[1]: "node2",
	}

	// Allocations:
	// CPU: 300m/600m (50% allocated)
	// Memory: 60/600 (10% allocated)

	testCases := []struct {
		name          string
		strategy      *config.ScoringStrategy
		expectedScore int64
	}{
		{
			name: "LeastAllocated",
			strategy: &config.ScoringStrategy{
				Type: config.LeastAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
			// Score: MaxNodeScore * ((1 - 0.5) + (1 - 0.1)) / 2 = 70
			expectedScore: 70,
		},
		{
			name: "MostAllocated",
			strategy: &config.ScoringStrategy{
				Type: config.MostAllocated,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
			},
			// Score: MaxNodeScore * (0.5 + 0.1) / 2 = 30
			expectedScore: 30,
		},
		{
			name: "RequestedToCapacityRatio with shape equivalent to LeastAllocated",
			strategy: &config.ScoringStrategy{
				Type: config.RequestedToCapacityRatio,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
				RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
					Shape: []config.UtilizationShapePoint{
						{Utilization: 0, Score: 10},
						{Utilization: 100, Score: 0},
					},
				},
			},
			expectedScore: 70,
		},
		{
			name: "RequestedToCapacityRatio with shape equivalent to MostAllocated",
			strategy: &config.ScoringStrategy{
				Type: config.RequestedToCapacityRatio,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
				RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
					Shape: []config.UtilizationShapePoint{
						{Utilization: 0, Score: 0},
						{Utilization: 100, Score: 10},
					},
				},
			},
			expectedScore: 30,
		},
		{
			name: "RequestedToCapacityRatio with custom shape",
			strategy: &config.ScoringStrategy{
				Type: config.RequestedToCapacityRatio,
				Resources: []config.ResourceSpec{
					{Name: "cpu", Weight: 1},
					{Name: "memory", Weight: 1},
				},
				RequestedToCapacityRatio: &config.RequestedToCapacityRatioParam{
					Shape: []config.UtilizationShapePoint{
						{Utilization: 0, Score: 0},
						{Utilization: 50, Score: 10},
						{Utilization: 100, Score: 0},
					},
				},
			},
			// Interpolation between the two bounding points (x1, y1) and (x2, y2):
			// y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
			// Results are scaled by MaxNodeScore/MaxShapeScore (100/10=10)
			// CPU: interpolate between (0,0) and (50,10) for x=50, scaled result: 100
			// Memory: interpolate between (0,0) and (50,10) for x=10, scaled result: 20
			// Score: (100 + 20) / 2 = 60
			expectedScore: 60,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			snapshot := cache.NewSnapshot(nil, nodes)
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			plugin, err := NewPlacementBinPacking(tCtx, &config.PlacementBinPackingArgs{ScoringStrategy: tc.strategy},
				fh, feature.Features{EnableTopologyAwareWorkloadScheduling: true})

			if err != nil {
				t.Fatal(err)
			}

			placementNodes := make([]fwk.NodeInfo, 0, len(nodes))
			for _, node := range nodes {
				nodeInfo, err := snapshot.NodeInfos().Get(node.Name)
				if err != nil {
					t.Fatal(err)
				}
				placementNodes = append(placementNodes, nodeInfo)
			}
			podGroupInfo := &framework.PodGroupInfo{
				UnscheduledPods: podGroup,
			}
			podGroupAssignments := &fwk.PodGroupAssignments{
				Placement: &fwk.Placement{
					Nodes: placementNodes,
				},
				ProposedAssignments: podGroupAssignments,
			}

			score, status := plugin.ScorePlacement(tCtx, framework.NewCycleState(), podGroupInfo, podGroupAssignments)

			if !status.IsSuccess() {
				t.Fatalf("ScorePlacement failed: %v", status.AsError())
			}
			if score != tc.expectedScore {
				t.Fatalf("Unexpected score, want %v got %v", tc.expectedScore, score)
			}
		})
	}
}
