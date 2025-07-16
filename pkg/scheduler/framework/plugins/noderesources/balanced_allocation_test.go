/*
Copyright 2019 The Kubernetes Authors.

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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func TestNodeResourcesBalancedAllocation(t *testing.T) {
	cpuAndMemoryAndGPU := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("3000"),
						"nvidia.com/gpu":  resource.MustParse("3"),
					},
				},
			},
		},
		NodeName: "node1",
	}
	cpuOnly := v1.PodSpec{
		NodeName: "node1",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("0"),
					},
				},
			},
		},
	}
	cpuOnly2 := cpuOnly
	cpuOnly2.NodeName = "node2"
	cpuAndMemory := v1.PodSpec{
		NodeName: "node2",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000m"),
						v1.ResourceMemory: resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("3000"),
					},
				},
			},
		},
	}

	defaultResourceBalancedAllocationSet := []config.ResourceSpec{
		{Name: string(v1.ResourceCPU), Weight: 1},
		{Name: string(v1.ResourceMemory), Weight: 1},
	}
	scalarResource := map[string]int64{
		"nvidia.com/gpu": 8,
	}

	tests := []struct {
		pod                    *v1.Pod
		pods                   []*v1.Pod
		nodes                  []*v1.Node
		expectedList           framework.NodeScoreList
		name                   string
		args                   config.NodeResourcesBalancedAllocationArgs
		runPreScore            bool
		wantPreScoreStatusCode framework.Code
	}{
		{
			// bestEffort pods, skip in PreScore
			pod:                    st.MakePod().Obj(),
			nodes:                  []*v1.Node{makeNode("node1", 4000, 10000, nil), makeNode("node2", 4000, 10000, nil)},
			name:                   "nothing scheduled, nothing requested, skip in PreScore",
			args:                   config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:            true,
			wantPreScoreStatusCode: framework.Skip,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 4000= 75%
			// Memory Fraction: 5000 / 10000 = 50%
			// Node1 std: (0.75 - 0.5) / 2 = 0.125
			// Node1 Score: (1 - 0.125)*MaxNodeScore = 87
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 6000= 50%
			// Memory Fraction: 5000/10000 = 50%
			// Node2 std: 0
			// Node2 Score: (1-0) * MaxNodeScore = MaxNodeScore
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 4000, 10000, nil), makeNode("node2", 6000, 10000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 87}, {Name: "node2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled, resources requested, differently sized nodes",
			args:         config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:  true,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node1 std: (0.6 - 0.25) / 2 = 0.175
			// Node1 Score: (1 - 0.175)*MaxNodeScore = 82
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 10000 / 20000 = 50%
			// Node2 std: (0.6 - 0.5) / 2 = 0.05
			// Node2 Score: (1 - 0.05)*MaxNodeScore = 95
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 20000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 95}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node1 std: (0.6 - 0.25) / 2 = 0.175
			// Node1 Score: (1 - 0.175)*MaxNodeScore = 82
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 10000 / 50000 = 20%
			// Node2 std: (0.6 - 0.2) / 2 = 0.2
			// Node2 Score: (1 - 0.2)*MaxNodeScore = 80
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 50000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 80}},
			name:         "resources requested, pods scheduled with resources, differently sized nodes",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 6000 = 1
			// Memory Fraction: 0 / 10000 = 0
			// Node1 std: (1 - 0) / 2 = 0.5
			// Node1 Score: (1 - 0.5)*MaxNodeScore = 50
			// Node1 Score: MaxNodeScore - (1 - 0) * MaxNodeScore = 0
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 6000 = 1
			// Memory Fraction 5000 / 10000 = 50%
			// Node2 std: (1 - 0.5) / 2 = 0.25
			// Node2 Score: (1 - 0.25)*MaxNodeScore = 75
			pod:          &v1.Pod{Spec: cpuOnly},
			nodes:        []*v1.Node{makeNode("node1", 6000, 10000, nil), makeNode("node2", 6000, 10000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 50}, {Name: "node2", Score: 75}},
			name:         "requested resources at node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
		},
		// Node1 scores on 0-MaxNodeScore scale
		// CPU Fraction: 3000 / 3500 = 85.71%
		// Memory Fraction: 5000 / 40000 = 12.5%
		// GPU Fraction: 4 / 8 = 0.5%
		// Node1 std: sqrt(((0.8571 - 0.503) *  (0.8571 - 0.503) + (0.503 - 0.125) * (0.503 - 0.125) + (0.503 - 0.5) * (0.503 - 0.5)) / 3) = 0.3002
		// Node1 Score: (1 - 0.3002)*MaxNodeScore = 70
		// Node2 scores on 0-MaxNodeScore scale
		// CPU Fraction: 3000 / 3500 = 85.71%
		// Memory Fraction: 5000 / 40000 = 12.5%
		// GPU Fraction: 1 / 8 = 12.5%
		// Node2 std: sqrt(((0.8571 - 0.378) *  (0.8571 - 0.378) + (0.378 - 0.125) * (0.378 - 0.125)) + (0.378 - 0.125) * (0.378 - 0.125)) / 3) = 0.345
		// Node2 Score: (1 - 0.358)*MaxNodeScore = 65
		{
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceMemory: "0",
				"nvidia.com/gpu":  "1",
			}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, scalarResource), makeNode("node2", 3500, 40000, scalarResource)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 70}, {Name: "node2", Score: 65}},
			name:         "include scalar resource on a node for balanced resource allocation",
			pods: []*v1.Pod{
				{Spec: cpuAndMemory},
				{Spec: cpuAndMemoryAndGPU},
			},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: "nvidia.com/gpu", Weight: 1},
			}},
			runPreScore: true,
		},
		// Only one node (node1) has the scalar resource, pod doesn't request the scalar resource and the scalar resource should be skipped for consideration.
		// Node1 scores on 0-MaxNodeScore scale
		// CPU Fraction: 3000 / 3500 = 85.71%
		// Memory Fraction: 5000 / 40000 = 12.5%
		// Node1 std: (0.8571 - 0.125) / 2 = 0.36605
		// Node1 Score: (1 - 0.22705)*MaxNodeScore = 63
		// Node2 scores on 0-MaxNodeScore scale
		// CPU Fraction: 3000 / 3500 = 85.71%
		// Memory Fraction: 5000 / 40000 = 12.5%
		// Node2 std: (0.8571 - 0.125) / 2 = 0.36605
		// Node2 Score: (1 - 0.22705)*MaxNodeScore = 63
		{
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, scalarResource), makeNode("node2", 3500, 40000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 63}, {Name: "node2", Score: 63}},
			name:         "node without the scalar resource should skip the scalar resource",
			pods:         []*v1.Pod{},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: "nvidia.com/gpu", Weight: 1},
			}},
			runPreScore: true,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 5000 / 20000 = 25%
			// Node1 std: (0.6 - 0.25) / 2 = 0.175
			// Node1 Score: (1 - 0.175)*MaxNodeScore = 82
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 6000 / 10000 = 60%
			// Memory Fraction: 10000 / 20000 = 50%
			// Node2 std: (0.6 - 0.5) / 2 = 0.05
			// Node2 Score: (1 - 0.05)*MaxNodeScore = 95
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 20000, nil)},
			expectedList: []framework.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 95}},
			name:         "resources requested, pods scheduled with resources if PreScore not called",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fh, _ := runtime.NewFramework(ctx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			p, _ := NewBalancedAllocation(ctx, &test.args, fh, feature.Features{})
			state := framework.NewCycleState()
			if test.runPreScore {
				status := p.(framework.PreScorePlugin).PreScore(ctx, state, test.pod, tf.BuildNodeInfos(test.nodes))
				if status.Code() != test.wantPreScoreStatusCode {
					t.Errorf("unexpected status code, want: %v, got: %v", test.wantPreScoreStatusCode, status.Code())
				}
				if status.Code() == framework.Skip {
					t.Log("skipping score test as PreScore returned skip")
					return
				}
			}
			for i := range test.nodes {
				nodeInfo, err := snapshot.Get(test.nodes[i].Name)
				if err != nil {
					t.Errorf("failed to get node %q from snapshot: %v", test.nodes[i].Name, err)
				}
				hostResult, status := p.(framework.ScorePlugin).Score(ctx, state, test.pod, nodeInfo)
				if !status.IsSuccess() {
					t.Errorf("Score is expected to return success, but didn't. Got status: %v", status)
				}
				if diff := cmp.Diff(test.expectedList[i].Score, hostResult); diff != "" {
					t.Errorf("unexpected score for host %v (-want,+got):\n%s", test.nodes[i].Name, diff)
				}
			}
		})
	}
}
