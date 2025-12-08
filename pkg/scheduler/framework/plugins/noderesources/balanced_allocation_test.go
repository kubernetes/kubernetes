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
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

func TestNodeResourcesBalancedAllocation(t *testing.T) {
	defaultResourceBalancedAllocationSet := []config.ResourceSpec{
		{Name: string(v1.ResourceCPU), Weight: 1},
		{Name: string(v1.ResourceMemory), Weight: 1},
	}
	scalarResource := map[string]int64{
		"nvidia.com/gpu": 8,
	}

	tests := []struct {
		name                       string
		requestedPod               *v1.Pod
		nodes                      []*v1.Node
		existingPods               []*v1.Pod
		expectedScores             fwk.NodeScoreList
		args                       config.NodeResourcesBalancedAllocationArgs
		runPreScore                bool
		expectedPreScoreStatusCode fwk.Code
	}{
		{
			// bestEffort pods, skip in PreScore
			name:                       "nothing scheduled, nothing requested, skip in PreScore",
			requestedPod:               st.MakePod().Obj(),
			nodes:                      []*v1.Node{makeNode("node1", 4000, 10000, nil), makeNode("node2", 4000, 10000, nil)},
			args:                       config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:                true,
			expectedPreScoreStatusCode: fwk.Skip,
		},
		{
			// Node1
			//  CPU: 0 -> 3000/4000 (0% -> 75%)
			//  Memory: 0 -> 5000/10000 (0% -> 50%)
			//  Score: 68 (100 -> 87)
			// Node2
			//  CPU: 0 -> 3000/6000 (0% -> 50%)
			//  Memory: 0 -> 5000/10000 (0% -> 50%)
			//  Score: 75 (100 -> 100)
			name:         "nothing scheduled, resources requested, differently sized nodes",
			requestedPod: st.MakePod().Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 4000, 10000, nil),
				makeNode("node2", 6000, 10000, nil)},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 68}, {Name: "node2", Score: 75}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    true,
		},
		{
			// Node1
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 0 -> 5000/20000 (0% -> 25%)
			//  Score: 73 (85 -> 82)
			// Node2
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 5000 -> 10000/20000 (25% -> 50%)
			//  Score: 74 (97 -> 95)
			name:         "resources requested, pods scheduled with resources",
			requestedPod: st.MakePod().Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 10000, 20000, nil),
				makeNode("node2", 10000, 20000, nil),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
				st.MakePod().Node("node2").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 74}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    true,
		},
		{
			// Node1
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 0 -> 5000/20000 (0% -> 25%)
			//  Score: 73 (85 -> 82)
			// Node2
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 5000 -> 10000/50000 (10% -> 20%)
			//  Score: 70 (90 -> 80)
			name:         "resources requested, pods scheduled with resources, differently sized nodes",
			requestedPod: st.MakePod().Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 10000, 20000, nil),
				makeNode("node2", 10000, 50000, nil),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
				st.MakePod().Node("node2").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 70}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    true,
		},
		{
			// Node1
			//  CPU: 3000 -> 3000/3000 (100% -> 100%)
			//  Memory: 0 -> 5000/5000 (0% -> 100%)
			//  Score: 100 (50 -> 100)
			// Node2
			//  CPU: 0 -> 0/10000 (0% -> 0%)
			//  Memory: 0 -> 5000/5000 (0% -> 100%)
			//  Score: 50 (100 -> 50)
			name:         "resources requested, pods scheduled with resources, nodes to reach min/max score",
			requestedPod: st.MakePod().Req(map[v1.ResourceName]string{"memory": "2000"}).Req(map[v1.ResourceName]string{"memory": "3000"}).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 3000, 5000, nil),
				makeNode("node2", 3000, 5000, nil),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 100}, {Name: "node2", Score: 50}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    true,
		},
		{
			// Node1
			//  CPU: 3000 -> 6000/6000 (50% -> 100%)
			//  Memory: 0 -> 0/10000 (0% -> 0%)
			//  Score: 62 (75 -> 50)
			// Node2
			//  CPU: 3000 -> 6000/6000 (50% -> 100%)
			//  Memory: 5000 -> 5000/10000 (50% -> 50%)
			//  Score: 62 (100 -> 75)
			name:         "requested resources at node capacity",
			requestedPod: st.MakePod().Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 6000, 10000, nil),
				makeNode("node2", 6000, 10000, nil),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
				st.MakePod().Node("node2").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 62}, {Name: "node2", Score: 62}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    true,
		},
		{
			// Node1
			//  CPU: 3000 -> 3000/3500 (85.7% -> 85.7%)
			//  Memory: 5000 -> 5000/40000 (12.5% -> 12.5%)
			//  GPU: 3 -> 4/8 (37.5% -> 50%)
			//  Score: 75 (69 -> 70)
			// Node2
			//  CPU: 3000 -> 3000/3500 (85.7% -> 85.7%)
			//  Memory: 5000 -> 5000/40000 (12.5% -> 12.5%)
			//  GPU: 0 -> 1/8 (0% -> 12.5%)
			//  Score: 76 (62 -> 65)
			name:         "include scalar resource on a node for balanced resource allocation",
			requestedPod: st.MakePod().Req(cpuAndMemoryAndGpu("0", "0", "1")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 3500, 40000, scalarResource),
				makeNode("node2", 3500, 40000, scalarResource),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemoryAndGpu("2000m", "3000", "3")).Obj(),
				st.MakePod().Node("node2").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 75}, {Name: "node2", Score: 76}},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: "nvidia.com/gpu", Weight: 1},
			}},
			runPreScore: true,
		},
		{
			// Node1
			//  CPU: 0 -> 3000/3500 (0% -> 85.7%)
			//  Memory: 0 -> 5000/40000 (0% -> 12.5%)
			//  GPU: 0 -> 0/8 (scalar resource not requested by pod, ignored)
			//  Score: 56 (100 -> 63)
			// Node2
			//  CPU: 0 -> 3000/3500 (0% -> 85.7%)
			//  Memory: 0 -> 5000/40000 (0% -> 12.5%)
			//  Score: 56 (100 -> 63)
			name:         "node without the scalar resource should skip the scalar resource",
			requestedPod: st.MakePod().Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 3500, 40000, scalarResource),
				makeNode("node2", 3500, 40000, nil),
			},
			existingPods:   []*v1.Pod{},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 56}, {Name: "node2", Score: 56}},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: "nvidia.com/gpu", Weight: 1},
			}},
			runPreScore: true,
		},
		{
			// Whether or not prescore was called, the end result should be the same.
			// Node1
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 0 -> 5000/20000 (0% -> 25%)
			//  Score: 73 (85 -> 82)
			// Node2
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 5000 -> 10000/20000 (25% -> 50%)
			//  Score: 74 (97 -> 95)
			name:         "resources requested, pods scheduled with resources if PreScore not called",
			requestedPod: st.MakePod().Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			nodes: []*v1.Node{
				makeNode("node1", 10000, 20000, nil),
				makeNode("node2", 10000, 20000, nil),
			},
			existingPods: []*v1.Pod{
				st.MakePod().Node("node1").Req(cpuOnly("1000m")).Req(cpuOnly("2000m")).Obj(),
				st.MakePod().Node("node2").Req(cpuAndMemory("1000m", "2000")).Req(cpuAndMemory("2000m", "3000")).Obj(),
			},
			expectedScores: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 74}},
			args:           config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:    false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(test.existingPods, test.nodes)
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			fh, _ := runtime.NewFramework(ctx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			p, _ := NewBalancedAllocation(ctx, &test.args, fh, feature.Features{})
			state := framework.NewCycleState()
			if test.runPreScore {
				status := p.(fwk.PreScorePlugin).PreScore(ctx, state, test.requestedPod, tf.BuildNodeInfos(test.nodes))
				if status.Code() != test.expectedPreScoreStatusCode {
					t.Errorf("unexpected status code, want: %v, got: %v", test.expectedPreScoreStatusCode, status.Code())
				}
				if status.Code() == fwk.Skip {
					t.Log("skipping score test as PreScore returned skip")
					return
				}
			}
			for i := range test.nodes {
				nodeInfo, err := snapshot.Get(test.nodes[i].Name)
				if err != nil {
					t.Errorf("failed to get node %q from snapshot: %v", test.nodes[i].Name, err)
				}
				hostResult, status := p.(fwk.ScorePlugin).Score(ctx, state, test.requestedPod, nodeInfo)
				if !status.IsSuccess() {
					t.Errorf("Score is expected to return success, but didn't. Got status: %v", status)
				}
				if diff := cmp.Diff(test.expectedScores[i].Score, hostResult); diff != "" {
					t.Errorf("unexpected score for host %v (-want,+got):\n%s", test.nodes[i].Name, diff)
				}
			}
		})
	}
}

func cpuOnly(req string) map[v1.ResourceName]string {
	return map[v1.ResourceName]string{"cpu": req}
}
func cpuAndMemory(cpuReq, memoryReq string) map[v1.ResourceName]string {
	return map[v1.ResourceName]string{"cpu": cpuReq, "memory": memoryReq}
}
func cpuAndMemoryAndGpu(cpuReq, memoryReq, gpuReq string) map[v1.ResourceName]string {
	return map[v1.ResourceName]string{"cpu": cpuReq, "memory": memoryReq, "nvidia.com/gpu": gpuReq}
}
