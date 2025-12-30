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
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestNodeResourcesBalancedAllocation(t *testing.T) {
	testNodeResourcesBalancedAllocation(ktesting.Init(t))
}
func testNodeResourcesBalancedAllocation(tCtx ktesting.TContext) {
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
		expectedList           fwk.NodeScoreList
		name                   string
		args                   config.NodeResourcesBalancedAllocationArgs
		runPreScore            bool
		wantPreScoreStatusCode fwk.Code
		draObjects             []apiruntime.Object
	}{
		{
			// bestEffort pods, skip in PreScore
			pod:                    st.MakePod().Obj(),
			nodes:                  []*v1.Node{makeNode("node1", 4000, 10000, nil), makeNode("node2", 4000, 10000, nil)},
			name:                   "nothing scheduled, nothing requested, skip in PreScore",
			args:                   config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:            true,
			wantPreScoreStatusCode: fwk.Skip,
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 87}, {Name: "node2", Score: fwk.MaxNodeScore}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 95}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 80}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 50}, {Name: "node2", Score: 75}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 70}, {Name: "node2", Score: 65}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 63}, {Name: "node2", Score: 63}},
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
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 82}, {Name: "node2", Score: 95}},
			name:         "resources requested, pods scheduled with resources if PreScore not called",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: false,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 3500 = 0.8571
			// Memory Fraction: 0 / 40000 = 0
			// DRA Fraction: 1 / 8 = 0.125
			// Fraction mean: (0.8571 + 0 + 0.125) / 3 = 0.3274
			// Node1 std: sqrt(((0.8571 - 0.3274)**2 + (0- 0.3274)**2 + (0.125 - 0.3274)**2) / 3) = 0.378
			// Node1 Score: (1 - 0.378)*MaxNodeScore = 62
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 3500 = 0.8571
			// Memory Fraction: 5000 / 40000 = 0.125
			// Node2 std: (0.8571 - 0.125) / 2 = 0.36605
			// Node2 Score: (1 - 0.36605)*MaxNodeScore = 63
			pod:          st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, nil), makeNode("node2", 3500, 40000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 62}, {Name: "node2", Score: 63}},
			name:         "include DRA resource on a node for balanced resource allocation",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: extendedResourceName, Weight: 1},
			}},
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice("node1", "test-driver").Device("device-1").Device("device-2").Device("device-3").Device("device-4").Device("device-5").Device("device-6").Device("device-7").Device("device-8").Obj(),
			},
			runPreScore: true,
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 35000 = 0.8571
			// Memory Fraction: 0 / 40000 = 0
			// DRA Fraction: 1 / 8 = 0.125
			// Fraction mean: (0.8571 + 0 + 0.125) / 3 = 0.3274
			// Node1 std: sqrt(((0.8571 - 0.3274)**2 + (0- 0.3274)**2 + (0.125 - 0.3274)**2) / 3) = 0.378
			// Node1 Score: (1 - 0.378)*MaxNodeScore = 62
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 3000 / 35000 = 0.8571
			// Memory Fraction: 5000 / 40000 = 0.125
			// Node2 std: (0.8571 - 0.125) / 2 = 0.36605
			// Node2 Score: (1 - 0.36605)*MaxNodeScore = 63
			pod:          st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, nil), makeNode("node2", 3500, 40000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 62}, {Name: "node2", Score: 63}},
			name:         "include DRA resource on a node for balanced resource allocation if PreScore not called",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args: config.NodeResourcesBalancedAllocationArgs{Resources: []config.ResourceSpec{
				{Name: string(v1.ResourceCPU), Weight: 1},
				{Name: string(v1.ResourceMemory), Weight: 1},
				{Name: extendedResourceName, Weight: 1},
			}},
			draObjects: []apiruntime.Object{
				deviceClassWithExtendResourceName,
				st.MakeResourceSlice("node1", "test-driver").Device("device-1").Device("device-2").Device("device-3").Device("device-4").Device("device-5").Device("device-6").Device("device-7").Device("device-8").Obj(),
			},
			runPreScore: false,
		},
	}

	for _, test := range tests {
		tCtx.SyncTest(test.name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, test.draObjects != nil)
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(tCtx, nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			defer func() {
				tCtx.Cancel("test has completed")
				runtime.WaitForShutdown(fh)
			}()
			p, _ := NewBalancedAllocation(tCtx, &test.args, fh, feature.Features{
				EnableDRAExtendedResource: test.draObjects != nil,
			})

			draManager := newTestDRAManager(tCtx, test.draObjects...)
			p.(*BalancedAllocation).draManager = draManager

			state := framework.NewCycleState()
			if test.runPreScore {
				status := p.(fwk.PreScorePlugin).PreScore(tCtx, state, test.pod, tf.BuildNodeInfos(test.nodes))
				if status.Code() != test.wantPreScoreStatusCode {
					tCtx.Errorf("unexpected status code, want: %v, got: %v", test.wantPreScoreStatusCode, status.Code())
				}
				if status.Code() == fwk.Skip {
					tCtx.Log("skipping score test as PreScore returned skip")
					return
				}
			}
			for i := range test.nodes {
				nodeInfo, err := snapshot.Get(test.nodes[i].Name)
				if err != nil {
					tCtx.Errorf("failed to get node %q from snapshot: %v", test.nodes[i].Name, err)
				}
				hostResult, status := p.(fwk.ScorePlugin).Score(tCtx, state, test.pod, nodeInfo)
				if !status.IsSuccess() {
					tCtx.Errorf("Score is expected to return success, but didn't. Got status: %v", status)
				}
				if diff := cmp.Diff(test.expectedList[i].Score, hostResult); diff != "" {
					tCtx.Errorf("unexpected score for host %v (-want,+got):\n%s", test.nodes[i].Name, diff)
				}
			}
		})
	}
}

func TestBalancedAllocationSignPod(t *testing.T) {
	tests := map[string]struct {
		name                      string
		pod                       *v1.Pod
		enableDRAExtendedResource bool
		expectedFragments         []fwk.SignFragment
		expectedStatusCode        fwk.Code
	}{
		"pod with CPU and memory requests": {
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "2000",
			}).Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "1000m",
					v1.ResourceMemory: "2000",
				}).Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"best-effort pod with no requests": {
			pod:                       st.MakePod().Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"pod with multiple containers": {
			pod: st.MakePod().Container("container1").Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "500m",
				v1.ResourceMemory: "1000",
			}).Container("container2").Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1500m",
				v1.ResourceMemory: "3000",
			}).Obj(),
			enableDRAExtendedResource: false,
			expectedFragments: []fwk.SignFragment{
				{Key: fwk.ResourcesSignerName, Value: computePodResourceRequest(st.MakePod().Container("container1").Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "500m",
					v1.ResourceMemory: "1000",
				}).Container("container2").Req(map[v1.ResourceName]string{
					v1.ResourceCPU:    "1500m",
					v1.ResourceMemory: "3000",
				}).Obj(), ResourceRequestsOptions{})},
			},
			expectedStatusCode: fwk.Success,
		},
		"DRA extended resource enabled - returns unschedulable": {
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceCPU:    "1000m",
				v1.ResourceMemory: "2000",
			}).Obj(),
			enableDRAExtendedResource: true,
			expectedFragments:         nil,
			expectedStatusCode:        fwk.Unschedulable,
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			p, err := NewBalancedAllocation(ctx, &config.NodeResourcesBalancedAllocationArgs{}, nil, feature.Features{
				EnableDRAExtendedResource: test.enableDRAExtendedResource,
			})
			if err != nil {
				t.Fatalf("failed to create plugin: %v", err)
			}

			ba := p.(*BalancedAllocation)
			fragments, status := ba.SignPod(ctx, test.pod)

			if status.Code() != test.expectedStatusCode {
				t.Errorf("unexpected status code, want: %v, got: %v, message: %v", test.expectedStatusCode, status.Code(), status.Message())
			}

			if test.expectedStatusCode == fwk.Success {
				if diff := cmp.Diff(test.expectedFragments, fragments); diff != "" {
					t.Errorf("unexpected fragments, diff (-want,+got):\n%s", diff)
				}
			}
		})
	}
}
