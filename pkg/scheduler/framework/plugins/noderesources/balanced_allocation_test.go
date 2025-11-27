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
	memoryOnly := v1.PodSpec{
		NodeName: "node1",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0"),
						v1.ResourceMemory: resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("0"),
						v1.ResourceMemory: resource.MustParse("3000"),
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
			// Node1
			//  CPU: 0 -> 3000/4000 (0% -> 75%)
			//  Memory: 0 -> 5000/10000 (0% -> 50%)
			//  Score: 68 (100 -> 87)
			// Node2
			//  CPU: 0 -> 3000/6000 (0% -> 50%)
			//  Memory: 0 -> 5000/10000 (0% -> 50%)
			//  Score: 75 (100 -> 100)
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 4000, 10000, nil), makeNode("node2", 6000, 10000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 68}, {Name: "node2", Score: 75}},
			name:         "nothing scheduled, resources requested, differently sized nodes",
			args:         config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore:  true,
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
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 20000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 74}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
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
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 50000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 70}},
			name:         "resources requested, pods scheduled with resources, differently sized nodes",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
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
			pod:          &v1.Pod{Spec: memoryOnly},
			nodes:        []*v1.Node{makeNode("node1", 3000, 5000, nil), makeNode("node2", 3000, 5000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 100}, {Name: "node2", Score: 50}},
			name:         "resources requested, pods scheduled with resources, nodes to reach min/max score",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
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
			pod:          &v1.Pod{Spec: cpuOnly},
			nodes:        []*v1.Node{makeNode("node1", 6000, 10000, nil), makeNode("node2", 6000, 10000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 62}, {Name: "node2", Score: 62}},
			name:         "requested resources at node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: true,
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
			pod: st.MakePod().Req(map[v1.ResourceName]string{
				v1.ResourceMemory: "0",
				"nvidia.com/gpu":  "1",
			}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, scalarResource), makeNode("node2", 3500, 40000, scalarResource)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 75}, {Name: "node2", Score: 76}},
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
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, scalarResource), makeNode("node2", 3500, 40000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 56}, {Name: "node2", Score: 56}},
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
			// Whether or not prescore was called, the end result should be the same.
			// Node1
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 0 -> 5000/20000 (0% -> 25%)
			//  Score: 73 (85 -> 82)
			// Node2
			//  CPU: 3000 -> 6000/10000 (30% -> 60%)
			//  Memory: 5000 -> 10000/20000 (25% -> 50%)
			//  Score: 74 (97 -> 95)
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("node1", 10000, 20000, nil), makeNode("node2", 10000, 20000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 73}, {Name: "node2", Score: 74}},
			name:         "resources requested, pods scheduled with resources if PreScore not called",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
			args:        config.NodeResourcesBalancedAllocationArgs{Resources: defaultResourceBalancedAllocationSet},
			runPreScore: false,
		},
		{
			// Node1
			//  CPU: 3000 -> 3000/3500 (86% -> 86%)
			//  Memory: 0 -> 0/40000 (0% -> 0%)
			//  DRA: 0 -> 1/8 (0% -> 12%)
			//  Score: 76 (60 -> 62)
			// Node2
			//  CPU: 3000 -> 3000/3500 (86% -> 86%)
			//  Memory: 5000 -> 5000/40000 (12% -> 12%)
			//  DRA: unsatisfiable
			//  Score: 75 (63 -> 63)
			pod:          st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, nil), makeNode("node2", 3500, 40000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 76}, {Name: "node2", Score: 75}},
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
			// Node1
			//  CPU: 3000 -> 3000/3500 (86% -> 86%)
			//  Memory: 0 -> 0/40000 (0% -> 0%)
			//  DRA: 0 -> 1/8 (0% -> 12%)
			//  Score: 76 (60 -> 62)
			// Node2
			//  CPU: 3000 -> 3000/3500 (86% -> 86%)
			//  Memory: 5000 -> 5000/40000 (12% -> 12%)
			//  DRA: unsatisfiable
			//  Score: 75 (63 -> 63)
			pod:          st.MakePod().Req(map[v1.ResourceName]string{extendedResourceDRA: "1"}).Obj(),
			nodes:        []*v1.Node{makeNode("node1", 3500, 40000, nil), makeNode("node2", 3500, 40000, nil)},
			expectedList: []fwk.NodeScore{{Name: "node1", Score: 76}, {Name: "node2", Score: 75}},
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
