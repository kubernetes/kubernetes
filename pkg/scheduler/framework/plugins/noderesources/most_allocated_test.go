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
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/internal/cache"
)

func TestNodeResourcesMostAllocated(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	cpuOnly := v1.PodSpec{
		NodeName: "machine1",
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
	cpuOnly2.NodeName = "machine2"
	cpuAndMemory := v1.PodSpec{
		NodeName: "machine2",
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
	bigCPUAndMemory := v1.PodSpec{
		NodeName: "machine1",
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("2000m"),
						v1.ResourceMemory: resource.MustParse("4000"),
					},
				},
			},
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("3000m"),
						v1.ResourceMemory: resource.MustParse("5000"),
					},
				},
			},
		},
	}
	nonZeroContainer := v1.PodSpec{
		Containers: []v1.Container{{}},
	}
	nonZeroContainer1 := v1.PodSpec{
		NodeName:   "machine1",
		Containers: []v1.Container{{}},
	}
	defaultResourceMostAllocatedSet := []config.ResourceSpec{
		{Name: string(v1.ResourceCPU), Weight: 1},
		{Name: string(v1.ResourceMemory), Weight: 1},
	}
	extendedRes := "abc.com/xyz"
	extendedResourceMostAllocatedSet := []config.ResourceSpec{
		{Name: string(v1.ResourceCPU), Weight: 1},
		{Name: string(v1.ResourceMemory), Weight: 1},
		{Name: extendedRes, Weight: 1},
	}
	cpuMemoryAndExtendedRes := *cpuAndMemory.DeepCopy()
	cpuMemoryAndExtendedRes.Containers[0].Resources.Requests[v1.ResourceName(extendedRes)] = resource.MustParse("2")
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		args         config.NodeResourcesMostAllocatedArgs
		wantErr      error
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			// Node1 scores (used resources) on 0-MaxNodeScore scale
			// CPU Score: (0 * MaxNodeScore)  / 4000 = 0
			// Memory Score: (0 * MaxNodeScore) / 10000 = 0
			// Node1 Score: (0 + 0) / 2 = 0
			// Node2 scores (used resources) on 0-MaxNodeScore scale
			// CPU Score: (0 * MaxNodeScore) / 4000 = 0
			// Memory Score: (0 * MaxNodeScore) / 10000 = 0
			// Node2 Score: (0 + 0) / 2 = 0
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: defaultResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: (3000 * MaxNodeScore) / 4000 = 75
			// Memory Score: (5000 * MaxNodeScore) / 10000 = 50
			// Node1 Score: (75 + 50) / 2 = 6
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: (3000 * MaxNodeScore) / 6000 = 50
			// Memory Score: (5000 * MaxNodeScore) / 10000 = 50
			// Node2 Score: (50 + 50) / 2 = 50
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: defaultResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 62}, {Name: "machine2", Score: 50}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "Resources not set, nothing scheduled, resources requested, differently sized machines",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: (6000 * MaxNodeScore) / 10000 = 60
			// Memory Score: (0 * MaxNodeScore) / 20000 = 0
			// Node1 Score: (60 + 0) / 2 = 30
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: (6000 * MaxNodeScore) / 10000 = 60
			// Memory Score: (5000 * MaxNodeScore) / 20000 = 25
			// Node2 Score: (60 + 25) / 2 = 42
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: defaultResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 30}, {Name: "machine2", Score: 42}},
			name:         "no resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly2, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemory, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: (6000 * MaxNodeScore) / 10000 = 60
			// Memory Score: (5000 * MaxNodeScore) / 20000 = 25
			// Node1 Score: (60 + 25) / 2 = 42
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: (6000 * MaxNodeScore) / 10000 = 60
			// Memory Score: (10000 * MaxNodeScore) / 20000 = 50
			// Node2 Score: (60 + 50) / 2 = 55
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: defaultResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 42}, {Name: "machine2", Score: 55}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: 5000 * MaxNodeScore / 5000 return 100
			// Memory Score: (9000 * MaxNodeScore) / 10000 = 90
			// Node1 Score: (100 + 90) / 2 = 95
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: (5000 * MaxNodeScore) / 10000 = 50
			// Memory Score: 8000 *MaxNodeScore / 8000 return 100
			// Node2 Score: (50 + 100) / 2 = 75
			pod:          &v1.Pod{Spec: bigCPUAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 5000, 10000), makeNode("machine2", 10000, 9000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: defaultResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 95}, {Name: "machine2", Score: 75}},
			name:         "resources requested equal node capacity",
		},
		{
			// CPU Score: (3000 *100) / 4000 = 75
			// Memory Score: (5000 *100) / 10000 = 50
			// Node1 Score: (75 * 1 + 50 * 2) / (1 + 2) = 58
			// CPU Score: (3000 *100) / 6000 = 50
			// Memory Score: (5000 *100) / 10000 = 50
			// Node2 Score: (50 * 1 + 50 * 2) / (1 + 2) = 50
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 2}, {Name: "cpu", Weight: 1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 58}, {Name: "machine2", Score: 50}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Fraction: 300 / 250 = 100%
			// Memory Fraction: 600 / 10000 = 60%
			// Node1 Score: (100 + 60) / 2 = 80
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Fraction: 100 / 250 = 40%
			// Memory Fraction: 200 / 10000 = 20%
			// Node2 Score: (20 + 40) / 2 = 30
			pod:          &v1.Pod{Spec: nonZeroContainer},
			nodes:        []*v1.Node{makeNode("machine1", 250, 1000*1024*1024), makeNode("machine2", 250, 1000*1024*1024)},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 1}, {Name: "cpu", Weight: 1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 80}, {Name: "machine2", Score: 30}},
			name:         "no resources requested, pods scheduled",
			pods: []*v1.Pod{
				{Spec: nonZeroContainer1},
				{Spec: nonZeroContainer1},
			},
		},
		{
			// resource with negtive weight is not allowed
			pod:   &v1.Pod{Spec: cpuAndMemory},
			nodes: []*v1.Node{makeNode("machine", 4000, 10000)},
			args:  config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: -1}, {Name: "cpu", Weight: 1}}},
			wantErr: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "resources[0].weight",
				},
			}.ToAggregate(),
			name: "resource with negtive weight",
		},
		{
			// resource with zero weight is not allowed
			pod:   &v1.Pod{Spec: cpuAndMemory},
			nodes: []*v1.Node{makeNode("machine", 4000, 10000)},
			args:  config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 1}, {Name: "cpu", Weight: 0}}},
			wantErr: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "resources[1].weight",
				},
			}.ToAggregate(),
			name: "resource with zero weight",
		},
		{
			// resource weight should be less than MaxNodeScore
			pod:   &v1.Pod{Spec: cpuAndMemory},
			nodes: []*v1.Node{makeNode("machine", 4000, 10000)},
			args:  config.NodeResourcesMostAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 120}}},
			wantErr: field.ErrorList{
				&field.Error{
					Type:  field.ErrorTypeInvalid,
					Field: "resources[0].weight",
				},
			}.ToAggregate(),
			name: "resource weight larger than MaxNodeScore",
		},
		{
			// Bypass extended resource if the pod does not request.
			// For both nodes: cpuScore and memScore are 50
			// Given that extended resource score are intentionally bypassed,
			// the final scores are:
			// - node1: (50 + 50) / 2 = 50
			// - node2: (50 + 50) / 2 = 50
			pod: &v1.Pod{Spec: cpuAndMemory},
			nodes: []*v1.Node{
				makeNode("machine1", 6000, 10000),
				makeNodeWithExtendedResource("machine2", 6000, 10000, map[string]int64{extendedRes: 4}),
			},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: extendedResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 50}},
			name:         "bypass extended resource if the pod does not request",
		},
		{
			// Honor extended resource if the pod requests.
			// For both nodes: cpuScore and memScore are 50.
			// In terms of extended resource score:
			// - node1 get: 2 / 4 * 100 = 50
			// - node2 get: 2 / 10 * 100 = 20
			// So the final scores are:
			// - node1: (50 + 50 + 50) / 3 = 50
			// - node2: (50 + 50 + 20) / 3 = 40
			pod: &v1.Pod{Spec: cpuMemoryAndExtendedRes},
			nodes: []*v1.Node{
				makeNodeWithExtendedResource("machine1", 6000, 10000, map[string]int64{extendedRes: 4}),
				makeNodeWithExtendedResource("machine2", 6000, 10000, map[string]int64{extendedRes: 10}),
			},
			args:         config.NodeResourcesMostAllocatedArgs{Resources: extendedResourceMostAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 40}},
			name:         "honor extended resource if the pod requests",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			p, err := NewMostAllocated(&test.args, fh, feature.Features{EnablePodOverhead: true})

			if test.wantErr != nil {
				if err != nil {
					diff := cmp.Diff(test.wantErr, err, cmpopts.IgnoreFields(field.Error{}, "BadValue", "Detail"))
					if diff != "" {
						t.Fatalf("got err (-want,+got):\n%s", diff)
					}
				} else {
					t.Fatalf("no error produced, wanted %v", test.wantErr)
				}
				return
			}

			if err != nil && test.wantErr == nil {
				t.Fatalf("failed to initialize plugin NodeResourcesMostAllocated, got error: %v", err)
			}

			for i := range test.nodes {
				hostResult, err := p.(framework.ScorePlugin).Score(context.Background(), nil, test.pod, test.nodes[i].Name)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !reflect.DeepEqual(test.expectedList[i].Score, hostResult) {
					t.Errorf("got score %v for host %v, expected %v", hostResult, test.nodes[i].Name, test.expectedList[i].Score)
				}
			}
		})
	}
}
