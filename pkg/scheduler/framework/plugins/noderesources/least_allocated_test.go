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

func TestNodeResourcesLeastAllocated(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	machine1Spec := v1.PodSpec{
		NodeName: "machine1",
	}
	machine2Spec := v1.PodSpec{
		NodeName: "machine2",
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
	defaultResourceLeastAllocatedSet := []config.ResourceSpec{
		{Name: string(v1.ResourceCPU), Weight: 1},
		{Name: string(v1.ResourceMemory), Weight: 1},
	}
	extendedRes := "abc.com/xyz"
	extendedResourceLeastAllocatedSet := []config.ResourceSpec{
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
		args         config.NodeResourcesLeastAllocatedArgs
		wantErr      error
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			// Node1 scores (remaining resources) on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 0) * MaxNodeScore) / 4000 = MaxNodeScore
			// Memory Score: ((10000 - 0) * MaxNodeScore) / 10000 = MaxNodeScore
			// Node1 Score: (100 + 100) / 2 = 100
			// Node2 scores (remaining resources) on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 0) * MaxNodeScore) / 4000 = MaxNodeScore
			// Memory Score: ((10000 - 0) * MaxNodeScore) / 10000 = MaxNodeScore
			// Node2 Score: (MaxNodeScore + MaxNodeScore) / 2 = MaxNodeScore
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "nothing scheduled, nothing requested",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 3000) * MaxNodeScore) / 4000 = 25
			// Memory Score: ((10000 - 5000) * MaxNodeScore) / 10000 = 50
			// Node1 Score: (25 + 50) / 2 = 37
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((6000 - 3000) * MaxNodeScore) / 6000 = 50
			// Memory Score: ((10000 - 5000) * MaxNodeScore) / 10000 = 50
			// Node2 Score: (50 + 50) / 2 = 50
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 37}, {Name: "machine2", Score: 50}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: []config.ResourceSpec{}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "Resources not set, nothing scheduled, resources requested, differently sized machines",
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 0) * MaxNodeScore) / 4000 = MaxNodeScore
			// Memory Score: ((10000 - 0) * MaxNodeScore) / 10000 = MaxNodeScore
			// Node1 Score: (MaxNodeScore + MaxNodeScore) / 2 = MaxNodeScore
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 0) * MaxNodeScore) / 4000 = MaxNodeScore
			// Memory Score: ((10000 - 0) * MaxNodeScore) / 10000 = MaxNodeScore
			// Node2 Score: (MaxNodeScore + MaxNodeScore) / 2 = MaxNodeScore
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: framework.MaxNodeScore}},
			name:         "no resources requested, pods scheduled",
			pods: []*v1.Pod{
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((20000 - 0) * MaxNodeScore) / 20000 = MaxNodeScore
			// Node1 Score: (40 + 100) / 2 = 70
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((20000 - 5000) * MaxNodeScore) / 20000 = 75
			// Node2 Score: (40 + 75) / 2 = 57
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 70}, {Name: "machine2", Score: 57}},
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
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((20000 - 5000) * MaxNodeScore) / 20000 = 75
			// Node1 Score: (40 + 75) / 2 = 57
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((20000 - 10000) * MaxNodeScore) / 20000 = 50
			// Node2 Score: (40 + 50) / 2 = 45
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 57}, {Name: "machine2", Score: 45}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((20000 - 5000) * MaxNodeScore) / 20000 = 75
			// Node1 Score: (40 + 75) / 2 = 57
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((10000 - 6000) * MaxNodeScore) / 10000 = 40
			// Memory Score: ((50000 - 10000) * MaxNodeScore) / 50000 = 80
			// Node2 Score: (40 + 80) / 2 = 60
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 50000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 57}, {Name: "machine2", Score: 60}},
			name:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// Node1 scores on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 6000) * MaxNodeScore) / 4000 = 0
			// Memory Score: ((10000 - 0) * MaxNodeScore) / 10000 = MaxNodeScore
			// Node1 Score: (0 + MaxNodeScore) / 2 = 50
			// Node2 scores on 0-MaxNodeScore scale
			// CPU Score: ((4000 - 6000) * MaxNodeScore) / 4000 = 0
			// Memory Score: ((10000 - 5000) * MaxNodeScore) / 10000 = 50
			// Node2 Score: (0 + 50) / 2 = 25
			pod:          &v1.Pod{Spec: cpuOnly},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 25}},
			name:         "requested resources exceed node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNode("machine1", 0, 0), makeNode("machine2", 0, 0)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: defaultResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 0}},
			name:         "zero node resources, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			// CPU Score: ((4000 - 3000) *100) / 4000 = 25
			// Memory Score: ((10000 - 5000) *100) / 10000 = 50
			// Node1 Score: (25 * 1 + 50 * 2) / (1 + 2) = 41
			// CPU Score: ((6000 - 3000) *100) / 6000 = 50
			// Memory Score: ((10000 - 5000) *100) / 10000 = 50
			// Node2 Score: (50 * 1 + 50 * 2) / (1 + 2) = 50
			pod:          &v1.Pod{Spec: cpuAndMemory},
			nodes:        []*v1.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 2}, {Name: "cpu", Weight: 1}}},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 41}, {Name: "machine2", Score: 50}},
			name:         "nothing scheduled, resources requested with different weight on CPU and memory, differently sized machines",
		},
		{
			// resource with negative weight is not allowed
			pod:   &v1.Pod{Spec: cpuAndMemory},
			nodes: []*v1.Node{makeNode("machine", 4000, 10000)},
			args:  config.NodeResourcesLeastAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: -1}, {Name: "cpu", Weight: 1}}},
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
			args:  config.NodeResourcesLeastAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 1}, {Name: "cpu", Weight: 0}}},
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
			args:  config.NodeResourcesLeastAllocatedArgs{Resources: []config.ResourceSpec{{Name: "memory", Weight: 120}}},
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
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: extendedResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 50}},
			name:         "bypass extended resource if the pod does not request",
		},
		{
			// Honor extended resource if the pod requests.
			// For both nodes: cpuScore and memScore are 50.
			// In terms of extended resource score:
			// - node1 get: 2 / 4 * 100 = 50
			// - node2 get: (10 - 2) / 10 * 100 = 80
			// So the final scores are:
			// - node1: (50 + 50 + 50) / 3 = 50
			// - node2: (50 + 50 + 80) / 3 = 60
			pod: &v1.Pod{Spec: cpuMemoryAndExtendedRes},
			nodes: []*v1.Node{
				makeNodeWithExtendedResource("machine1", 6000, 10000, map[string]int64{extendedRes: 4}),
				makeNodeWithExtendedResource("machine2", 6000, 10000, map[string]int64{extendedRes: 10}),
			},
			args:         config.NodeResourcesLeastAllocatedArgs{Resources: extendedResourceLeastAllocatedSet},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 50}, {Name: "machine2", Score: 60}},
			name:         "honor extended resource if the pod requests",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			snapshot := cache.NewSnapshot(test.pods, test.nodes)
			fh, _ := runtime.NewFramework(nil, nil, runtime.WithSnapshotSharedLister(snapshot))
			p, err := NewLeastAllocated(&test.args, fh, feature.Features{EnablePodOverhead: true})

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
				t.Fatalf("failed to initialize plugin NodeResourcesLeastAllocated, got error: %v", err)
			}

			for i := range test.nodes {
				hostResult, err := p.(framework.ScorePlugin).Score(context.Background(), nil, test.pod, test.nodes[i].Name)
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if !reflect.DeepEqual(test.expectedList[i].Score, hostResult) {
					t.Errorf("expected %#v, got %#v", test.expectedList[i].Score, hostResult)
				}
			}
		})
	}
}
