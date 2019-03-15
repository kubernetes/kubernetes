/*
Copyright 2016 The Kubernetes Authors.

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

package priorities

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestLeastRequestedExtended(t *testing.T) {
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
	cpuAndMemoryAndGpu := v1.PodSpec{
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
						NvidiaGpu:         resource.MustParse("1"),
					},
				},
			},
		},
	}
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			/*
			   Node1 scores (remaining resources) on 0-10 scale
			   CPU Score: ((4000 - 0) *10) / 4000 = 10
			   Memory Score: ((10000 - 0) *10) / 10000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node1 Score: (10 + 10 + 10) / 3 = 10

			   Node2 scores (remaining resources) on 0-10 scale
			   CPU Score: ((4000 - 0) *10) / 4000 = 10
			   Memory Score: ((10000 - 0) *10) / 10000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node2 Score: (10 + 10 + 10) / 3 = 10
			*/
			pod: &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			name:         "nothing scheduled, nothing requested",
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((4000 - 3000) *10) / 4000 = 2.5
			   Memory Score: ((10000 - 5000) *10) / 10000 = 5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node1 Score: (2.5 + 5 + 7.5) / 3 = 5

			   Node2 scores on 0-10 scale
			   CPU Score: ((6000 - 3000) *10) / 6000 = 5
			   Memory Score: ((10000 - 5000) *10) / 10000 = 5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node2 Score: (5 + 5 + 7.5) / 3 = 5.833 -> 5
			*/
			pod: &v1.Pod{Spec: cpuAndMemoryAndGpu},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 6000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 5}},
			name:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((4000 - 0) *10) / 4000 = 10
			   Memory Score: ((10000 - 0) *10) / 10000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node1 Score: (10 + 10 + 10) / 3 = 10

			   Node2 scores on 0-10 scale
			   CPU Score: ((4000 - 0) *10) / 4000 = 10
			   Memory Score: ((10000 - 0) *10) / 10000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node2 Score: (10 + 10 + 10) / 3 = 10
			*/
			pod: &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: schedulerapi.MaxPriority}, {Host: "machine2", Score: schedulerapi.MaxPriority}},
			name:         "no resources requested, pods scheduled without resources",
			pods: []*v1.Pod{
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: machine1Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			    Memory Score: ((20000 - 0) *10) / 20000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node1 Score: (4 + 10 + 10) / 3 = 8

			   Node2 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			   Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node2 Score: (4 + 7.5 + 7.5) / 3 = 6.333 -> 6
			*/
			pod: &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 10000, 20000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 10000, 20000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 8}, {Host: "machine2", Score: 6}},
			name:         "no resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly2, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemoryAndGpu, ObjectMeta: metav1.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			   Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node1 Score: (4 + 7.5 + 7.5) / 3 = 6.333 -> 6

			   Node2 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			   Memory Score: ((20000 - 10000) *10) / 20000 = 5
			   GPU Score: ((4 - 2) *10) / 4 = 5
			   Node2 Score: (4 + 5 + 5) / 3 = 4.6667 -> 4
			*/
			pod: &v1.Pod{Spec: cpuAndMemoryAndGpu},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 10000, 20000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 10000, 20000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 4}},
			name:         "resources requested, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemoryAndGpu},
			},
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			   Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node1 Score: (4 + 7.5 + 7.5) / 3 = 6.333 -> 6

			   Node2 scores on 0-10 scale
			   CPU Score: ((10000 - 6000) *10) / 10000 = 4
			   Memory Score: ((50000 - 10000) *10) / 50000 = 8
			   GPU Score: ((4 - 2) *10) / 4 = 5
			   Node2 Score: (4 + 8 + 5) / 3 = 5.6667 -> 5
			*/
			pod: &v1.Pod{Spec: cpuAndMemoryAndGpu},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 10000, 20000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 10000, 50000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemoryAndGpu},
			},
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((4000 - 6000) *10) / 4000 = 0
			   Memory Score: ((10000 - 0) *10) / 10000 = 10
			   GPU Score: ((4 - 0) *10) / 4 = 10
			   Node1 Score: (0 + 10 + 10) / 3 = 6.6667 -> 6

			   Node2 scores on 0-10 scale
			   CPU Score: ((4000 - 6000) *10) / 4000 = 0
			   Memory Score: ((10000 - 5000) *10) / 10000 = 5
			   GPU Score: ((4 - 1) *10) / 4 = 7.5
			   Node2 Score: (0 + 5 + 7.5) / 3 = 4.5 -> 4
			*/
			pod: &v1.Pod{Spec: cpuOnly},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
				makeNodeWithAllResources("machine2", 4000, 10000, 0, map[v1.ResourceName]int64{NvidiaGpu: 4}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 4}},
			name:         "requested resources exceed node capacity",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemoryAndGpu},
			},
		},
		{
			/*
			   Node1 scores on 0-10 scale
			   CPU Score: ((4000 - 6000) *10) / 4000 = 0
			   Memory Score: ((10000 - 0) *10) / 10000 = 0
			   Node1 Score: (0 + 0) / 2 = 0

			   Node2 scores on 0-10 scale
			   CPU Score: ((4000 - 6000) *10) / 4000 = 0
			   Memory Score: ((10000 - 5000) *10) / 10000 = 5
			   Node2 Score: (0 + 0) / 2 = 0
			*/
			pod: &v1.Pod{Spec: noResources},
			nodes: []*v1.Node{
				makeNodeWithAllResources("machine1", 0, 0, 0, map[v1.ResourceName]int64{NvidiaGpu: 0}),
				makeNodeWithAllResources("machine2", 0, 0, 0, map[v1.ResourceName]int64{NvidiaGpu: 0}),
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "zero node resources, pods scheduled with resources",
			pods: []*v1.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemoryAndGpu},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(test.pods, test.nodes)
			list, err := priorityFunction(LeastExtendedRequestedPriorityMap, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}
