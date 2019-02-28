/*
Copyright 2018 The Kubernetes Authors.

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
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

func TestResourceBinPackingSingle(t *testing.T) {
	scarceResource := "intel.com/foo"
	scarceResources1 := map[string]int64{
		"intel.com/foo": 4,
	}

	scarceResources2 := map[string]int64{
		"intel.com/foo": 8,
	}

	resources := []Resources{
		{Resource: v1.ResourceName("intel.com/foo")},
	}

	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	scarceResourcePod1 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource): resource.MustParse("2"),
					},
				},
			},
		},
	}
	scarceResourcePod2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource): resource.MustParse("4"),
					},
				},
			},
		},
	}
	machine2Pod := scarceResourcePod1
	machine2Pod.NodeName = "machine2"
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			/*

				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				Node1 Score: { (0 + 0) / 4 } * 10 = 0

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				Node2 Score: { (0 + 0) / 8 } * 10 = 0
			*/
			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				Node1 Score: { (0 + 2) / 8 } * 10 = 3

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				Node2 Score: { (0 + 2) / 4 } * 10 = 5
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 3}, {Host: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				Node1 Score: { (0 + 2) / 8 } * 10 = 3

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				Node2 Score: { (2 + 2) / 4 } * 10 = 10
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 3}, {Host: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				Node1 Score: { (0 + 4) / 8 } * 10 = 5

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				Node2 Score: { (0 + 4) / 4 } * 10 = 10
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod2},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 10}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			//print(makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1))
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
			prior, _ := NewResourceBinPacking(resources)
			list, err := priorityFunction(prior, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}

func TestResourceBinPackingMultiple(t *testing.T) {
	scarceResource := "intel.com/foo"
	scarceResources1 := map[string]int64{
		"intel.com/foo": 4,
	}

	scarceResources2 := map[string]int64{
		"intel.com/foo": 8,
	}
	resources := []Resources{
		{Resource: v1.ResourceName("cpu"), Weight: 2},
		{Resource: v1.ResourceName("memory"), Weight: 1},
		{Resource: v1.ResourceName("intel.com/foo"), Weight: 5},
	}

	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	scarceResourcePod1 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource): resource.MustParse("2"),
					},
				},
			},
		},
	}
	scarceResourcePod2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource): resource.MustParse("4"),
					},
				},
			},
		},
	}
	machine2Pod := scarceResourcePod1
	machine2Pod.NodeName = "machine2"
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			/*
				    resources["cpu"] = 2
					resources["memory"] = 1
					resources["intel.com/foo"] = 5
					Node1 scores (used resources) on 0-10 scale
					used + requested / available
					CPU Score: { (0 + 0) / 4000 } * 10 = 0
					Meomory Score: { (0 + 0) / 10000 } * 10 = 0
					intel.com/foo Score: { (0 + 0) / 8 } * 10 = 0
					Node1 Score: (0 * 2) + (0 * 1) + (0 * 5)/ 8 = 0

					resources["cpu"] = 2
					resources["memory"] = 1
					resources["intel.com/foo"] = 5
					Node1 scores (used resources) on 0-10 scale
					used + requested / available
					CPU Score: { (0 + 0) / 4000 } * 10 = 0
					Meomory Score: { (0 + 0) / 10000 } * 10 = 0
					intel.com/foo Score: { (0 + 0) / 4 } * 10 = 0
					Node1 Score: (0 * 2) + (0 * 1) + (0 * 5)/ 8 = 0
			*/

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (0.25 * 5)/ 8 = 2

				Node1 Score: { (0 + 2) / 8 } * 10 = 2

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (0.5 * 5)/ 8 = 3
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 2}, {Host: "machine2", Score: 3}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (0.25 * 5)/ 8 = 2

				Node1 Score: { (0 + 2) / 8 } * 10 = 2

				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (1 * 5)/ 8 = 6
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 2}, {Host: "machine2", Score: 6}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{
			/*
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (0.5 * 5)/ 8 = 3



				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				CPU Score: { (0 + 0) / 4000 } * 10 = 0
				Meomory Score: { (0 + 0) / 10000 } * 10 = 0
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				Node1 Score: (0.025 * 2) + (0.02 * 1) + (1 * 5)/ 8 = 6
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod2},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 3}, {Host: "machine2", Score: 6}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
			prior, _ := NewResourceBinPacking(resources)
			list, err := priorityFunction(prior, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}

func TestResourceBinPackingMultipleExtended(t *testing.T) {
	scarceResource1 := "intel.com/foo"
	scarceResource2 := "intel.com/bar"
	scarceResources1 := map[string]int64{
		"intel.com/foo": 4,
		"intel.com/bar": 8,
	}

	scarceResources2 := map[string]int64{
		"intel.com/foo": 8,
		"intel.com/bar": 4,
	}

	resources := []Resources{
		{Resource: v1.ResourceName("intel.com/foo"), Weight: 3},
		{Resource: v1.ResourceName("intel.com/bar"), Weight: 5},
	}

	noResources := v1.PodSpec{
		Containers: []v1.Container{},
	}
	scarceResourcePod1 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource1): resource.MustParse("2"),
						v1.ResourceName(scarceResource2): resource.MustParse("2"),
					},
				},
			},
		},
	}
	scarceResourcePod2 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(scarceResource1): resource.MustParse("4"),
						v1.ResourceName(scarceResource2): resource.MustParse("2"),
					},
				},
			},
		},
	}
	machine2Pod := scarceResourcePod1
	machine2Pod.NodeName = "machine2"
	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		name         string
	}{
		{
			/*
				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 0) / 8 } * 10 = 0
				intel.com/bar Score: { (0 + 0) / 4 } * 10 = 0
				Node1 Score: (0 * 2) + (0 * 5) / 8 = 0

				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 0) / 4 } * 10 = 0
				intel.com/bar Score: { (0 + 0) / 8 } * 10 = 0
				Node2 Score: (0 * 3) + (0 * 5) / 8 = 0
			*/

			pod:          &v1.Pod{Spec: noResources},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			name:         "nothing scheduled, nothing requested",
		},

		{
			/*
				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				intel.com/bar Score: { (0 + 2) / 4 } * 10 = 0
				Node1 Score: (0.25 * 3) + (0.5 * 5) / 8 = 4

				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 2) / 4 } * 10 = 0
				intel.com/bar Score: { (0 + 2) / 8 } * 10 = 0
				Node2 Score: (0.5 * 3) + (0.25 * 5) / 8 = 3
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 3}},
			name:         "resources requested, pods scheduled with less resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},

		{
			/*
				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 2) / 8 } * 10 = 0
				intel.com/bar Score: { (0 + 2) / 4 } * 10 = 0
				Node1 Score: (0.25 * 3) + (0.5 * 5) / 8 = 4

				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (2 + 2) / 4 } * 10 = 0
				intel.com/bar Score: { (2 + 2) / 8 } * 10 = 0
				Node2 Score: (1 * 3) + (0.5 * 5) / 8 = 7
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod1},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 7}},
			name:         "resources requested, pods scheduled with resources, on node with existing pod running ",
			pods: []*v1.Pod{
				{Spec: machine2Pod},
			},
		},

		{
			/*
				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node1 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 4) / 8 } * 10 = 0
				intel.com/bar Score: { (0 + 2) / 4 } * 10 = 0
				Node1 Score: (0.25 * 3) + (0.5 * 5) / 8 = 5

				resources["intel.com/foo"] = 3
				resources["intel.com/bar"] = 5
				Node2 scores (used resources) on 0-10 scale
				used + requested / available
				intel.com/foo Score: { (0 + 4) / 4 } * 10 = 0
				intel.com/bar Score: { (0 + 2) / 8 } * 10 = 0
				Node2 Score: (1 * 3) + (0.25 * 5) / 8 = 5
			*/
			pod:          &v1.Pod{Spec: scarceResourcePod2},
			nodes:        []*v1.Node{makeNodeScarceResource("machine1", 4000, 10000*1024*1024, scarceResources2), makeNodeScarceResource("machine2", 4000, 10000*1024*1024, scarceResources1)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 5}},
			name:         "resources requested, pods scheduled with more resources",
			pods: []*v1.Pod{
				{Spec: noResources},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
			prior, _ := NewResourceBinPacking(resources)
			list, err := priorityFunction(prior, nil, nil)(test.pod, nodeNameToInfo, test.nodes)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expectedList, list) {
				t.Errorf("expected %#v, got %#v", test.expectedList, list)
			}
		})
	}
}
