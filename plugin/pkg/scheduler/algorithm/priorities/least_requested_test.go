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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func TestLeastRequested(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	machine1Spec := api.PodSpec{
		NodeName: "machine1",
	}
	machine2Spec := api.PodSpec{
		NodeName: "machine2",
	}
	noResources := api.PodSpec{
		Containers: []api.Container{},
	}
	cpuOnly := api.PodSpec{
		NodeName: "machine1",
		Containers: []api.Container{
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu":    resource.MustParse("1000m"),
						"memory": resource.MustParse("0"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu":    resource.MustParse("2000m"),
						"memory": resource.MustParse("0"),
					},
				},
			},
		},
	}
	cpuOnly2 := cpuOnly
	cpuOnly2.NodeName = "machine2"
	cpuAndMemory := api.PodSpec{
		NodeName: "machine2",
		Containers: []api.Container{
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu":    resource.MustParse("1000m"),
						"memory": resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu":    resource.MustParse("2000m"),
						"memory": resource.MustParse("3000"),
					},
				},
			},
		},
	}
	tests := []struct {
		pod          *api.Pod
		pods         []*api.Pod
		nodes        []*api.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			/*
				Node1 scores (remaining resources) on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Node1 Score: (10 + 10) / 2 = 10

				Node2 scores (remaining resources) on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Node2 Score: (10 + 10) / 2 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((4000 - 3000) *10) / 4000 = 2.5
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Node1 Score: (2.5 + 5) / 2 = 3

				Node2 scores on 0-10 scale
				CPU Score: ((6000 - 3000) *10) / 6000 = 5
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Node2 Score: (5 + 5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 3}, {Host: "machine2", Score: 5}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Node1 Score: (10 + 10) / 2 = 10

				Node2 scores on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Node2 Score: (10 + 10) / 2 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "no resources requested, pods scheduled",
			pods: []*api.Pod{
				{Spec: machine1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: machine1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: machine2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 0) *10) / 20000 = 10
				Node1 Score: (4 + 10) / 2 = 7

				Node2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Node2 Score: (4 + 7.5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 7}, {Host: "machine2", Score: 5}},
			test:         "no resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly2, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemory, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Node1 Score: (4 + 7.5) / 2 = 5

				Node2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 10000) *10) / 20000 = 5
				Node2 Score: (4 + 5) / 2 = 4
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 4}},
			test:         "resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Node1 Score: (4 + 7.5) / 2 = 5

				Node2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((50000 - 10000) *10) / 50000 = 8
				Node2 Score: (4 + 8) / 2 = 6
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 50000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 6}},
			test:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: ((4000 - 6000) *10) / 4000 = 0
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Node1 Score: (0 + 10) / 2 = 5

				Node2 scores on 0-10 scale
				CPU Score: ((4000 - 6000) *10) / 4000 = 0
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Node2 Score: (0 + 5) / 2 = 2
			*/
			pod:          &api.Pod{Spec: cpuOnly},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 2}},
			test:         "requested resources exceed node capacity",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 0, 0), makeNode("machine2", 0, 0)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			test:         "zero node resources, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
		list, err := priorityFunction(LeastRequestedPriorityMap, nil)(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}
