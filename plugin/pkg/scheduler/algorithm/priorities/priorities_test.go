/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"sort"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm"
)

func makeMinion(node string, milliCPU, memory int64) api.Node {
	return api.Node{
		ObjectMeta: api.ObjectMeta{Name: node},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

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
					Limits: api.ResourceList{
						"cpu": resource.MustParse("1000m"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
						"cpu": resource.MustParse("2000m"),
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
					Limits: api.ResourceList{
						"cpu":    resource.MustParse("1000m"),
						"memory": resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
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
		nodes        []api.Node
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			/*
				Minion1 scores (remaining resources) on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Minion1 Score: (10 + 10) / 2 = 10

				Minion2 scores (remaining resources) on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Minion2 Score: (10 + 10) / 2 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: ((4000 - 3000) *10) / 4000 = 2.5
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Minion1 Score: (2.5 + 5) / 2 = 3

				Minion2 scores on 0-10 scale
				CPU Score: ((6000 - 3000) *10) / 6000 = 5
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Minion2 Score: (5 + 5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 6000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 3}, {"machine2", 5}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Minion1 Score: (10 + 10) / 2 = 10

				Minion2 scores on 0-10 scale
				CPU Score: ((4000 - 0) *10) / 4000 = 10
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Minion2 Score: (10 + 10) / 2 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
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
				Minion1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 0) *10) / 20000 = 10
				Minion1 Score: (4 + 10) / 2 = 7

				Minion2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Minion2 Score: (4 + 7.5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []algorithm.HostPriority{{"machine1", 7}, {"machine2", 5}},
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
				Minion1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Minion1 Score: (4 + 7.5) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 10000) *10) / 20000 = 5
				Minion2 Score: (4 + 5) / 2 = 4
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []algorithm.HostPriority{{"machine1", 5}, {"machine2", 4}},
			test:         "resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((20000 - 5000) *10) / 20000 = 7.5
				Minion1 Score: (4 + 7.5) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: ((10000 - 6000) *10) / 10000 = 4
				Memory Score: ((50000 - 10000) *10) / 50000 = 8
				Minion2 Score: (4 + 8) / 2 = 6
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 50000)},
			expectedList: []algorithm.HostPriority{{"machine1", 5}, {"machine2", 6}},
			test:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: ((4000 - 6000) *10) / 4000 = 0
				Memory Score: ((10000 - 0) *10) / 10000 = 10
				Minion1 Score: (0 + 10) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: ((4000 - 6000) *10) / 4000 = 0
				Memory Score: ((10000 - 5000) *10) / 10000 = 5
				Minion2 Score: (0 + 5) / 2 = 2
			*/
			pod:          &api.Pod{Spec: cpuOnly},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 5}, {"machine2", 2}},
			test:         "requested resources exceed minion capacity",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 0, 0), makeMinion("machine2", 0, 0)},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "zero minion resources, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		list, err := LeastRequestedPriority(test.pod, algorithm.FakePodLister(test.pods), algorithm.FakeMinionLister(api.NodeList{Items: test.nodes}))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestNewNodeLabelPriority(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"bar": "foo"}
	label3 := map[string]string{"bar": "baz"}
	tests := []struct {
		nodes        []api.Node
		label        string
		presence     bool
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}, {"machine3", 0}},
			label:        "baz",
			presence:     true,
			test:         "no match found, presence true",
		},
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}, {"machine3", 10}},
			label:        "baz",
			presence:     false,
			test:         "no match found, presence false",
		},
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}, {"machine3", 0}},
			label:        "foo",
			presence:     true,
			test:         "one match found, presence true",
		},
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 10}, {"machine3", 10}},
			label:        "foo",
			presence:     false,
			test:         "one match found, presence false",
		},
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 10}, {"machine3", 10}},
			label:        "bar",
			presence:     true,
			test:         "two matches found, presence true",
		},
		{
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}, {"machine3", 0}},
			label:        "bar",
			presence:     false,
			test:         "two matches found, presence false",
		},
	}

	for _, test := range tests {
		prioritizer := NodeLabelPrioritizer{
			label:    test.label,
			presence: test.presence,
		}
		list, err := prioritizer.CalculateNodeLabelPriority(nil, nil, algorithm.FakeMinionLister(api.NodeList{Items: test.nodes}))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// sort the two lists to avoid failures on account of different ordering
		sort.Sort(test.expectedList)
		sort.Sort(list)
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestBalancedResourceAllocation(t *testing.T) {
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
					Limits: api.ResourceList{
						"cpu": resource.MustParse("1000m"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
						"cpu": resource.MustParse("2000m"),
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
					Limits: api.ResourceList{
						"cpu":    resource.MustParse("1000m"),
						"memory": resource.MustParse("2000"),
					},
				},
			},
			{
				Resources: api.ResourceRequirements{
					Limits: api.ResourceList{
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
		nodes        []api.Node
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			/*
				Minion1 scores (remaining resources) on 0-10 scale
				CPU Fraction: 0 / 4000 = 0%
				Memory Fraction: 0 / 10000 = 0%
				Minion1 Score: 10 - (0-0)*10 = 10

				Minion2 scores (remaining resources) on 0-10 scale
				CPU Fraction: 0 / 4000 = 0 %
				Memory Fraction: 0 / 10000 = 0%
				Minion2 Score: 10 - (0-0)*10 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Fraction: 3000 / 4000= 75%
				Memory Fraction: 5000 / 10000 = 50%
				Minion1 Score: 10 - (0.75-0.5)*10 = 7

				Minion2 scores on 0-10 scale
				CPU Fraction: 3000 / 6000= 50%
				Memory Fraction: 5000/10000 = 50%
				Minion2 Score: 10 - (0.5-0.5)*10 = 10
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 6000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 7}, {"machine2", 10}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Minion1 Score: 10 - (0-0)*10 = 10

				Minion2 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Minion2 Score: 10 - (0-0)*10 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
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
				Minion1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 0 / 20000 = 0%
				Minion1 Score: 10 - (0.6-0)*10 = 4

				Minion2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Minion2 Score: 10 - (0.6-0.25)*10 = 6
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []algorithm.HostPriority{{"machine1", 4}, {"machine2", 6}},
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
				Minion1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Minion1 Score: 10 - (0.6-0.25)*10 = 6

				Minion2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 20000 = 50%
				Minion2 Score: 10 - (0.6-0.5)*10 = 9
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []algorithm.HostPriority{{"machine1", 6}, {"machine2", 9}},
			test:         "resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Minion1 Score: 10 - (0.6-0.25)*10 = 6

				Minion2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 50000 = 20%
				Minion2 Score: 10 - (0.6-0.2)*10 = 6
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 50000)},
			expectedList: []algorithm.HostPriority{{"machine1", 6}, {"machine2", 6}},
			test:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction: 0 / 10000 = 0
				Minion1 Score: 0

				Minion2 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction 5000 / 10000 = 50%
				Minion2 Score: 0
			*/
			pod:          &api.Pod{Spec: cpuOnly},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "requested resources exceed minion capacity",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			pod:          &api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 0, 0), makeMinion("machine2", 0, 0)},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "zero minion resources, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		list, err := BalancedResourceAllocation(test.pod, algorithm.FakePodLister(test.pods), algorithm.FakeMinionLister(api.NodeList{Items: test.nodes}))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}
