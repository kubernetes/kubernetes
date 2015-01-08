/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
)

func makeMinion(node string, milliCPU, memory int64) api.Node {
	return api.Node{
		ObjectMeta: api.ObjectMeta{Name: node},
		Spec: api.NodeSpec{
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
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
	machine1Status := api.PodStatus{
		Host: "machine1",
	}
	machine2Status := api.PodStatus{
		Host: "machine2",
	}
	noResources := api.PodSpec{
		Containers: []api.Container{},
	}
	cpuOnly := api.PodSpec{
		Containers: []api.Container{
			{CPU: resource.MustParse("1000m")},
			{CPU: resource.MustParse("2000m")},
		},
	}
	cpuAndMemory := api.PodSpec{
		Containers: []api.Container{
			{CPU: resource.MustParse("1000m"), Memory: resource.MustParse("2000")},
			{CPU: resource.MustParse("2000m"), Memory: resource.MustParse("3000")},
		},
	}
	tests := []struct {
		pod          api.Pod
		pods         []api.Pod
		nodes        []api.Node
		expectedList HostPriorityList
		test         string
	}{
		{
			/*
				Minion1 scores (remaining resources) on 0-10 scale
				CPU Score: (4000 - 0) / 4000 = 10
				Memory Score: (10000 - 0) / 10000 = 10
				Minion1 Score: (10 + 10) / 2 = 10

				Minion2 scores (remaining resources) on 0-10 scale
				CPU Score: (4000 - 0) / 4000 = 10
				Memory Score: (10000 - 0) / 10000 = 10
				Minion2 Score: (10 + 10) / 2 = 10
			*/
			pod:          api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (4000 - 3000) / 4000 = 2.5
				Memory Score: (10000 - 5000) / 10000 = 5
				Minion1 Score: (2.5 + 5) / 2 = 3

				Minion2 scores on 0-10 scale
				CPU Score: (6000 - 3000) / 6000 = 5
				Memory Score: (10000 - 5000) / 10000 = 5
				Minion2 Score: (5 + 5) / 2 = 5
			*/
			pod:          api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 6000, 10000)},
			expectedList: []HostPriority{{"machine1", 3}, {"machine2", 5}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (4000 - 0) / 4000 = 10
				Memory Score: (10000 - 0) / 10000 = 10
				Minion1 Score: (10 + 10) / 2 = 10

				Minion2 scores on 0-10 scale
				CPU Score: (4000 - 0) / 4000 = 10
				Memory Score: (10000 - 0) / 10000 = 10
				Minion2 Score: (10 + 10) / 2 = 10
			*/
			pod:          api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "no resources requested, pods scheduled",
			pods: []api.Pod{
				{Status: machine1Status, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Status: machine1Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Status: machine2Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Status: machine2Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (20000 - 0) / 20000 = 10
				Minion1 Score: (4 + 10) / 2 = 7

				Minion2 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (20000 - 5000) / 20000 = 7.5
				Minion2 Score: (4 + 7.5) / 2 = 5
			*/
			pod:          api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []HostPriority{{"machine1", 7}, {"machine2", 5}},
			test:         "no resources requested, pods scheduled with resources",
			pods: []api.Pod{
				{Spec: cpuOnly, Status: machine1Status, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: cpuOnly, Status: machine1Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: cpuOnly, Status: machine2Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: cpuAndMemory, Status: machine2Status, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (20000 - 5000) / 20000 = 7.5
				Minion1 Score: (4 + 7.5) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (20000 - 10000) / 20000 = 5
				Minion2 Score: (4 + 5) / 2 = 4
			*/
			pod:          api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 20000)},
			expectedList: []HostPriority{{"machine1", 5}, {"machine2", 4}},
			test:         "resources requested, pods scheduled with resources",
			pods: []api.Pod{
				{Spec: cpuOnly, Status: machine1Status},
				{Spec: cpuAndMemory, Status: machine2Status},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (20000 - 5000) / 20000 = 7.5
				Minion1 Score: (4 + 7.5) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: (10000 - 6000) / 10000 = 4
				Memory Score: (50000 - 10000) / 50000 = 8
				Minion2 Score: (4 + 8) / 2 = 6
			*/
			pod:          api.Pod{Spec: cpuAndMemory},
			nodes:        []api.Node{makeMinion("machine1", 10000, 20000), makeMinion("machine2", 10000, 50000)},
			expectedList: []HostPriority{{"machine1", 5}, {"machine2", 6}},
			test:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []api.Pod{
				{Spec: cpuOnly, Status: machine1Status},
				{Spec: cpuAndMemory, Status: machine2Status},
			},
		},
		{
			/*
				Minion1 scores on 0-10 scale
				CPU Score: (4000 - 6000) / 4000 = 0
				Memory Score: (10000 - 0) / 10000 = 10
				Minion1 Score: (0 + 10) / 2 = 5

				Minion2 scores on 0-10 scale
				CPU Score: (4000 - 6000) / 4000 = 0
				Memory Score: (10000 - 5000) / 10000 = 5
				Minion2 Score: (0 + 5) / 2 = 2
			*/
			pod:          api.Pod{Spec: cpuOnly},
			nodes:        []api.Node{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 5}, {"machine2", 2}},
			test:         "requested resources exceed minion capacity",
			pods: []api.Pod{
				{Spec: cpuOnly, Status: machine1Status},
				{Spec: cpuAndMemory, Status: machine2Status},
			},
		},
		{
			pod:          api.Pod{Spec: noResources},
			nodes:        []api.Node{makeMinion("machine1", 0, 0), makeMinion("machine2", 0, 0)},
			expectedList: []HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "zero minion resources, pods scheduled with resources",
			pods: []api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		list, err := LeastRequestedPriority(test.pod, FakePodLister(test.pods), FakeMinionLister(api.NodeList{Items: test.nodes}))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}
