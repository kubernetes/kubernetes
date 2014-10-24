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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resources"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func makeMinion(node string, cpu, memory int) api.Minion {
	return api.Minion{
		ObjectMeta: api.ObjectMeta{Name: node},
		NodeResources: api.NodeResources{
			Capacity: api.ResourceList{
				resources.CPU:    util.NewIntOrStringFromInt(cpu),
				resources.Memory: util.NewIntOrStringFromInt(memory),
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
	machine1State := api.PodState{
		Host: "machine1",
	}
	machine2State := api.PodState{
		Host: "machine2",
	}
	cpuOnly := api.PodState{
		Manifest: api.ContainerManifest{
			Containers: []api.Container{
				{CPU: 1000},
				{CPU: 2000},
			},
		},
		Host: "machine1",
	}
	cpuAndMemory := api.PodState{
		Manifest: api.ContainerManifest{
			Containers: []api.Container{
				{CPU: 1000, Memory: 2000},
				{CPU: 2000, Memory: 3000},
			},
		},
		Host: "machine2",
	}
	tests := []struct {
		pod          api.Pod
		pods         []api.Pod
		nodes        []api.Minion
		expectedList HostPriorityList
		test         string
	}{
		{
			nodes:        []api.Minion{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "nothing scheduled",
		},
		{
			nodes:        []api.Minion{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "no resources requested",
			pods: []api.Pod{
				{DesiredState: machine1State, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{DesiredState: machine1State, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{DesiredState: machine2State, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{DesiredState: machine2State, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
		},
		{
			nodes:        []api.Minion{makeMinion("machine1", 4000, 10000), makeMinion("machine2", 4000, 10000)},
			expectedList: []HostPriority{{"machine1", 37 /* int(75% / 2) */}, {"machine2", 62 /* int( 75% + 50% / 2) */}},
			test:         "no resources requested",
			pods: []api.Pod{
				{DesiredState: cpuOnly},
				{DesiredState: cpuAndMemory},
			},
		},
		{
			nodes:        []api.Minion{makeMinion("machine1", 0, 0), makeMinion("machine2", 0, 0)},
			expectedList: []HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "zero minion resources",
			pods: []api.Pod{
				{DesiredState: cpuOnly},
				{DesiredState: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		list, err := LeastRequestedPriority(test.pod, FakePodLister(test.pods), FakeMinionLister(api.MinionList{Items: test.nodes}))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}
