/*
Copyright 2014 The Kubernetes Authors.

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
	"os/exec"
	"reflect"
	"sort"
	"strconv"
	"testing"

	"k8s.io/kubernetes/cmd/libs/go2idl/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/codeinspector"
	"k8s.io/kubernetes/plugin/pkg/scheduler"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func makeNode(node string, milliCPU, memory int64) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: node},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
			Allocatable: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

func TestZeroRequest(t *testing.T) {
	// A pod with no resources. We expect spreading to count it as having the default resources.
	noResources := api.PodSpec{
		Containers: []api.Container{
			{},
		},
	}
	noResources1 := noResources
	noResources1.NodeName = "machine1"
	// A pod with the same resources as a 0-request pod gets by default as its resources (for spreading).
	small := api.PodSpec{
		Containers: []api.Container{
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu": resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCpuRequest, 10) + "m"),
						"memory": resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest, 10)),
					},
				},
			},
		},
	}
	small2 := small
	small2.NodeName = "machine2"
	// A larger pod.
	large := api.PodSpec{
		Containers: []api.Container{
			{
				Resources: api.ResourceRequirements{
					Requests: api.ResourceList{
						"cpu": resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMilliCpuRequest*3, 10) + "m"),
						"memory": resource.MustParse(
							strconv.FormatInt(priorityutil.DefaultMemoryRequest*3, 10)),
					},
				},
			},
		},
	}
	large1 := large
	large1.NodeName = "machine1"
	large2 := large
	large2.NodeName = "machine2"
	tests := []struct {
		pod   *api.Pod
		pods  []*api.Pod
		nodes []*api.Node
		test  string
	}{
		// The point of these next two tests is to show you get the same priority for a zero-request pod
		// as for a pod with the defaults requests, both when the zero-request pod is already on the machine
		// and when the zero-request pod is the one being scheduled.
		{
			pod:   &api.Pod{Spec: noResources},
			nodes: []*api.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of zero-request pod with machine with zero-request pod",
			pods: []*api.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
		{
			pod:   &api.Pod{Spec: small},
			nodes: []*api.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of nonzero-request pod with machine with zero-request pod",
			pods: []*api.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
		// The point of this test is to verify that we're not just getting the same score no matter what we schedule.
		{
			pod:   &api.Pod{Spec: large},
			nodes: []*api.Node{makeNode("machine1", 1000, priorityutil.DefaultMemoryRequest*10), makeNode("machine2", 1000, priorityutil.DefaultMemoryRequest*10)},
			test:  "test priority of larger pod with machine with zero-request pod",
			pods: []*api.Pod{
				{Spec: large1}, {Spec: noResources1},
				{Spec: large2}, {Spec: small2},
			},
		},
	}

	const expectedPriority int = 25
	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
		list, err := scheduler.PrioritizeNodes(
			test.pod,
			nodeNameToInfo,
			// This should match the configuration in defaultPriorities() in
			// plugin/pkg/scheduler/algorithmprovider/defaults/defaults.go if you want
			// to test what's actually in production.
			[]algorithm.PriorityConfig{
				{Function: LeastRequestedPriority, Weight: 1},
				{Function: BalancedResourceAllocation, Weight: 1},
				{
					Function: NewSelectorSpreadPriority(
						algorithm.FakePodLister(test.pods),
						algorithm.FakeServiceLister([]api.Service{}),
						algorithm.FakeControllerLister([]api.ReplicationController{}),
						algorithm.FakeReplicaSetLister([]extensions.ReplicaSet{})),
					Weight: 1,
				},
			},
			algorithm.FakeNodeLister(test.nodes), []algorithm.SchedulerExtender{})
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		for _, hp := range list {
			if test.test == "test priority of larger pod with machine with zero-request pod" {
				if hp.Score == expectedPriority {
					t.Errorf("%s: expected non-%d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			} else {
				if hp.Score != expectedPriority {
					t.Errorf("%s: expected %d for all priorities, got list %#v", test.test, expectedPriority, list)
				}
			}
		}
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
		list, err := LeastRequestedPriority(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestMostRequested(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
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
				Node1 scores (used resources) on 0-10 scale
				CPU Score: (0 * 10  / 4000 = 0
				Memory Score: (0 * 10) / 10000 = 0
				Node1 Score: (0 + 0) / 2 = 0

				Node2 scores (used resources) on 0-10 scale
				CPU Score: (0 * 10 / 4000 = 0
				Memory Score: (0 * 10 / 10000 = 0
				Node2 Score: (0 + 0) / 2 = 0
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: (3000 * 10 / 4000 = 7.5
				Memory Score: (5000 * 10) / 10000 = 5
				Node1 Score: (7.5 + 5) / 2 = 6

				Node2 scores on 0-10 scale
				CPU Score: (3000 * 10 / 6000 = 5
				Memory Score: (5000 * 10 / 10000 = 5
				Node2 Score: (5 + 5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []schedulerapi.HostPriority{{"machine1", 6}, {"machine2", 5}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Score: (6000 * 10) / 10000 = 6
				Memory Score: (0 * 10) / 20000 = 10
				Node1 Score: (6 + 0) / 2 = 3

				Node2 scores on 0-10 scale
				CPU Score: (6000 * 10) / 10000 = 6
				Memory Score: (5000 * 10) / 20000 = 2.5
				Node2 Score: (6 + 2.5) / 2 = 4
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{"machine1", 3}, {"machine2", 4}},
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
				CPU Score: (6000 * 10) / 10000 = 6
				Memory Score: (5000 * 10) / 20000 = 2.5
				Node1 Score: (6 + 2.5) / 2 = 4

				Node2 scores on 0-10 scale
				CPU Score: (6000 * 10) / 10000 = 6
				Memory Score: (10000 * 10) / 20000 = 5
				Node2 Score: (6 + 5) / 2 = 5
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{"machine1", 4}, {"machine2", 5}},
			test:         "resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
		list, err := MostRequestedPriority(test.pod, nodeNameToInfo, test.nodes)
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
		nodes        []*api.Node
		label        string
		presence     bool
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			label:        "baz",
			presence:     true,
			test:         "no match found, presence true",
		},
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "baz",
			presence:     false,
			test:         "no match found, presence false",
		},
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			label:        "foo",
			presence:     true,
			test:         "one match found, presence true",
		},
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "foo",
			presence:     false,
			test:         "one match found, presence false",
		},
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			label:        "bar",
			presence:     true,
			test:         "two matches found, presence true",
		},
		{
			nodes: []*api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
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
		list, err := prioritizer.CalculateNodeLabelPriority(nil, map[string]*schedulercache.NodeInfo{}, test.nodes)
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
				CPU Fraction: 0 / 4000 = 0%
				Memory Fraction: 0 / 10000 = 0%
				Node1 Score: 10 - (0-0)*10 = 10

				Node2 scores (remaining resources) on 0-10 scale
				CPU Fraction: 0 / 4000 = 0 %
				Memory Fraction: 0 / 10000 = 0%
				Node2 Score: 10 - (0-0)*10 = 10
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "nothing scheduled, nothing requested",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 3000 / 4000= 75%
				Memory Fraction: 5000 / 10000 = 50%
				Node1 Score: 10 - (0.75-0.5)*10 = 7

				Node2 scores on 0-10 scale
				CPU Fraction: 3000 / 6000= 50%
				Memory Fraction: 5000/10000 = 50%
				Node2 Score: 10 - (0.5-0.5)*10 = 10
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 6000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 7}, {Host: "machine2", Score: 10}},
			test:         "nothing scheduled, resources requested, differently sized machines",
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Node1 Score: 10 - (0-0)*10 = 10

				Node2 scores on 0-10 scale
				CPU Fraction: 0 / 4000= 0%
				Memory Fraction: 0 / 10000 = 0%
				Node2 Score: 10 - (0-0)*10 = 10
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
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 0 / 20000 = 0%
				Node1 Score: 10 - (0.6-0)*10 = 4

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node2 Score: 10 - (0.6-0.25)*10 = 6
			*/
			pod:          &api.Pod{Spec: noResources},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 4}, {Host: "machine2", Score: 6}},
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
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node1 Score: 10 - (0.6-0.25)*10 = 6

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 20000 = 50%
				Node2 Score: 10 - (0.6-0.5)*10 = 9
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 20000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 9}},
			test:         "resources requested, pods scheduled with resources",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 5000 / 20000 = 25%
				Node1 Score: 10 - (0.6-0.25)*10 = 6

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 10000 = 60%
				Memory Fraction: 10000 / 50000 = 20%
				Node2 Score: 10 - (0.6-0.2)*10 = 6
			*/
			pod:          &api.Pod{Spec: cpuAndMemory},
			nodes:        []*api.Node{makeNode("machine1", 10000, 20000), makeNode("machine2", 10000, 50000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 6}, {Host: "machine2", Score: 6}},
			test:         "resources requested, pods scheduled with resources, differently sized machines",
			pods: []*api.Pod{
				{Spec: cpuOnly},
				{Spec: cpuAndMemory},
			},
		},
		{
			/*
				Node1 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction: 0 / 10000 = 0
				Node1 Score: 0

				Node2 scores on 0-10 scale
				CPU Fraction: 6000 / 4000 > 100% ==> Score := 0
				Memory Fraction 5000 / 10000 = 50%
				Node2 Score: 0
			*/
			pod:          &api.Pod{Spec: cpuOnly},
			nodes:        []*api.Node{makeNode("machine1", 4000, 10000), makeNode("machine2", 4000, 10000)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
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
		list, err := BalancedResourceAllocation(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestImageLocalityPriority(t *testing.T) {
	test_40_250 := api.PodSpec{
		Containers: []api.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/250",
			},
		},
	}

	test_40_140 := api.PodSpec{
		Containers: []api.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/140",
			},
		},
	}

	test_min_max := api.PodSpec{
		Containers: []api.Container{
			{
				Image: "gcr.io/10",
			},
			{
				Image: "gcr.io/2000",
			},
		},
	}

	node_40_140_2000 := api.NodeStatus{
		Images: []api.ContainerImage{
			{
				Names: []string{
					"gcr.io/40",
					"gcr.io/40:v1",
					"gcr.io/40:v1",
				},
				SizeBytes: int64(40 * mb),
			},
			{
				Names: []string{
					"gcr.io/140",
					"gcr.io/140:v1",
				},
				SizeBytes: int64(140 * mb),
			},
			{
				Names: []string{
					"gcr.io/2000",
				},
				SizeBytes: int64(2000 * mb),
			},
		},
	}

	node_250_10 := api.NodeStatus{
		Images: []api.ContainerImage{
			{
				Names: []string{
					"gcr.io/250",
				},
				SizeBytes: int64(250 * mb),
			},
			{
				Names: []string{
					"gcr.io/10",
					"gcr.io/10:v1",
				},
				SizeBytes: int64(10 * mb),
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
			// Pod: gcr.io/40 gcr.io/250

			// Node1
			// Image: gcr.io/40 40MB
			// Score: (40M-23M)/97.7M + 1 = 1

			// Node2
			// Image: gcr.io/250 250MB
			// Score: (250M-23M)/97.7M + 1 = 3
			pod:          &api.Pod{Spec: test_40_250},
			nodes:        []*api.Node{makeImageNode("machine1", node_40_140_2000), makeImageNode("machine2", node_250_10)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 1}, {Host: "machine2", Score: 3}},
			test:         "two images spread on two nodes, prefer the larger image one",
		},
		{
			// Pod: gcr.io/40 gcr.io/140

			// Node1
			// Image: gcr.io/40 40MB, gcr.io/140 140MB
			// Score: (40M+140M-23M)/97.7M + 1 = 2

			// Node2
			// Image: not present
			// Score: 0
			pod:          &api.Pod{Spec: test_40_140},
			nodes:        []*api.Node{makeImageNode("machine1", node_40_140_2000), makeImageNode("machine2", node_250_10)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 2}, {Host: "machine2", Score: 0}},
			test:         "two images on one node, prefer this node",
		},
		{
			// Pod: gcr.io/2000 gcr.io/10

			// Node1
			// Image: gcr.io/2000 2000MB
			// Score: 2000 > max score = 10

			// Node2
			// Image: gcr.io/10 10MB
			// Score: 10 < min score = 0
			pod:          &api.Pod{Spec: test_min_max},
			nodes:        []*api.Node{makeImageNode("machine1", node_40_140_2000), makeImageNode("machine2", node_250_10)},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}},
			test:         "if exceed limit, use limit",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, test.nodes)
		list, err := ImageLocalityPriority(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}

		sort.Sort(test.expectedList)
		sort.Sort(list)

		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func makeImageNode(node string, status api.NodeStatus) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: node},
		Status:     status,
	}
}

func TestPrioritiesRegistered(t *testing.T) {
	var functions []*types.Type

	// Files and directories which priorities may be referenced
	targetFiles := []string{
		"./../../algorithmprovider/defaults/defaults.go", // Default algorithm
		"./../../factory/plugins.go",                     // Registered in init()
	}

	// List all golang source files under ./priorities/, excluding test files and sub-directories.
	files, err := codeinspector.GetSourceCodeFiles(".")

	if err != nil {
		t.Errorf("unexpected error: %v when listing files in current directory", err)
	}

	// Get all public priorities in files.
	for _, filePath := range files {
		fileFunctions, err := codeinspector.GetPublicFunctions("k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities", filePath)
		if err == nil {
			functions = append(functions, fileFunctions...)
		} else {
			t.Errorf("unexpected error when parsing %s", filePath)
		}
	}

	// Check if all public priorities are referenced in target files.
	for _, function := range functions {
		args := []string{"-rl", function.Name.Name}
		args = append(args, targetFiles...)

		err := exec.Command("grep", args...).Run()
		if err != nil {
			switch err.Error() {
			case "exit status 2":
				t.Errorf("unexpected error when checking %s", function.Name)
			case "exit status 1":
				t.Errorf("priority %s is implemented as public but seems not registered or used in any other place",
					function.Name)
			}
		}
	}
}

func TestNodePreferAvoidPriority(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"bar": "foo"}
	annotations1 := map[string]string{
		api.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
							                    "uid": "abcdef123456",
							                    "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
	}
	annotations2 := map[string]string{
		api.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicaSet",
							                    "name": "foo",
							                    "uid": "qwert12345",
							                    "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
	}
	testNodes := []*api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: "machine1", Annotations: annotations1},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "machine2", Annotations: annotations2},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "machine3"},
		},
	}
	tests := []struct {
		pod          *api.Pod
		rcs          []api.ReplicationController
		rss          []extensions.ReplicaSet
		nodes        []*api.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "default", Labels: label1}},
			rcs: []api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{
						Namespace: "default",
						Name:      "foo",
						UID:       "abcdef123456",
					},
					Spec: api.ReplicationControllerSpec{Selector: label1},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			test:         "pod managed by ReplicationController should avoid a node, this node get lowest priority score",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "default", Labels: label2}},
			rss: []extensions.ReplicaSet{
				{
					TypeMeta: unversioned.TypeMeta{
						APIVersion: "v1",
						Kind:       "ReplicaSet",
					},
					ObjectMeta: api.ObjectMeta{
						Namespace: "default",
						Name:      "bar",
						UID:       "qwert12345",
					},
					Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: label2}},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 10}},
			test:         "pod managed by ReplicaSet should avoid a node, this node get lowest priority score",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Namespace: "default"}},
			rcs: []api.ReplicationController{
				{
					ObjectMeta: api.ObjectMeta{
						Namespace: "default",
						Name:      "foo",
						UID:       "abcdef123456",
					},
					Spec: api.ReplicationControllerSpec{Selector: label1},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			test:         "pod should not avoid these nodes, all nodes get highest priority score",
		},
	}

	for _, test := range tests {
		prioritizer := NodePreferAvoidPod{
			controllerLister: algorithm.FakeControllerLister(test.rcs),
			replicaSetLister: algorithm.FakeReplicaSetLister(test.rss),
		}
		list, err := prioritizer.CalculateNodePreferAvoidPodsPriority(test.pod, map[string]*schedulercache.NodeInfo{}, test.nodes)
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
