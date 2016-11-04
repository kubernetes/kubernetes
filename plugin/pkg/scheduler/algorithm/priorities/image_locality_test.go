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
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

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
		list, err := priorityFunction(ImageLocalityPriorityMap, nil)(test.pod, nodeNameToInfo, test.nodes)
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
