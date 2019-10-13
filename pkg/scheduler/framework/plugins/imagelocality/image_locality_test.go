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

package imagelocality

import (
	"reflect"
	"testing"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/util/parsers"
)

func TestImageLocalityPriority(t *testing.T) {
	test40250 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/250",
			},
		},
	}

	test40300 := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/40",
			},
			{
				Image: "gcr.io/300",
			},
		},
	}

	testMinMax := v1.PodSpec{
		Containers: []v1.Container{
			{
				Image: "gcr.io/10",
			},
			{
				Image: "gcr.io/2000",
			},
		},
	}

	node403002000 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/40:" + parsers.DefaultImageTag,
					"gcr.io/40:v1",
					"gcr.io/40:v1",
				},
				SizeBytes: int64(40 * mb),
			},
			{
				Names: []string{
					"gcr.io/300:" + parsers.DefaultImageTag,
					"gcr.io/300:v1",
				},
				SizeBytes: int64(300 * mb),
			},
			{
				Names: []string{
					"gcr.io/2000:" + parsers.DefaultImageTag,
				},
				SizeBytes: int64(2000 * mb),
			},
		},
	}

	node25010 := v1.NodeStatus{
		Images: []v1.ContainerImage{
			{
				Names: []string{
					"gcr.io/250:" + parsers.DefaultImageTag,
				},
				SizeBytes: int64(250 * mb),
			},
			{
				Names: []string{
					"gcr.io/10:" + parsers.DefaultImageTag,
					"gcr.io/10:v1",
				},
				SizeBytes: int64(10 * mb),
			},
		},
	}

	nodeWithNoImages := v1.NodeStatus{}

	tests := []struct {
		pod          *v1.Pod
		pods         []*v1.Pod
		nodes        []*v1.Node
		expectedList framework.NodeScoreList
		name         string
	}{
		{
			// Pod: gcr.io/40 gcr.io/250

			// Node1
			// Image: gcr.io/40:latest 40MB
			// Score: 0 (40M/2 < 23M, min-threshold)

			// Node2
			// Image: gcr.io/250:latest 250MB
			// Score: 100 * (250M/2 - 23M)/(1000M - 23M) = 100
			pod:          &v1.Pod{Spec: test40250},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 0}, {Name: "machine2", Score: 10}},
			name:         "two images spread on two nodes, prefer the larger image one",
		},
		{
			// Pod: gcr.io/40 gcr.io/300

			// Node1
			// Image: gcr.io/40:latest 40MB, gcr.io/300:latest 300MB
			// Score: 100 * ((40M + 300M)/2 - 23M)/(1000M - 23M) = 15

			// Node2
			// Image: not present
			// Score: 0
			pod:          &v1.Pod{Spec: test40300},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 15}, {Name: "machine2", Score: 0}},
			name:         "two images on one node, prefer this node",
		},
		{
			// Pod: gcr.io/2000 gcr.io/10

			// Node1
			// Image: gcr.io/2000:latest 2000MB
			// Score: 100 (2000M/2 >= 1000M, max-threshold)

			// Node2
			// Image: gcr.io/10:latest 10MB
			// Score: 0 (10M/2 < 23M, min-threshold)
			pod:          &v1.Pod{Spec: testMinMax},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: framework.MaxNodeScore}, {Name: "machine2", Score: 0}},
			name:         "if exceed limit, use limit",
		},
		{
			// Pod: gcr.io/2000 gcr.io/10

			// Node1
			// Image: gcr.io/2000:latest 2000MB
			// Score: 100 * (2000M/3 - 23M)/(1000M - 23M) = 65

			// Node2
			// Image: gcr.io/10:latest 10MB
			// Score: 0 (10M/2 < 23M, min-threshold)

			// Node3
			// Image:
			// Score: 0
			pod:          &v1.Pod{Spec: testMinMax},
			nodes:        []*v1.Node{makeImageNode("machine1", node403002000), makeImageNode("machine2", node25010), makeImageNode("machine3", nodeWithNoImages)},
			expectedList: []framework.NodeScore{{Name: "machine1", Score: 65}, {Name: "machine2", Score: 0}, {Name: "machine3", Score: 0}},
			name:         "if exceed limit, use limit (with node which has no images present)",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metaDataProducer := priorities.NewPriorityMetadataFactory(
				schedulertesting.FakeServiceLister([]*v1.Service{}),
				schedulertesting.FakeControllerLister([]*v1.ReplicationController{}),
				schedulertesting.FakeReplicaSetLister([]*apps.ReplicaSet{}),
				schedulertesting.FakeStatefulSetLister([]*apps.StatefulSet{}))

			nodeNameToInfo := schedulernodeinfo.CreateNodeNameToInfoMap(nil, test.nodes)

			meta := metaDataProducer(test.pod, nodeNameToInfo)

			state := framework.NewCycleState()
			state.Write(migration.PrioritiesStateKey, &migration.PrioritiesStateData{Reference: meta})

			fh, _ := framework.NewFramework(nil, nil, nil)
			snapshot := fh.NodeInfoSnapshot()
			snapshot.NodeInfoMap = schedulernodeinfo.CreateNodeNameToInfoMap(nil, test.nodes)

			p, _ := New(nil, fh)
			var gotList framework.NodeScoreList
			for _, n := range test.nodes {
				nodeName := n.ObjectMeta.Name
				score, status := p.(framework.ScorePlugin).Score(state, test.pod, nodeName)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotList = append(gotList, framework.NodeScore{Name: nodeName, Score: score})
			}

			if !reflect.DeepEqual(test.expectedList, gotList) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedList, gotList)
			}
		})
	}
}

func makeImageNode(node string, status v1.NodeStatus) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: node},
		Status:     status,
	}
}
