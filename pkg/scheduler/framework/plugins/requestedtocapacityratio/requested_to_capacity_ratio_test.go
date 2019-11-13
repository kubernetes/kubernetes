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

package requestedtocapacityratio

import (
	"context"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
)

func TestRequestedToCapacityRatio(t *testing.T) {
	type test struct {
		name               string
		requestedPod       *v1.Pod
		nodes              []*v1.Node
		scheduledPods      []*v1.Pod
		expectedPriorities framework.NodeScoreList
	}

	tests := []test{
		{
			name:               "nothing scheduled, nothing requested (default - least requested nodes have priority)",
			requestedPod:       makePod("", 0, 0),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 4000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 0, 0), makePod("node2", 0, 0)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 100}, {Name: "node2", Score: 100}},
		},
		{
			name:               "nothing scheduled, resources requested, differently sized machines (default - least requested nodes have priority)",
			requestedPod:       makePod("", 3000, 5000),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 6000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 0, 0), makePod("node2", 0, 0)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 38}, {Name: "node2", Score: 50}},
		},
		{
			name:               "no resources requested, pods scheduled with resources (default - least requested nodes have priority)",
			requestedPod:       makePod("", 0, 0),
			nodes:              []*v1.Node{makeNode("node1", 4000, 10000), makeNode("node2", 6000, 10000)},
			scheduledPods:      []*v1.Pod{makePod("node1", 3000, 5000), makePod("node2", 3000, 5000)},
			expectedPriorities: []framework.NodeScore{{Name: "node1", Score: 38}, {Name: "node2", Score: 50}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := framework.NewCycleState()
			snapshot := nodeinfosnapshot.NewSnapshot(nodeinfosnapshot.CreateNodeInfoMap(test.scheduledPods, test.nodes))
			fh, _ := framework.NewFramework(nil, nil, nil, framework.WithSnapshotSharedLister(snapshot))
			args := &runtime.Unknown{Raw: []byte(`{"FunctionShape" : [{"Utilization" : 0, "Score" : 100}, {"Utilization" : 100, "Score" : 0}], "ResourceToWeightMap" : {"memory" : 1, "cpu" : 1}}`)}
			p, _ := New(args, fh)

			var gotPriorities framework.NodeScoreList
			for _, n := range test.nodes {
				score, status := p.(framework.ScorePlugin).Score(context.Background(), state, test.requestedPod, n.Name)
				if !status.IsSuccess() {
					t.Errorf("unexpected error: %v", status)
				}
				gotPriorities = append(gotPriorities, framework.NodeScore{Name: n.Name, Score: score})
			}

			if !reflect.DeepEqual(test.expectedPriorities, gotPriorities) {
				t.Errorf("expected:\n\t%+v,\ngot:\n\t%+v", test.expectedPriorities, gotPriorities)
			}
		})
	}
}

func makeNode(name string, milliCPU, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
			},
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

func makePod(node string, milliCPU, memory int64) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			NodeName: node,
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(memory, resource.DecimalSI),
						},
					},
				},
			},
		},
	}
}
