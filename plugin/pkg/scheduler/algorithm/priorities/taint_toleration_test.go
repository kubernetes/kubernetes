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
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func nodeWithTaints(nodeName string, taints []v1.Taint) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: nodeName,
		},
		Spec: v1.NodeSpec{
			Taints: taints,
		},
	}
}

func podWithTolerations(tolerations []v1.Toleration) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Tolerations: tolerations,
		},
	}
}

// This function will create a set of nodes and pods and test the priority
// Nodes with zero,one,two,three,four and hundred taints are created
// Pods with zero,one,two,three,four and hundred tolerations are created

func TestTaintAndToleration(t *testing.T) {
	tests := []struct {
		pod          *v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		// basic test case
		{
			test: "node with taints tolerated by the pod, gets a higher score than those node with intolerable taints",
			pod: podWithTolerations([]v1.Toleration{{
				Key:      "foo",
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{{
					Key:    "foo",
					Value:  "bar",
					Effect: v1.TaintEffectPreferNoSchedule,
				}}),
				nodeWithTaints("nodeB", []v1.Taint{{
					Key:    "foo",
					Value:  "blah",
					Effect: v1.TaintEffectPreferNoSchedule,
				}}),
			},
			expectedList: []schedulerapi.HostPriority{
				{Host: "nodeA", Score: 10},
				{Host: "nodeB", Score: 0},
			},
		},
		// the count of taints that are tolerated by pod, does not matter.
		{
			test: "the nodes that all of their taints are tolerated by the pod, get the same score, no matter how many tolerable taints a node has",
			pod: podWithTolerations([]v1.Toleration{
				{
					Key:      "cpu-type",
					Operator: v1.TolerationOpEqual,
					Value:    "arm64",
					Effect:   v1.TaintEffectPreferNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: v1.TolerationOpEqual,
					Value:    "ssd",
					Effect:   v1.TaintEffectPreferNoSchedule,
				},
			}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []schedulerapi.HostPriority{
				{Host: "nodeA", Score: 10},
				{Host: "nodeB", Score: 10},
				{Host: "nodeC", Score: 10},
			},
		},
		// the count of taints on a node that are not tolerated by pod, matters.
		{
			test: "the more intolerable taints a node has, the lower score it gets.",
			pod: podWithTolerations([]v1.Toleration{{
				Key:      "foo",
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []schedulerapi.HostPriority{
				{Host: "nodeA", Score: 10},
				{Host: "nodeB", Score: 5},
				{Host: "nodeC", Score: 0},
			},
		},
		// taints-tolerations priority only takes care about the taints and tolerations that have effect PreferNoSchedule
		{
			test: "only taints and tolerations that have effect PreferNoSchedule are checked by taints-tolerations priority function",
			pod: podWithTolerations([]v1.Toleration{
				{
					Key:      "cpu-type",
					Operator: v1.TolerationOpEqual,
					Value:    "arm64",
					Effect:   v1.TaintEffectNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: v1.TolerationOpEqual,
					Value:    "ssd",
					Effect:   v1.TaintEffectNoSchedule,
				},
			}),
			nodes: []*v1.Node{
				nodeWithTaints("nodeA", []v1.Taint{}),
				nodeWithTaints("nodeB", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []v1.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: v1.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: v1.TaintEffectPreferNoSchedule,
					},
				}),
			},
			expectedList: []schedulerapi.HostPriority{
				{Host: "nodeA", Score: 10},
				{Host: "nodeB", Score: 10},
				{Host: "nodeC", Score: 0},
			},
		},
	}
	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(nil, test.nodes)
		ttp := priorityFunction(ComputeTaintTolerationPriorityMap, ComputeTaintTolerationPriorityReduce)
		list, err := ttp(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("%s, unexpected error: %v", test.test, err)
		}

		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s,\nexpected:\n\t%+v,\ngot:\n\t%+v", test.test, test.expectedList, list)
		}
	}

}
