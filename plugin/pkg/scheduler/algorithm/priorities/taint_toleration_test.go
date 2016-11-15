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
	"encoding/json"
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func nodeWithTaints(nodeName string, taints []api.Taint) *api.Node {
	taintsData, _ := json.Marshal(taints)
	return &api.Node{
		ObjectMeta: api.ObjectMeta{
			Name: nodeName,
			Annotations: map[string]string{
				api.TaintsAnnotationKey: string(taintsData),
			},
		},
	}
}

func podWithTolerations(tolerations []api.Toleration) *api.Pod {
	tolerationData, _ := json.Marshal(tolerations)
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Annotations: map[string]string{
				api.TolerationsAnnotationKey: string(tolerationData),
			},
		},
	}
}

// This function will create a set of nodes and pods and test the priority
// Nodes with zero,one,two,three,four and hundred taints are created
// Pods with zero,one,two,three,four and hundred tolerations are created

func TestTaintAndToleration(t *testing.T) {
	tests := []struct {
		pod          *api.Pod
		nodes        []*api.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		// basic test case
		{
			test: "node with taints tolerated by the pod, gets a higher score than those node with intolerable taints",
			pod: podWithTolerations([]api.Toleration{{
				Key:      "foo",
				Operator: api.TolerationOpEqual,
				Value:    "bar",
				Effect:   api.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*api.Node{
				nodeWithTaints("nodeA", []api.Taint{{
					Key:    "foo",
					Value:  "bar",
					Effect: api.TaintEffectPreferNoSchedule,
				}}),
				nodeWithTaints("nodeB", []api.Taint{{
					Key:    "foo",
					Value:  "blah",
					Effect: api.TaintEffectPreferNoSchedule,
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
			pod: podWithTolerations([]api.Toleration{
				{
					Key:      "cpu-type",
					Operator: api.TolerationOpEqual,
					Value:    "arm64",
					Effect:   api.TaintEffectPreferNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: api.TolerationOpEqual,
					Value:    "ssd",
					Effect:   api.TaintEffectPreferNoSchedule,
				},
			}),
			nodes: []*api.Node{
				nodeWithTaints("nodeA", []api.Taint{}),
				nodeWithTaints("nodeB", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: api.TaintEffectPreferNoSchedule,
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
			pod: podWithTolerations([]api.Toleration{{
				Key:      "foo",
				Operator: api.TolerationOpEqual,
				Value:    "bar",
				Effect:   api.TaintEffectPreferNoSchedule,
			}}),
			nodes: []*api.Node{
				nodeWithTaints("nodeA", []api.Taint{}),
				nodeWithTaints("nodeB", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectPreferNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: api.TaintEffectPreferNoSchedule,
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
			pod: podWithTolerations([]api.Toleration{
				{
					Key:      "cpu-type",
					Operator: api.TolerationOpEqual,
					Value:    "arm64",
					Effect:   api.TaintEffectNoSchedule,
				}, {
					Key:      "disk-type",
					Operator: api.TolerationOpEqual,
					Value:    "ssd",
					Effect:   api.TaintEffectNoSchedule,
				},
			}),
			nodes: []*api.Node{
				nodeWithTaints("nodeA", []api.Taint{}),
				nodeWithTaints("nodeB", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectNoSchedule,
					},
				}),
				nodeWithTaints("nodeC", []api.Taint{
					{
						Key:    "cpu-type",
						Value:  "arm64",
						Effect: api.TaintEffectPreferNoSchedule,
					}, {
						Key:    "disk-type",
						Value:  "ssd",
						Effect: api.TaintEffectPreferNoSchedule,
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
