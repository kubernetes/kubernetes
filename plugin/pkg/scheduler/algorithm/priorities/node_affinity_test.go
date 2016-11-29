/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/v1"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func TestNodeAffinityPriority(t *testing.T) {
	label1 := map[string]string{"foo": "bar"}
	label2 := map[string]string{"key": "value"}
	label3 := map[string]string{"az": "az1"}
	label4 := map[string]string{"abc": "az11", "def": "az22"}
	label5 := map[string]string{"foo": "bar", "key": "value", "az": "az1"}

	affinity1 := map[string]string{
		v1.AffinityAnnotationKey: `
		{"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": [
			{
				"weight": 2,
				"preference": {
					"matchExpressions": [
						{
							"key": "foo",
							"operator": "In", "values": ["bar"]
						}
					]
				}
			}
		]}}`,
	}

	affinity2 := map[string]string{
		v1.AffinityAnnotationKey: `
		{"nodeAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": [
			{
				"weight": 2,
				"preference": {"matchExpressions": [
					{
						"key": "foo",
						"operator": "In", "values": ["bar"]
					}
				]}
			},
			{
				"weight": 4,
				"preference": {"matchExpressions": [
					{
						"key": "key",
						"operator": "In", "values": ["value"]
					}
				]}
			},
			{
				"weight": 5,
				"preference": {"matchExpressions": [
					{
						"key": "foo",
						"operator": "In", "values": ["bar"]
					},
					{
						"key": "key",
						"operator": "In", "values": ["value"]
					},
					{
						"key": "az",
						"operator": "In", "values": ["az1"]
					}
				]}
			}
		]}}`,
	}

	tests := []struct {
		pod          *v1.Pod
		nodes        []*v1.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Annotations: map[string]string{},
				},
			},
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			test:         "all machines are same priority as NodeAffinity is nil",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Annotations: affinity1,
				},
			},
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label4}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			test:         "no machine macthes preferred scheduling requirements in NodeAffinity of pod so all machines' priority is zero",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Annotations: affinity1,
				},
			},
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine3", Labels: label3}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 0}},
			test:         "only machine1 matches the preferred scheduling requirements of pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Annotations: affinity2,
				},
			},
			nodes: []*v1.Node{
				{ObjectMeta: v1.ObjectMeta{Name: "machine1", Labels: label1}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine5", Labels: label5}},
				{ObjectMeta: v1.ObjectMeta{Name: "machine2", Labels: label2}},
			},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 1}, {Host: "machine5", Score: 10}, {Host: "machine2", Score: 3}},
			test:         "all machines matches the preferred scheduling requirements of pod but with different priorities ",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(nil, test.nodes)
		nap := priorityFunction(CalculateNodeAffinityPriorityMap, CalculateNodeAffinityPriorityReduce)
		list, err := nap(test.pod, nodeNameToInfo, test.nodes)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: \nexpected %#v, \ngot      %#v", test.test, test.expectedList, list)
		}
	}
}
