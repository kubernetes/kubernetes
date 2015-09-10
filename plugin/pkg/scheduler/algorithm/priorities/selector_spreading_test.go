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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

func TestSelectorSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	zone1Spec := api.PodSpec{
		NodeName: "machine1",
	}
	zone2Spec := api.PodSpec{
		NodeName: "machine2",
	}
	tests := []struct {
		pod          *api.Pod
		pods         []*api.Pod
		nodes        []string
		rcs          []api.ReplicationController
		services     []api.Service
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			pod:          new(api.Pod),
			nodes:        []string{"machine1", "machine2"},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "nothing scheduled",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec}},
			nodes:        []string{"machine1", "machine2"},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "no services",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "different services",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "two pods, one service pod",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "five pods, one service pod in no namespace",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "four pods, one service pod in default namespace",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns2"}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "five pods, one service pod in specific namespace",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "three pods, two service pods on different machines",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 5}, {"machine2", 0}},
			test:         "four pods, three service pods",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "service with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			rcs:      []api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// "baz=blah" matches both labels1 and labels2, and "foo=bar" matches only labels 1. This means that we assume that we want to
			// do spreading between all pods. The result should be exactly as above.
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "service with partial pod label matches with service and replication controller",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			rcs:      []api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// Taken together Service and Replication Controller should match all Pods, hence result should be equal to one above.
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "disjoined service and replication controller should be treated equally",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes: []string{"machine1", "machine2"},
			rcs:   []api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// Both Nodes have one pod from the given RC, hence both get 0 score.
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "Replication controller with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			rcs:          []api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "Replication controller with partial pod label matches",
		},
	}

	for _, test := range tests {
		selectorSpread := SelectorSpread{serviceLister: algorithm.FakeServiceLister(test.services), controllerLister: algorithm.FakeControllerLister(test.rcs)}
		list, err := selectorSpread.CalculateSpreadPriority(test.pod, algorithm.FakePodLister(test.pods), algorithm.FakeNodeLister(makeNodeList(test.nodes)))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestZoneSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"foo": "bar",
		"baz": "blah",
	}
	labels2 := map[string]string{
		"bar": "foo",
		"baz": "blah",
	}
	zone1 := map[string]string{
		"zone": "zone1",
	}
	zone2 := map[string]string{
		"zone": "zone2",
	}
	nozone := map[string]string{
		"name": "value",
	}
	zone0Spec := api.PodSpec{
		NodeName: "machine01",
	}
	zone1Spec := api.PodSpec{
		NodeName: "machine11",
	}
	zone2Spec := api.PodSpec{
		NodeName: "machine21",
	}
	labeledNodes := map[string]map[string]string{
		"machine01": nozone, "machine02": nozone,
		"machine11": zone1, "machine12": zone1,
		"machine21": zone2, "machine22": zone2,
	}
	tests := []struct {
		pod          *api.Pod
		pods         []*api.Pod
		nodes        map[string]map[string]string
		services     []api.Service
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			pod:   new(api.Pod),
			nodes: labeledNodes,
			expectedList: []algorithm.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "nothing scheduled",
		},
		{
			pod:   &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:  []*api.Pod{{Spec: zone1Spec}},
			nodes: labeledNodes,
			expectedList: []algorithm.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "no services",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:     []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []algorithm.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "different services",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone0Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 0}, {"machine22", 0},
				{"machine01", 0}, {"machine02", 0}},
			test: "three pods, one service pod",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine11", 5}, {"machine12", 5},
				{"machine21", 5}, {"machine22", 5},
				{"machine01", 0}, {"machine02", 0}},
			test: "three pods, two service pods on different machines",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: api.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, Namespace: "ns1"}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}}},
			expectedList: []algorithm.HostPriority{{"machine11", 0}, {"machine12", 0},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "three service label match pods in different namespaces",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine11", 6}, {"machine12", 6},
				{"machine21", 3}, {"machine22", 3},
				{"machine01", 0}, {"machine02", 0}},
			test: "four pods, three service pods",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []algorithm.HostPriority{{"machine11", 3}, {"machine12", 3},
				{"machine21", 6}, {"machine22", 6},
				{"machine01", 0}, {"machine02", 0}},
			test: "service with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone0Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []algorithm.HostPriority{{"machine11", 7}, {"machine12", 7},
				{"machine21", 5}, {"machine22", 5},
				{"machine01", 0}, {"machine02", 0}},
			test: "service pod on non-zoned node",
		},
	}

	for _, test := range tests {
		zoneSpread := ServiceAntiAffinity{serviceLister: algorithm.FakeServiceLister(test.services), label: "zone"}
		list, err := zoneSpread.CalculateAntiAffinityPriority(test.pod, algorithm.FakePodLister(test.pods), algorithm.FakeNodeLister(makeLabeledNodeList(test.nodes)))
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

func makeLabeledNodeList(nodeMap map[string]map[string]string) (result api.NodeList) {
	nodes := []api.Node{}
	for nodeName, labels := range nodeMap {
		nodes = append(nodes, api.Node{ObjectMeta: api.ObjectMeta{Name: nodeName, Labels: labels}})
	}
	return api.NodeList{Items: nodes}
}

func makeNodeList(nodeNames []string) api.NodeList {
	result := api.NodeList{
		Items: make([]api.Node, len(nodeNames)),
	}
	for ix := range nodeNames {
		result.Items[ix].Name = nodeNames[ix]
	}
	return result
}
