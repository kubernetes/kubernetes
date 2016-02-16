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
	"k8s.io/kubernetes/pkg/api/unversioned"
	wellknownlabels "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
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
		rss          []extensions.ReplicaSet
		services     []api.Service
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod:          new(api.Pod),
			nodes:        []string{"machine1", "machine2"},
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "nothing scheduled",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec}},
			nodes:        []string{"machine1", "machine2"},
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "no services",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 10}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 10}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 5}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "service with partial pod label matches with service and replication controller",
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
			rss:      []extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "disjoined service and replication controller should be treated equally",
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
			rss:      []extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "Replication controller with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 0}},
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
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "Replication controller with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{"machine1", 0}, {"machine2", 5}},
			test:         "Replication controller with partial pod label matches",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods)
		selectorSpread := SelectorSpread{podLister: algorithm.FakePodLister(test.pods), serviceLister: algorithm.FakeServiceLister(test.services), controllerLister: algorithm.FakeControllerLister(test.rcs), replicaSetLister: algorithm.FakeReplicaSetLister(test.rss)}
		list, err := selectorSpread.CalculateSpreadPriority(test.pod, nodeNameToInfo, algorithm.FakeNodeLister(makeNodeList(test.nodes)))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func TestZoneSelectorSpreadPriority(t *testing.T) {
	labels1 := map[string]string{
		"label1": "l1",
		"baz":    "blah",
	}
	labels2 := map[string]string{
		"label2": "l2",
		"baz":    "blah",
	}

	const nodeMachine1Zone1 = "machine1.zone1"
	const nodeMachine1Zone2 = "machine1.zone2"
	const nodeMachine2Zone2 = "machine2.zone2"
	const nodeMachine1Zone3 = "machine1.zone3"
	const nodeMachine2Zone3 = "machine2.zone3"
	const nodeMachine3Zone3 = "machine3.zone3"

	buildNodeLabels := func(failureDomain string) map[string]string {
		labels := map[string]string{
			wellknownlabels.LabelZoneFailureDomain: failureDomain,
		}
		return labels
	}
	labeledNodes := map[string]map[string]string{
		nodeMachine1Zone1: buildNodeLabels("zone1"),
		nodeMachine1Zone2: buildNodeLabels("zone2"),
		nodeMachine2Zone2: buildNodeLabels("zone2"),
		nodeMachine1Zone3: buildNodeLabels("zone3"),
		nodeMachine2Zone3: buildNodeLabels("zone3"),
		nodeMachine3Zone3: buildNodeLabels("zone3"),
	}

	buildPod := func(nodeName string, labels map[string]string) *api.Pod {
		pod := &api.Pod{Spec: api.PodSpec{NodeName: nodeName}, ObjectMeta: api.ObjectMeta{Labels: labels}}
		return pod
	}

	tests := []struct {
		pod          *api.Pod
		pods         []*api.Pod
		nodes        []string
		rcs          []api.ReplicationController
		rss          []extensions.ReplicaSet
		services     []api.Service
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod: new(api.Pod),
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 10},
				{nodeMachine1Zone2, 10},
				{nodeMachine2Zone2, 10},
				{nodeMachine1Zone3, 10},
				{nodeMachine2Zone3, 10},
				{nodeMachine3Zone3, 10},
			},
			test: "nothing scheduled",
		},
		{
			pod:  buildPod("", labels1),
			pods: []*api.Pod{buildPod(nodeMachine1Zone1, nil)},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 10},
				{nodeMachine1Zone2, 10},
				{nodeMachine2Zone2, 10},
				{nodeMachine1Zone3, 10},
				{nodeMachine2Zone3, 10},
				{nodeMachine3Zone3, 10},
			},
			test: "no services",
		},
		{
			pod:      buildPod("", labels1),
			pods:     []*api.Pod{buildPod(nodeMachine1Zone1, labels2)},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 10},
				{nodeMachine1Zone2, 10},
				{nodeMachine2Zone2, 10},
				{nodeMachine1Zone3, 10},
				{nodeMachine2Zone3, 10},
				{nodeMachine3Zone3, 10},
			},
			test: "different services",
		},
		{
			pod: buildPod("", labels1),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels2),
				buildPod(nodeMachine1Zone2, labels1),
			},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 10},
				{nodeMachine1Zone2, 0}, // Already have pod on machine
				{nodeMachine2Zone2, 3}, // Already have pod in zone
				{nodeMachine1Zone3, 10},
				{nodeMachine2Zone3, 10},
				{nodeMachine3Zone3, 10},
			},
			test: "two pods, 1 matching (in z2)",
		},
		{
			pod: buildPod("", labels1),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels2),
				buildPod(nodeMachine1Zone2, labels1),
				buildPod(nodeMachine2Zone2, labels1),
				buildPod(nodeMachine1Zone3, labels2),
				buildPod(nodeMachine2Zone3, labels1),
			},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 10},
				{nodeMachine1Zone2, 0}, // Pod on node
				{nodeMachine2Zone2, 0}, // Pod on node
				{nodeMachine1Zone3, 6}, // Pod in zone
				{nodeMachine2Zone3, 3}, // Pod on node
				{nodeMachine3Zone3, 6}, // Pod in zone
			},
			test: "five pods, 3 matching (z2=2, z3=1)",
		},
		{
			pod: buildPod("", labels1),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels1),
				buildPod(nodeMachine1Zone2, labels1),
				buildPod(nodeMachine2Zone2, labels2),
				buildPod(nodeMachine1Zone3, labels1),
			},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 0}, // Pod on node
				{nodeMachine1Zone2, 0}, // Pod on node
				{nodeMachine2Zone2, 3}, // Pod in zone
				{nodeMachine1Zone3, 0}, // Pod on node
				{nodeMachine2Zone3, 3}, // Pod in zone
				{nodeMachine3Zone3, 3}, // Pod in zone
			},
			test: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels1),
				buildPod(nodeMachine1Zone2, labels1),
				buildPod(nodeMachine1Zone3, labels1),
				buildPod(nodeMachine2Zone2, labels2),
			},
			services: []api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{nodeMachine1Zone1, 0}, // Pod on node
				{nodeMachine1Zone2, 0}, // Pod on node
				{nodeMachine2Zone2, 3}, // Pod in zone
				{nodeMachine1Zone3, 0}, // Pod on node
				{nodeMachine2Zone3, 3}, // Pod in zone
				{nodeMachine3Zone3, 3}, // Pod in zone
			},
			test: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone3, labels1),
				buildPod(nodeMachine1Zone2, labels1),
				buildPod(nodeMachine1Zone3, labels1),
			},
			rcs: []api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				// Note that because we put two pods on the same node (nodeMachine1Zone3),
				// the values here are questionable for zone2, in particular for nodeMachine1Zone2.
				// However they kind of make sense; zone1 is still most-highly favored.
				// zone3 is in general least favored, and m1.z3 particularly low priority.
				// We would probably prefer to see a bigger gap between putting a second
				// pod on m1.z2 and putting a pod on m2.z2, but the ordering is correct.
				// This is also consistent with what we have already.
				{nodeMachine1Zone1, 10}, // No pods in zone
				{nodeMachine1Zone2, 5},  // Pod on node
				{nodeMachine2Zone2, 6},  // Pod in zone
				{nodeMachine1Zone3, 0},  // Two pods on node
				{nodeMachine2Zone3, 3},  // Pod in zone
				{nodeMachine3Zone3, 3},  // Pod in zone
			},
			test: "Replication controller spreading (z1=0, z2=1, z3=2)",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods)
		selectorSpread := SelectorSpread{podLister: algorithm.FakePodLister(test.pods), serviceLister: algorithm.FakeServiceLister(test.services), controllerLister: algorithm.FakeControllerLister(test.rcs), replicaSetLister: algorithm.FakeReplicaSetLister(test.rss)}
		list, err := selectorSpread.CalculateSpreadPriority(test.pod, nodeNameToInfo, algorithm.FakeNodeLister(makeLabeledNodeList(labeledNodes)))
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
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod:   new(api.Pod),
			nodes: labeledNodes,
			expectedList: []schedulerapi.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "nothing scheduled",
		},
		{
			pod:   &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:  []*api.Pod{{Spec: zone1Spec}},
			nodes: labeledNodes,
			expectedList: []schedulerapi.HostPriority{{"machine11", 10}, {"machine12", 10},
				{"machine21", 10}, {"machine22", 10},
				{"machine01", 0}, {"machine02", 0}},
			test: "no services",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:     []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:    labeledNodes,
			services: []api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{{"machine11", 10}, {"machine12", 10},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 10}, {"machine12", 10},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 5}, {"machine12", 5},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 0}, {"machine12", 0},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 6}, {"machine12", 6},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 3}, {"machine12", 3},
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
			expectedList: []schedulerapi.HostPriority{{"machine11", 7}, {"machine12", 7},
				{"machine21", 5}, {"machine22", 5},
				{"machine01", 0}, {"machine02", 0}},
			test: "service pod on non-zoned node",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods)
		zoneSpread := ServiceAntiAffinity{podLister: algorithm.FakePodLister(test.pods), serviceLister: algorithm.FakeServiceLister(test.services), label: "zone"}
		list, err := zoneSpread.CalculateAntiAffinityPriority(test.pod, nodeNameToInfo, algorithm.FakeNodeLister(makeLabeledNodeList(test.nodes)))
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
