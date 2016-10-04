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
	"sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func controllerRef(kind, name, uid string) []api.OwnerReference {
	// TODO: When ControllerRef will be implemented uncomment code below.
	return nil
	//trueVar := true
	//return []api.OwnerReference{
	//	{Kind: kind, Name: name, UID: types.UID(uid), Controller: &trueVar},
	//}
}

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
		rcs          []*api.ReplicationController
		rss          []*extensions.ReplicaSet
		services     []*api.Service
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod:          new(api.Pod),
			nodes:        []string{"machine1", "machine2"},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "nothing scheduled",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec}},
			nodes:        []string{"machine1", "machine2"},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "no services",
		},
		{
			pod:          &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:         []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:        []string{"machine1", "machine2"},
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}},
			test:         "different services",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 5}, {Host: "machine2", Score: 0}},
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
			services:     []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "service with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			rcs:      []*api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			// "baz=blah" matches both labels1 and labels2, and "foo=bar" matches only labels 1. This means that we assume that we want to
			// do spreading between all pods. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "service with partial pod label matches with service and replication controller",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			rss:      []*extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "service with partial pod label matches with service and replica set",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			rcs:      []*api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			// Taken together Service and Replication Controller should match all Pods, hence result should be equal to one above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "disjoined service and replication controller should be treated equally",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "bar", "bar": "foo"}, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes:    []string{"machine1", "machine2"},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"bar": "foo"}}}},
			rss:      []*extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "disjoined service and replica set should be treated equally",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rcs:   []*api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"foo": "bar"}}}},
			// Both Nodes have one pod from the given RC, hence both get 0 score.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			test:         "Replication controller with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []*extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 0}},
			test:         "Replica set with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicationController", "name", "abc123")}},
			},
			nodes:        []string{"machine1", "machine2"},
			rcs:          []*api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "Another replication controller with partial pod label matches",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: labels1, OwnerReferences: controllerRef("ReplicaSet", "name", "abc123")}},
			},
			nodes: []string{"machine1", "machine2"},
			rss:   []*extensions.ReplicaSet{{Spec: extensions.ReplicaSetSpec{Selector: &unversioned.LabelSelector{MatchLabels: map[string]string{"baz": "blah"}}}}},
			// We use ReplicaSet, instead of ReplicationController. The result should be exactly as above.
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 5}},
			test:         "Another replication set with partial pod label matches",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nil)
		selectorSpread := SelectorSpread{
			serviceLister:    algorithm.FakeServiceLister(test.services),
			controllerLister: algorithm.FakeControllerLister(test.rcs),
			replicaSetLister: algorithm.FakeReplicaSetLister(test.rss),
		}
		list, err := selectorSpread.CalculateSpreadPriority(test.pod, nodeNameToInfo, makeNodeList(test.nodes))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: expected %#v, got %#v", test.test, test.expectedList, list)
		}
	}
}

func buildPod(nodeName string, labels map[string]string, ownerRefs []api.OwnerReference) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Labels: labels, OwnerReferences: ownerRefs},
		Spec:       api.PodSpec{NodeName: nodeName},
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
			unversioned.LabelZoneFailureDomain: failureDomain,
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

	tests := []struct {
		pod          *api.Pod
		pods         []*api.Pod
		nodes        []string
		rcs          []*api.ReplicationController
		rss          []*extensions.ReplicaSet
		services     []*api.Service
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod: new(api.Pod),
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 10},
				{Host: nodeMachine1Zone2, Score: 10},
				{Host: nodeMachine2Zone2, Score: 10},
				{Host: nodeMachine1Zone3, Score: 10},
				{Host: nodeMachine2Zone3, Score: 10},
				{Host: nodeMachine3Zone3, Score: 10},
			},
			test: "nothing scheduled",
		},
		{
			pod:  buildPod("", labels1, nil),
			pods: []*api.Pod{buildPod(nodeMachine1Zone1, nil, nil)},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 10},
				{Host: nodeMachine1Zone2, Score: 10},
				{Host: nodeMachine2Zone2, Score: 10},
				{Host: nodeMachine1Zone3, Score: 10},
				{Host: nodeMachine2Zone3, Score: 10},
				{Host: nodeMachine3Zone3, Score: 10},
			},
			test: "no services",
		},
		{
			pod:      buildPod("", labels1, nil),
			pods:     []*api.Pod{buildPod(nodeMachine1Zone1, labels2, nil)},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 10},
				{Host: nodeMachine1Zone2, Score: 10},
				{Host: nodeMachine2Zone2, Score: 10},
				{Host: nodeMachine1Zone3, Score: 10},
				{Host: nodeMachine2Zone3, Score: 10},
				{Host: nodeMachine3Zone3, Score: 10},
			},
			test: "different services",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
			},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 10},
				{Host: nodeMachine1Zone2, Score: 0}, // Already have pod on machine
				{Host: nodeMachine2Zone2, Score: 3}, // Already have pod in zone
				{Host: nodeMachine1Zone3, Score: 10},
				{Host: nodeMachine2Zone3, Score: 10},
				{Host: nodeMachine3Zone3, Score: 10},
			},
			test: "two pods, 1 matching (in z2)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels2, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels2, nil),
				buildPod(nodeMachine2Zone3, labels1, nil),
			},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 10},
				{Host: nodeMachine1Zone2, Score: 0}, // Pod on node
				{Host: nodeMachine2Zone2, Score: 0}, // Pod on node
				{Host: nodeMachine1Zone3, Score: 6}, // Pod in zone
				{Host: nodeMachine2Zone3, Score: 3}, // Pod on node
				{Host: nodeMachine3Zone3, Score: 6}, // Pod in zone
			},
			test: "five pods, 3 matching (z2=2, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
			},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 0}, // Pod on node
				{Host: nodeMachine1Zone2, Score: 0}, // Pod on node
				{Host: nodeMachine2Zone2, Score: 3}, // Pod in zone
				{Host: nodeMachine1Zone3, Score: 0}, // Pod on node
				{Host: nodeMachine2Zone3, Score: 3}, // Pod in zone
				{Host: nodeMachine3Zone3, Score: 3}, // Pod in zone
			},
			test: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1, nil),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone1, labels1, nil),
				buildPod(nodeMachine1Zone2, labels1, nil),
				buildPod(nodeMachine1Zone3, labels1, nil),
				buildPod(nodeMachine2Zone2, labels2, nil),
			},
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				{Host: nodeMachine1Zone1, Score: 0}, // Pod on node
				{Host: nodeMachine1Zone2, Score: 0}, // Pod on node
				{Host: nodeMachine2Zone2, Score: 3}, // Pod in zone
				{Host: nodeMachine1Zone3, Score: 0}, // Pod on node
				{Host: nodeMachine2Zone3, Score: 3}, // Pod in zone
				{Host: nodeMachine3Zone3, Score: 3}, // Pod in zone
			},
			test: "four pods, 3 matching (z1=1, z2=1, z3=1)",
		},
		{
			pod: buildPod("", labels1, controllerRef("ReplicationController", "name", "abc123")),
			pods: []*api.Pod{
				buildPod(nodeMachine1Zone3, labels1, controllerRef("ReplicationController", "name", "abc123")),
				buildPod(nodeMachine1Zone2, labels1, controllerRef("ReplicationController", "name", "abc123")),
				buildPod(nodeMachine1Zone3, labels1, controllerRef("ReplicationController", "name", "abc123")),
			},
			rcs: []*api.ReplicationController{{Spec: api.ReplicationControllerSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{
				// Note that because we put two pods on the same node (nodeMachine1Zone3),
				// the values here are questionable for zone2, in particular for nodeMachine1Zone2.
				// However they kind of make sense; zone1 is still most-highly favored.
				// zone3 is in general least favored, and m1.z3 particularly low priority.
				// We would probably prefer to see a bigger gap between putting a second
				// pod on m1.z2 and putting a pod on m2.z2, but the ordering is correct.
				// This is also consistent with what we have already.
				{Host: nodeMachine1Zone1, Score: 10}, // No pods in zone
				{Host: nodeMachine1Zone2, Score: 5},  // Pod on node
				{Host: nodeMachine2Zone2, Score: 6},  // Pod in zone
				{Host: nodeMachine1Zone3, Score: 0},  // Two pods on node
				{Host: nodeMachine2Zone3, Score: 3},  // Pod in zone
				{Host: nodeMachine3Zone3, Score: 3},  // Pod in zone
			},
			test: "Replication controller spreading (z1=0, z2=1, z3=2)",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nil)
		selectorSpread := SelectorSpread{
			serviceLister:    algorithm.FakeServiceLister(test.services),
			controllerLister: algorithm.FakeControllerLister(test.rcs),
			replicaSetLister: algorithm.FakeReplicaSetLister(test.rss),
		}
		list, err := selectorSpread.CalculateSpreadPriority(test.pod, nodeNameToInfo, makeLabeledNodeList(labeledNodes))
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
		services     []*api.Service
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod:   new(api.Pod),
			nodes: labeledNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 10}, {Host: "machine12", Score: 10},
				{Host: "machine21", Score: 10}, {Host: "machine22", Score: 10},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
			test: "nothing scheduled",
		},
		{
			pod:   &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:  []*api.Pod{{Spec: zone1Spec}},
			nodes: labeledNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 10}, {Host: "machine12", Score: 10},
				{Host: "machine21", Score: 10}, {Host: "machine22", Score: 10},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
			test: "no services",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: labels1}},
			pods:     []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: labels2}}},
			nodes:    labeledNodes,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"key": "value"}}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 10}, {Host: "machine12", Score: 10},
				{Host: "machine21", Score: 10}, {Host: "machine22", Score: 10},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 10}, {Host: "machine12", Score: 10},
				{Host: "machine21", Score: 0}, {Host: "machine22", Score: 0},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 5}, {Host: "machine12", Score: 5},
				{Host: "machine21", Score: 5}, {Host: "machine22", Score: 5},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}, ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 0}, {Host: "machine12", Score: 0},
				{Host: "machine21", Score: 10}, {Host: "machine22", Score: 10},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 6}, {Host: "machine12", Score: 6},
				{Host: "machine21", Score: 3}, {Host: "machine22", Score: 3},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: map[string]string{"baz": "blah"}}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 3}, {Host: "machine12", Score: 3},
				{Host: "machine21", Score: 6}, {Host: "machine22", Score: 6},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
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
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: labels1}}},
			expectedList: []schedulerapi.HostPriority{{Host: "machine11", Score: 7}, {Host: "machine12", Score: 7},
				{Host: "machine21", Score: 5}, {Host: "machine22", Score: 5},
				{Host: "machine01", Score: 0}, {Host: "machine02", Score: 0}},
			test: "service pod on non-zoned node",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(test.pods, nil)
		zoneSpread := ServiceAntiAffinity{podLister: algorithm.FakePodLister(test.pods), serviceLister: algorithm.FakeServiceLister(test.services), label: "zone"}
		list, err := zoneSpread.CalculateAntiAffinityPriority(test.pod, nodeNameToInfo, makeLabeledNodeList(test.nodes))
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

func makeLabeledNodeList(nodeMap map[string]map[string]string) []*api.Node {
	nodes := make([]*api.Node, 0, len(nodeMap))
	for nodeName, labels := range nodeMap {
		nodes = append(nodes, &api.Node{ObjectMeta: api.ObjectMeta{Name: nodeName, Labels: labels}})
	}
	return nodes
}

func makeNodeList(nodeNames []string) []*api.Node {
	nodes := make([]*api.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		nodes = append(nodes, &api.Node{ObjectMeta: api.ObjectMeta{Name: nodeName}})
	}
	return nodes
}
