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
	_ "sort"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

func TestServiceAffinityPriority(t *testing.T) {
	serviceLabel1 := map[string]string{
		"name": "service1",
	}
	serviceLabel2 := map[string]string{
		"name": "service2",
	}
	affinityLabel1 := map[string]string{
		"service-name": "service1",
	}
	affinityLabel2 := map[string]string{
		"service-name": "service2",
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
		services     []api.Service
		controllers  []api.ReplicationController
		expectedList algorithm.HostPriorityList
		test         string
	}{
		{
			pod:          new(api.Pod),
			nodes:        []string{"machine1", "machine2"},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "nothing scheduled",
		},
		{
			pod:          &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods:         []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}}},
			nodes:        []string{"machine1", "machine2"},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "no service to affiliate",
		},
		{
			pod:          &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods:         []*api.Pod{{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel2}}},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: serviceLabel2}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel2}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "different service",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "one service pod to affiliate",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "two service pods to affiliate on two machines",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1, Namespace: "ns1"}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 10}},
			test:         "two service pods, one in no namespace to affiliate",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1, Namespace: api.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel2}},
			},
			nodes:        []string{"machine1", "machine2"},
			services:     []api.Service{{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}}},
			expectedList: []algorithm.HostPriority{{"machine1", 0}, {"machine2", 0}},
			test:         "two service pods, one in default namespace, none to affiliate",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}, ObjectMeta: api.ObjectMeta{Namespace: api.NamespaceDefault}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1, Namespace: api.NamespaceDefault}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			},
			nodes: []string{"machine1", "machine2"},
			services: []api.Service{
				{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1, Namespace: api.NamespaceDefault}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "two service pods, one in default namespace to affiliate",
		},
		{
			pod: &api.Pod{Spec: api.PodSpec{AffinitySelector: affinityLabel1}},
			pods: []*api.Pod{
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
				{Spec: zone1Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
				{Spec: zone2Spec, ObjectMeta: api.ObjectMeta{Labels: serviceLabel2}},
			},
			nodes: []string{"machine1", "machine2"},
			services: []api.Service{
				{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 5}},
			test:         "four service pods, two on machine1 to affiliate, one on machine2 to affiliate",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			pods: []*api.Pod{
				{Spec: api.PodSpec{AffinitySelector: affinityLabel1, NodeName: "machine1"}},
			},
			nodes: []string{"machine1", "machine2"},
			services: []api.Service{
				{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "one pod, affiliated by existing",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			pods: []*api.Pod{
				{Spec: api.PodSpec{AffinitySelector: affinityLabel1, NodeName: "machine1"}},
				{Spec: api.PodSpec{AffinitySelector: affinityLabel1, NodeName: "machine2"}},
			},
			nodes: []string{"machine1", "machine2"},
			services: []api.Service{
				{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 10}},
			test:         "two pods, affiliated by each one on each machine",
		},
		{
			pod: &api.Pod{ObjectMeta: api.ObjectMeta{Labels: serviceLabel1}},
			pods: []*api.Pod{
				{Spec: api.PodSpec{AffinitySelector: affinityLabel1, NodeName: "machine1"}},
				{Spec: api.PodSpec{AffinitySelector: affinityLabel2, NodeName: "machine2"}},
			},
			nodes: []string{"machine1", "machine2"},
			services: []api.Service{
				{Spec: api.ServiceSpec{Selector: serviceLabel1}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel1}},
				{Spec: api.ServiceSpec{Selector: serviceLabel2}, ObjectMeta: api.ObjectMeta{Labels: affinityLabel2}},
			},
			expectedList: []algorithm.HostPriority{{"machine1", 10}, {"machine2", 0}},
			test:         "two pods, affiliated by one of them",
		},
	}

	for _, test := range tests {
		serviceAffinity := ServiceAffinity{serviceLister: algorithm.FakeServiceLister(test.services), controllerLister: algorithm.FakeControllerLister(test.controllers)}
		list, err := serviceAffinity.CalculateAffinityPriority(test.pod, algorithm.FakePodLister(test.pods), algorithm.FakeNodeLister(makeNodeList(test.nodes)))
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(test.expectedList, list) {
			t.Errorf("%s: \nexpected %#v, \ngot      %#v", test.test, test.expectedList, list)
		}
	}
}
