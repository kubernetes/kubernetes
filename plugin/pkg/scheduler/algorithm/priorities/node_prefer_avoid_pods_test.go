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

func TestNodePreferAvoidPriority(t *testing.T) {
	annotations1 := map[string]string{
		api.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicationController",
							                    "name": "foo",
							                    "uid": "abcdef123456",
							                    "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
	}
	annotations2 := map[string]string{
		api.PreferAvoidPodsAnnotationKey: `
							{
							    "preferAvoidPods": [
							        {
							            "podSignature": {
							                "podController": {
							                    "apiVersion": "v1",
							                    "kind": "ReplicaSet",
							                    "name": "foo",
							                    "uid": "qwert12345",
							                    "controller": true
							                }
							            },
							            "reason": "some reason",
							            "message": "some message"
							        }
							    ]
							}`,
	}
	testNodes := []*api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: "machine1", Annotations: annotations1},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "machine2", Annotations: annotations2},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "machine3"},
		},
	}
	trueVar := true
	tests := []struct {
		pod          *api.Pod
		nodes        []*api.Node
		expectedList schedulerapi.HostPriorityList
		test         string
	}{
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					OwnerReferences: []api.OwnerReference{
						{Kind: "ReplicationController", Name: "foo", UID: "abcdef123456", Controller: &trueVar},
					},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 0}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			test:         "pod managed by ReplicationController should avoid a node, this node get lowest priority score",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					OwnerReferences: []api.OwnerReference{
						{Kind: "RandomController", Name: "foo", UID: "abcdef123456", Controller: &trueVar},
					},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			test:         "ownership by random controller should be ignored",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					OwnerReferences: []api.OwnerReference{
						{Kind: "ReplicationController", Name: "foo", UID: "abcdef123456"},
					},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 10}, {Host: "machine3", Score: 10}},
			test:         "owner without Controller field set should be ignored",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Namespace: "default",
					OwnerReferences: []api.OwnerReference{
						{Kind: "ReplicaSet", Name: "foo", UID: "qwert12345", Controller: &trueVar},
					},
				},
			},
			nodes:        testNodes,
			expectedList: []schedulerapi.HostPriority{{Host: "machine1", Score: 10}, {Host: "machine2", Score: 0}, {Host: "machine3", Score: 10}},
			test:         "pod managed by ReplicaSet should avoid a node, this node get lowest priority score",
		},
	}

	for _, test := range tests {
		nodeNameToInfo := schedulercache.CreateNodeNameToInfoMap(nil, test.nodes)
		list, err := priorityFunction(CalculateNodePreferAvoidPodsPriorityMap, nil)(test.pod, nodeNameToInfo, test.nodes)
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
