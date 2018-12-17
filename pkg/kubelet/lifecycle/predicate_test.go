/*
Copyright 2018 The Kubernetes Authors.

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

package lifecycle

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	quantity              = *resource.NewQuantity(1, resource.DecimalSI)
	extendedResourceName1 = "example.com/er1"
	extendedResourceName2 = "example.com/er2"
)

func TestRemoveMissingExtendedResources(t *testing.T) {
	for _, test := range []struct {
		desc string
		pod  *v1.Pod
		node *v1.Node

		expectedPod *v1.Pod
	}{
		{
			desc: "requests in Limits should be ignored",
			pod: makeTestPod(
				v1.ResourceList{},                        // Requests
				v1.ResourceList{"foo.com/bar": quantity}, // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/baz": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{},                        // Requests
				v1.ResourceList{"foo.com/bar": quantity}, // Limits
			),
		},
		{
			desc: "requests for resources available in node should not be removed",
			pod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{},                        // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/bar": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{}),                       // Limits
		},
		{
			desc: "requests for resources unavailable in node should be removed",
			pod: makeTestPod(
				v1.ResourceList{"foo.com/bar": quantity}, // Requests
				v1.ResourceList{},                        // Limits
			),
			node: makeTestNode(
				v1.ResourceList{"foo.com/baz": quantity}, // Allocatable
			),
			expectedPod: makeTestPod(
				v1.ResourceList{}, // Requests
				v1.ResourceList{}, // Limits
			),
		},
	} {
		nodeInfo := schedulernodeinfo.NewNodeInfo()
		nodeInfo.SetNode(test.node)
		pod := removeMissingExtendedResources(test.pod, nodeInfo)
		if !reflect.DeepEqual(pod, test.expectedPod) {
			t.Errorf("%s: Expected pod\n%v\ngot\n%v\n", test.desc, test.expectedPod, pod)
		}
	}
}

func makeTestPod(requests, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

func makeTestNode(allocatable v1.ResourceList) *v1.Node {
	return &v1.Node{
		Status: v1.NodeStatus{
			Allocatable: allocatable,
		},
	}
}
