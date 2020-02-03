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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	quantity = *resource.NewQuantity(1, resource.DecimalSI)
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

var (
	extendedResourceA = v1.ResourceName("example.com/aaa")
	hugePageResourceA = v1helper.HugePageResourceName(resource.MustParse("2Mi"))
)

func makeResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.NodeResources {
	return v1.NodeResources{
		Capacity: v1.ResourceList{
			v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
			v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
			v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
			extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
			v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
			hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
		},
	}
}

func makeAllocatableResources(milliCPU, memory, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func newResourcePod(usage ...schedulernodeinfo.Resource) *v1.Pod {
	containers := []v1.Container{}
	for _, req := range usage {
		containers = append(containers, v1.Container{
			Resources: v1.ResourceRequirements{Requests: req.ResourceList()},
		})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: containers,
		},
	}
}

func newPodWithPort(hostPorts ...int) *v1.Pod {
	networkPorts := []v1.ContainerPort{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, v1.ContainerPort{HostPort: int32(port)})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestGeneralPredicates(t *testing.T) {
	resourceTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulernodeinfo.NodeInfo
		node     *v1.Node
		fits     bool
		name     string
		wErr     error
		reasons  []PredicateFailureReason
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 19})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			fits: true,
			wErr: nil,
			name: "no resources/port/host requested always fits",
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 8, Memory: 10}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			fits: false,
			wErr: nil,
			reasons: []PredicateFailureReason{
				&InsufficientResourceError{ResourceName: v1.ResourceCPU, Requested: 8, Used: 5, Capacity: 10},
				&InsufficientResourceError{ResourceName: v1.ResourceMemory, Requested: 10, Used: 19, Capacity: 20},
			},
			name: "not enough cpu and memory resource",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "machine2",
				},
			},
			nodeInfo: schedulernodeinfo.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			fits:    false,
			wErr:    nil,
			reasons: []PredicateFailureReason{&PredicateFailureError{nodename.Name, nodename.ErrReason}},
			name:    "host not match",
		},
		{
			pod:      newPodWithPort(123),
			nodeInfo: schedulernodeinfo.NewNodeInfo(newPodWithPort(123)),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 0, 0, 0)},
			},
			fits:    false,
			wErr:    nil,
			reasons: []PredicateFailureReason{&PredicateFailureError{nodeports.Name, nodeports.ErrReason}},
			name:    "hostport conflict",
		},
	}
	for _, test := range resourceTests {
		t.Run(test.name, func(t *testing.T) {
			test.nodeInfo.SetNode(test.node)
			reasons, err := GeneralPredicates(test.pod, test.nodeInfo)
			fits := len(reasons) == 0 && err == nil
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !fits && !reflect.DeepEqual(reasons, test.reasons) {
				t.Errorf("unexpected failure reasons: %v, want: %v", reasons, test.reasons)
			}
			if fits != test.fits {
				t.Errorf("expected: %v got %v", test.fits, fits)
			}
		})
	}
}
