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

package predicates

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"k8s.io/gengo/parser"
	"k8s.io/gengo/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/util/codeinspector"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type FakeNodeInfo api.Node

func (n FakeNodeInfo) GetNodeInfo(nodeName string) (*api.Node, error) {
	node := api.Node(n)
	return &node, nil
}

type FakeNodeListInfo []api.Node

func (nodes FakeNodeListInfo) GetNodeInfo(nodeName string) (*api.Node, error) {
	for _, node := range nodes {
		if node.Name == nodeName {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("Unable to find node: %s", nodeName)
}

type FakePersistentVolumeClaimInfo []api.PersistentVolumeClaim

func (pvcs FakePersistentVolumeClaimInfo) GetPersistentVolumeClaimInfo(namespace string, pvcID string) (*api.PersistentVolumeClaim, error) {
	for _, pvc := range pvcs {
		if pvc.Name == pvcID && pvc.Namespace == namespace {
			return &pvc, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume claim: %s/%s", namespace, pvcID)
}

type FakePersistentVolumeInfo []api.PersistentVolume

func (pvs FakePersistentVolumeInfo) GetPersistentVolumeInfo(pvID string) (*api.PersistentVolume, error) {
	for _, pv := range pvs {
		if pv.Name == pvID {
			return &pv, nil
		}
	}
	return nil, fmt.Errorf("Unable to find persistent volume: %s", pvID)
}

var (
	opaqueResourceA = api.OpaqueIntResourceName("AAA")
	opaqueResourceB = api.OpaqueIntResourceName("BBB")
)

func makeResources(milliCPU, memory, nvidiaGPUs, pods, opaqueA int64) api.NodeResources {
	return api.NodeResources{
		Capacity: api.ResourceList{
			api.ResourceCPU:       *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
			api.ResourceMemory:    *resource.NewQuantity(memory, resource.BinarySI),
			api.ResourcePods:      *resource.NewQuantity(pods, resource.DecimalSI),
			api.ResourceNvidiaGPU: *resource.NewQuantity(nvidiaGPUs, resource.DecimalSI),
			opaqueResourceA:       *resource.NewQuantity(opaqueA, resource.DecimalSI),
		},
	}
}

func makeAllocatableResources(milliCPU, memory, nvidiaGPUs, pods, opaqueA int64) api.ResourceList {
	return api.ResourceList{
		api.ResourceCPU:       *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		api.ResourceMemory:    *resource.NewQuantity(memory, resource.BinarySI),
		api.ResourcePods:      *resource.NewQuantity(pods, resource.DecimalSI),
		api.ResourceNvidiaGPU: *resource.NewQuantity(nvidiaGPUs, resource.DecimalSI),
		opaqueResourceA:       *resource.NewQuantity(opaqueA, resource.DecimalSI),
	}
}

func newResourcePod(usage ...schedulercache.Resource) *api.Pod {
	containers := []api.Container{}
	for _, req := range usage {
		containers = append(containers, api.Container{
			Resources: api.ResourceRequirements{Requests: req.ResourceList()},
		})
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: containers,
		},
	}
}

func newResourceInitPod(pod *api.Pod, usage ...schedulercache.Resource) *api.Pod {
	pod.Spec.InitContainers = newResourcePod(usage...).Spec.Containers
	return pod
}

func PredicateMetadata(p *api.Pod, nodeInfo map[string]*schedulercache.NodeInfo) interface{} {
	pm := PredicateMetadataFactory{algorithm.FakePodLister{p}}
	return pm.GetMetadata(p, nodeInfo)
}

func TestPodFitsResources(t *testing.T) {
	enoughPodsTests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: &api.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 10, Memory: 20})),
			fits: true,
			test: "no resources requested always fits",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 10, Memory: 20})),
			fits: false,
			test: "too many resources fails",
			reasons: []algorithm.PredicateFailureReason{
				NewInsufficientResourceError(api.ResourceCPU, 1, 10, 10),
				NewInsufficientResourceError(api.ResourceMemory, 1, 20, 20),
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 3, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to init container cpu",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 3, Memory: 1}, schedulercache.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to highest init container cpu",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 3}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to init container memory",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceMemory, 3, 19, 20)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 3}, schedulercache.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to highest init container memory",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceMemory, 3, 19, 20)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits: true,
			test: "init container fits because it's the max, not sum, of containers and init containers",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 1}, schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits: true,
			test: "multiple init containers fit because it's the max, not sum, of containers and init containers",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 5})),
			fits: true,
			test: "both resources fit",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 5})),
			fits:    false,
			test:    "one resource memory fits",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceCPU, 2, 9, 10)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "one resource cpu fits",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceMemory, 2, 19, 20)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits: true,
			test: "equal edge case",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 4, Memory: 1}), schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits: true,
			test: "equal edge case for init container",
		},
		{
			pod:      newResourcePod(schedulercache.Resource{OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(newResourcePod(schedulercache.Resource{})),
			fits:     true,
			test:     "opaque resource fits",
		},
		{
			pod:      newResourceInitPod(newResourcePod(schedulercache.Resource{}), schedulercache.Resource{OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(newResourcePod(schedulercache.Resource{})),
			fits:     true,
			test:     "opaque resource fits for init container",
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 0}})),
			fits:    false,
			test:    "opaque resource capacity enforced",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 10, 0, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 0}})),
			fits:    false,
			test:    "opaque resource capacity enforced for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 10, 0, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 5}})),
			fits:    false,
			test:    "opaque resource allocatable enforced",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 1, 5, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 5}})),
			fits:    false,
			test:    "opaque resource allocatable enforced for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 1, 5, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 3}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 2}})),
			fits:    false,
			test:    "opaque resource allocatable enforced for multiple containers",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 6, 2, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 3}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 2}})),
			fits: true,
			test: "opaque resource allocatable admits multiple init containers",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 6}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceA: 2}})),
			fits:    false,
			test:    "opaque resource allocatable enforced for multiple init containers",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceA, 6, 2, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceB: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			test:    "opaque resource allocatable enforced for unknown resource",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceB, 1, 0, 0)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, OpaqueIntResources: map[api.ResourceName]int64{opaqueResourceB: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			test:    "opaque resource allocatable enforced for unknown resource for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(opaqueResourceB, 1, 0, 0)},
		},
	}

	for _, test := range enoughPodsTests {
		node := api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 5).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 5)}}
		test.nodeInfo.SetNode(&node)
		fits, reasons, err := PodFitsResources(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, test.reasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, test.reasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}

	notEnoughPodsTests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: &api.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 10, Memory: 20})),
			fits:    false,
			test:    "even without specified resources predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 5})),
			fits:    false,
			test:    "even if both resources fit predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "even for equal edge case predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 1}), schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "even for equal edge case predicate fails when there's no space for additional pod due to init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourcePods, 1, 1, 1)},
		},
	}
	for _, test := range notEnoughPodsTests {
		node := api.Node{Status: api.NodeStatus{Capacity: api.ResourceList{}, Allocatable: makeAllocatableResources(10, 20, 0, 1, 0)}}
		test.nodeInfo.SetNode(&node)
		fits, reasons, err := PodFitsResources(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, test.reasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, test.reasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestPodFitsHost(t *testing.T) {
	tests := []struct {
		pod  *api.Pod
		node *api.Node
		fits bool
		test string
	}{
		{
			pod:  &api.Pod{},
			node: &api.Node{},
			fits: true,
			test: "no host specified",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeName: "foo",
				},
			},
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fits: true,
			test: "host matches",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeName: "bar",
				},
			},
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
			fits: false,
			test: "host doesn't match",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrPodNotMatchHostName}

	for _, test := range tests {
		nodeInfo := schedulercache.NewNodeInfo()
		nodeInfo.SetNode(test.node)
		fits, reasons, err := PodFitsHost(test.pod, PredicateMetadata(test.pod, nil), nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: unexpected difference: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func newPod(host string, hostPorts ...int) *api.Pod {
	networkPorts := []api.ContainerPort{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, api.ContainerPort{HostPort: int32(port)})
	}
	return &api.Pod{
		Spec: api.PodSpec{
			NodeName: host,
			Containers: []api.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestPodFitsHostPorts(t *testing.T) {
	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
	}{
		{
			pod:      &api.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(),
			fits:     true,
			test:     "nothing running",
		},
		{
			pod: newPod("m1", 8080),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", 9090)),
			fits: true,
			test: "other port",
		},
		{
			pod: newPod("m1", 8080),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", 8080)),
			fits: false,
			test: "same port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", 8080)),
			fits: false,
			test: "second port",
		},
		{
			pod: newPod("m1", 8000, 8080),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", 8001, 8080)),
			fits: false,
			test: "second port",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrPodNotFitsHostPorts}

	for _, test := range tests {
		fits, reasons, err := PodFitsHostPorts(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if test.fits != fits {
			t.Errorf("%s: expected %v, saw %v", test.test, test.fits, fits)
		}
	}
}

func TestGetUsedPorts(t *testing.T) {
	tests := []struct {
		pods []*api.Pod

		ports map[int]bool
	}{
		{
			[]*api.Pod{
				newPod("m1", 9090),
			},
			map[int]bool{9090: true},
		},
		{
			[]*api.Pod{
				newPod("m1", 9090),
				newPod("m1", 9091),
			},
			map[int]bool{9090: true, 9091: true},
		},
		{
			[]*api.Pod{
				newPod("m1", 9090),
				newPod("m2", 9091),
			},
			map[int]bool{9090: true, 9091: true},
		},
	}

	for _, test := range tests {
		ports := GetUsedPorts(test.pods...)
		if !reflect.DeepEqual(test.ports, ports) {
			t.Errorf("%s: expected %v, got %v", "test get used ports", test.ports, ports)
		}
	}
}

func TestDiskConflicts(t *testing.T) {
	volState := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "foo",
					},
				},
			},
		},
	}
	volState2 := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&api.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&api.Pod{}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "one state"},
		{&api.Pod{Spec: volState}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), false, "same state"},
		{&api.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "different state"},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrDiskConflict}

	for _, test := range tests {
		ok, reasons, err := NoDiskConflict(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !ok && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if test.isOk && !ok {
			t.Errorf("%s: expected ok, got none.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
		if !test.isOk && ok {
			t.Errorf("%s: expected no ok, got one.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
	}
}

func TestAWSDiskConflicts(t *testing.T) {
	volState := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: "foo",
					},
				},
			},
		},
	}
	volState2 := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{
						VolumeID: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&api.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&api.Pod{}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "one state"},
		{&api.Pod{Spec: volState}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), false, "same state"},
		{&api.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "different state"},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrDiskConflict}

	for _, test := range tests {
		ok, reasons, err := NoDiskConflict(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !ok && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if test.isOk && !ok {
			t.Errorf("%s: expected ok, got none.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
		if !test.isOk && ok {
			t.Errorf("%s: expected no ok, got one.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
	}
}

func TestRBDDiskConflicts(t *testing.T) {
	volState := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					RBD: &api.RBDVolumeSource{
						CephMonitors: []string{"a", "b"},
						RBDPool:      "foo",
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}
	volState2 := api.PodSpec{
		Volumes: []api.Volume{
			{
				VolumeSource: api.VolumeSource{
					RBD: &api.RBDVolumeSource{
						CephMonitors: []string{"c", "d"},
						RBDPool:      "foo",
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&api.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&api.Pod{}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "one state"},
		{&api.Pod{Spec: volState}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), false, "same state"},
		{&api.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&api.Pod{Spec: volState}), true, "different state"},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrDiskConflict}

	for _, test := range tests {
		ok, reasons, err := NoDiskConflict(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !ok && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if test.isOk && !ok {
			t.Errorf("%s: expected ok, got none.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
		if !test.isOk && ok {
			t.Errorf("%s: expected no ok, got one.  %v %s %s", test.test, test.pod, test.nodeInfo, test.test)
		}
	}
}

func TestPodFitsSelector(t *testing.T) {
	tests := []struct {
		pod    *api.Pod
		labels map[string]string
		fits   bool
		test   string
	}{
		{
			pod:  &api.Pod{},
			fits: true,
			test: "no selector",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			fits: false,
			test: "missing labels",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "same labels",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
				"baz": "blah",
			},
			fits: true,
			test: "node labels are superset",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
						"baz": "blah",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "node labels are subset",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
									"operator": "In",
									"values": ["bar", "value2"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with matchExpressions using In operator that matches the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "kernel-version",
									"operator": "Gt",
									"values": ["0204"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				// We use two digit to denote major version and two digit for minor version.
				"kernel-version": "0206",
			},
			fits: true,
			test: "Pod with matchExpressions using Gt operator that matches the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "mem-type",
									"operator": "NotIn",
									"values": ["DDR", "DDR2"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"mem-type": "DDR3",
			},
			fits: true,
			test: "Pod with matchExpressions using NotIn operator that matches the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "GPU",
									"operator": "Exists"
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"GPU": "NVIDIA-GRID-K1",
			},
			fits: true,
			test: "Pod with matchExpressions using Exists operator that matches the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
									"operator": "In",
									"values": ["value1", "value2"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with affinity that don't match node's labels won't schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": null
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with a nil []NodeSelectorTerm in affinity, can't match the node's labels and won't schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": []
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with an empty []NodeSelectorTerm in affinity, can't match the node's labels and won't schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{}, {}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with invalid NodeSelectTerms in affinity will match no objects and won't schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{"matchExpressions": [{}]}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with empty MatchExpressions is not a valid value will match no objects and won't schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"some-key": "some-value",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with no Affinity will schedule onto a node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": null
						}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with Affinity but nil NodeSelector will schedule onto a node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "GPU",
									"operator": "Exists"
								}, {
									"key": "GPU",
									"operator": "NotIn",
									"values": ["AMD", "INTER"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"GPU": "NVIDIA-GRID-K1",
			},
			fits: true,
			test: "Pod with multiple matchExpressions ANDed that matches the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "GPU",
									"operator": "Exists"
								}, {
									"key": "GPU",
									"operator": "In",
									"values": ["AMD", "INTER"]
								}]
							}]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"GPU": "NVIDIA-GRID-K1",
			},
			fits: false,
			test: "Pod with multiple matchExpressions ANDed that doesn't match the existing node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [
								{
									"matchExpressions": [{
										"key": "foo",
										"operator": "In",
										"values": ["bar", "value2"]
									}]
								},
								{
									"matchExpressions": [{
										"key": "diffkey",
										"operator": "In",
										"values": ["wrong", "value2"]
									}]
								}
							]
						}}}`,
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with multiple NodeSelectorTerms ORed in affinity, matches the node's labels and will schedule onto the node",
		},
		// TODO: Uncomment this test when implement RequiredDuringSchedulingRequiredDuringExecution
		//		{
		//			pod: &api.Pod{
		//				ObjectMeta: api.ObjectMeta{
		//					Annotations: map[string]string{
		//						api.AffinityAnnotationKey: `
		//						{"nodeAffinity": {
		//							"requiredDuringSchedulingRequiredDuringExecution": {
		//								"nodeSelectorTerms": [{
		//									"matchExpressions": [{
		//										"key": "foo",
		//										"operator": "In",
		//										"values": ["bar", "value2"]
		//									}]
		//								}]
		//							},
		//							"requiredDuringSchedulingIgnoredDuringExecution": {
		//								"nodeSelectorTerms": [{
		//									"matchExpressions": [{
		//										"key": "foo",
		//										"operator": "NotIn",
		//										"values": ["bar", "value2"]
		//									}]
		//								}]
		//							}
		//						}}`,
		//					},
		//				},
		//			},
		//			labels: map[string]string{
		//				"foo": "bar",
		//			},
		//			fits: false,
		//			test: "Pod with an Affinity both requiredDuringSchedulingRequiredDuringExecution and " +
		//				"requiredDuringSchedulingIgnoredDuringExecution indicated that don't match node's labels and won't schedule onto the node",
		//		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
									"operator": "Exists"
								}]
							}]
						}}}`,
					},
				},
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with an Affinity and a PodSpec.NodeSelector(the old thing that we are deprecating) " +
				"both are satisfied, will schedule onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": [{
									"key": "foo",
									"operator": "Exists"
								}]
							}]
						}}}`,
					},
				},
				Spec: api.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			labels: map[string]string{
				"foo": "barrrrrr",
			},
			fits: false,
			test: "Pod with an Affinity matches node's labels but the PodSpec.NodeSelector(the old thing that we are deprecating) " +
				"is not satisfied, won't schedule onto the node",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeSelectorNotMatch}

	for _, test := range tests {
		node := api.Node{ObjectMeta: api.ObjectMeta{Labels: test.labels}}
		nodeInfo := schedulercache.NewNodeInfo()
		nodeInfo.SetNode(&node)

		fits, reasons, err := PodSelectorMatches(test.pod, PredicateMetadata(test.pod, nil), nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestNodeLabelPresence(t *testing.T) {
	label := map[string]string{"foo": "bar", "bar": "foo"}
	tests := []struct {
		pod      *api.Pod
		labels   []string
		presence bool
		fits     bool
		test     string
	}{
		{
			labels:   []string{"baz"},
			presence: true,
			fits:     false,
			test:     "label does not match, presence true",
		},
		{
			labels:   []string{"baz"},
			presence: false,
			fits:     true,
			test:     "label does not match, presence false",
		},
		{
			labels:   []string{"foo", "baz"},
			presence: true,
			fits:     false,
			test:     "one label matches, presence true",
		},
		{
			labels:   []string{"foo", "baz"},
			presence: false,
			fits:     false,
			test:     "one label matches, presence false",
		},
		{
			labels:   []string{"foo", "bar"},
			presence: true,
			fits:     true,
			test:     "all labels match, presence true",
		},
		{
			labels:   []string{"foo", "bar"},
			presence: false,
			fits:     false,
			test:     "all labels match, presence false",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeLabelPresenceViolated}

	for _, test := range tests {
		node := api.Node{ObjectMeta: api.ObjectMeta{Labels: label}}
		nodeInfo := schedulercache.NewNodeInfo()
		nodeInfo.SetNode(&node)

		labelChecker := NodeLabelChecker{test.labels, test.presence}
		fits, reasons, err := labelChecker.CheckNodeLabelPresence(test.pod, PredicateMetadata(test.pod, nil), nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestServiceAffinity(t *testing.T) {
	selector := map[string]string{"foo": "bar"}
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	labels2 := map[string]string{
		"region": "r1",
		"zone":   "z12",
	}
	labels3 := map[string]string{
		"region": "r2",
		"zone":   "z21",
	}
	labels4 := map[string]string{
		"region": "r2",
		"zone":   "z22",
	}
	node1 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: labels1}}
	node2 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: labels2}}
	node3 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: labels3}}
	node4 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine4", Labels: labels4}}
	node5 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine5", Labels: labels4}}
	tests := []struct {
		pod      *api.Pod
		pods     []*api.Pod
		services []*api.Service
		node     *api.Node
		labels   []string
		fits     bool
		test     string
	}{
		{
			pod:    new(api.Pod),
			node:   &node1,
			fits:   true,
			labels: []string{"region"},
			test:   "nothing scheduled",
		},
		{
			pod:    &api.Pod{Spec: api.PodSpec{NodeSelector: map[string]string{"region": "r1"}}},
			node:   &node1,
			fits:   true,
			labels: []string{"region"},
			test:   "pod with region label match",
		},
		{
			pod:    &api.Pod{Spec: api.PodSpec{NodeSelector: map[string]string{"region": "r2"}}},
			node:   &node1,
			fits:   false,
			labels: []string{"region"},
			test:   "pod with region label mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on same node",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine2"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on different node, region match",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service pod on different node, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns2"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service in different namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns2"}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "pod in different namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine3"}, ObjectMeta: api.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}, ObjectMeta: api.ObjectMeta{Namespace: "ns1"}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service and pod in same namespace, region mismatch",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine2"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region", "zone"},
			test:     "service pod on different node, multiple labels, not all match",
		},
		{
			pod:      &api.Pod{ObjectMeta: api.ObjectMeta{Labels: selector}},
			pods:     []*api.Pod{{Spec: api.PodSpec{NodeName: "machine5"}, ObjectMeta: api.ObjectMeta{Labels: selector}}},
			node:     &node4,
			services: []*api.Service{{Spec: api.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region", "zone"},
			test:     "service pod on different node, multiple labels, all match",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrServiceAffinityViolated}
	for _, test := range tests {
		testIt := func(skipPrecompute bool) {
			nodes := []api.Node{node1, node2, node3, node4, node5}
			nodeInfo := schedulercache.NewNodeInfo()
			nodeInfo.SetNode(test.node)
			nodeInfoMap := map[string]*schedulercache.NodeInfo{test.node.Name: nodeInfo}
			// Reimplementing the logic that the scheduler implements: Any time it makes a predicate, it registers any precomputations.
			predicate, precompute := NewServiceAffinityPredicate(algorithm.FakePodLister(test.pods), algorithm.FakeServiceLister(test.services), FakeNodeListInfo(nodes), test.labels)
			// Register a precomputation or Rewrite the precomputation to a no-op, depending on the state we want to test.
			RegisterPredicatePrecomputation("checkServiceAffinity-unitTestPredicate", func(pm *predicateMetadata) {
				if !skipPrecompute {
					precompute(pm)
				}
			})
			if pmeta, ok := (PredicateMetadata(test.pod, nodeInfoMap)).(*predicateMetadata); ok {
				fits, reasons, err := predicate(test.pod, pmeta, nodeInfo)
				if err != nil {
					t.Errorf("%s: unexpected error: %v", test.test, err)
				}
				if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
					t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
				}
				if fits != test.fits {
					t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
				}
			} else {
				t.Errorf("Error casting.")
			}
		}

		testIt(false) // Confirm that the predicate works without precomputed data (resilience)
		testIt(true)  // Confirm that the predicate works with the precomputed data (better performance)
	}
}

func TestEBSVolumeCountConflicts(t *testing.T) {
	oneVolPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp"},
					},
				},
			},
		},
	}
	ebsPVCPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol",
						},
					},
				},
			},
		},
	}
	splitPVCPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "someNonEBSVol",
						},
					},
				},
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol",
						},
					},
				},
			},
		},
	}
	twoVolPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp1"},
					},
				},
				{
					VolumeSource: api.VolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp2"},
					},
				},
			},
		},
	}
	splitVolsPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{},
					},
				},
				{
					VolumeSource: api.VolumeSource{
						AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "svp"},
					},
				},
			},
		},
	}
	nonApplicablePod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						HostPath: &api.HostPathVolumeSource{},
					},
				},
			},
		},
	}
	deletedPVCPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "deletedPVC",
						},
					},
				},
			},
		},
	}
	deletedPVPod := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "pvcWithDeletedPV",
						},
					},
				},
			},
		},
	}
	emptyPod := &api.Pod{
		Spec: api.PodSpec{},
	}

	tests := []struct {
		newPod       *api.Pod
		existingPods []*api.Pod
		maxVols      int
		fits         bool
		test         string
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*api.Pod{twoVolPod, oneVolPod},
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pod's EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*api.Pod{oneVolPod},
			maxVols:      2,
			fits:         false,
			test:         "doesn't fit when node capacity < new pod's EBS volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*api.Pod{twoVolPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores non-EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*api.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts ignore non-EBS volumes",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*api.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count considers PVCs backed by EBS volumes",
		},
		{
			newPod:       splitPVCPod,
			existingPods: []*api.Pod{splitVolsPod, oneVolPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores PVCs not backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*api.Pod{oneVolPod, ebsPVCPod},
			maxVols:      3,
			fits:         false,
			test:         "existing pods' counts considers PVCs backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*api.Pod{oneVolPod, twoVolPod, ebsPVCPod},
			maxVols:      4,
			fits:         true,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*api.Pod{oneVolPod, oneVolPod, ebsPVCPod},
			maxVols:      3,
			fits:         true,
			test:         "the same EBS volumes are not counted multiple times",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*api.Pod{oneVolPod, deletedPVCPod},
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*api.Pod{oneVolPod, deletedPVCPod},
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*api.Pod{oneVolPod, deletedPVPod},
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*api.Pod{oneVolPod, deletedPVPod},
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
	}

	pvInfo := FakePersistentVolumeInfo{
		{
			ObjectMeta: api.ObjectMeta{Name: "someEBSVol"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{
					AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{VolumeID: "ebsVol"},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "someNonEBSVol"},
			Spec: api.PersistentVolumeSpec{
				PersistentVolumeSource: api.PersistentVolumeSource{},
			},
		},
	}

	pvcInfo := FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: api.ObjectMeta{Name: "someEBSVol"},
			Spec:       api.PersistentVolumeClaimSpec{VolumeName: "someEBSVol"},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "someNonEBSVol"},
			Spec:       api.PersistentVolumeClaimSpec{VolumeName: "someNonEBSVol"},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "pvcWithDeletedPV"},
			Spec:       api.PersistentVolumeClaimSpec{VolumeName: "pvcWithDeletedPV"},
		},
	}

	filter := VolumeFilter{
		FilterVolume: func(vol *api.Volume) (string, bool) {
			if vol.AWSElasticBlockStore != nil {
				return vol.AWSElasticBlockStore.VolumeID, true
			}
			return "", false
		},
		FilterPersistentVolume: func(pv *api.PersistentVolume) (string, bool) {
			if pv.Spec.AWSElasticBlockStore != nil {
				return pv.Spec.AWSElasticBlockStore.VolumeID, true
			}
			return "", false
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}

	for _, test := range tests {
		pred := NewMaxPDVolumeCountPredicate(filter, test.maxVols, pvInfo, pvcInfo)
		fits, reasons, err := pred(test.newPod, PredicateMetadata(test.newPod, nil), schedulercache.NewNodeInfo(test.existingPods...))
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected %v, got %v", test.test, test.fits, fits)
		}
	}
}

func getPredicateSignature() (*types.Signature, error) {
	filePath := "./../types.go"
	pkgName := filepath.Dir(filePath)
	builder := parser.New()
	if err := builder.AddDir(pkgName); err != nil {
		return nil, err
	}
	universe, err := builder.FindTypes()
	if err != nil {
		return nil, err
	}
	result, ok := universe[pkgName].Types["FitPredicate"]
	if !ok {
		return nil, fmt.Errorf("FitPredicate type not defined")
	}
	return result.Signature, nil
}

func TestPredicatesRegistered(t *testing.T) {
	var functions []*types.Type

	// Files and directories which predicates may be referenced
	targetFiles := []string{
		"./../../algorithmprovider/defaults/defaults.go", // Default algorithm
		"./../../factory/plugins.go",                     // Registered in init()
		"./../../../../../pkg/",                          // kubernetes/pkg, often used by kubelet or controller
	}

	// List all golang source files under ./predicates/, excluding test files and sub-directories.
	files, err := codeinspector.GetSourceCodeFiles(".")

	if err != nil {
		t.Errorf("unexpected error: %v when listing files in current directory", err)
	}

	// Get all public predicates in files.
	for _, filePath := range files {
		fileFunctions, err := codeinspector.GetPublicFunctions("k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates", filePath)
		if err == nil {
			functions = append(functions, fileFunctions...)
		} else {
			t.Errorf("unexpected error %s when parsing %s", err, filePath)
		}
	}

	predSignature, err := getPredicateSignature()
	if err != nil {
		t.Fatalf("Couldn't get predicates signature")
	}

	// Check if all public predicates are referenced in target files.
	for _, function := range functions {
		// Ignore functions that don't match FitPredicate signature.
		signature := function.Underlying.Signature
		if len(predSignature.Parameters) != len(signature.Parameters) {
			continue
		}
		if len(predSignature.Results) != len(signature.Results) {
			continue
		}
		// TODO: Check exact types of parameters and results.

		args := []string{"-rl", function.Name.Name}
		args = append(args, targetFiles...)

		err := exec.Command("grep", args...).Run()
		if err != nil {
			switch err.Error() {
			case "exit status 2":
				t.Errorf("unexpected error when checking %s", function.Name)
			case "exit status 1":
				t.Errorf("predicate %s is implemented as public but seems not registered or used in any other place",
					function.Name)
			}
		}
	}
}

func newPodWithPort(hostPorts ...int) *api.Pod {
	networkPorts := []api.ContainerPort{}
	for _, port := range hostPorts {
		networkPorts = append(networkPorts, api.ContainerPort{HostPort: int32(port)})
	}
	return &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestRunGeneralPredicates(t *testing.T) {
	resourceTests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		node     *api.Node
		fits     bool
		test     string
		wErr     error
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: &api.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: "machine1"},
				Status:     api.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0)},
			},
			fits: true,
			wErr: nil,
			test: "no resources/port/host requested always fits",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 10}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: "machine1"},
				Status:     api.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0)},
			},
			fits: false,
			wErr: nil,
			reasons: []algorithm.PredicateFailureReason{
				NewInsufficientResourceError(api.ResourceCPU, 8, 5, 10),
				NewInsufficientResourceError(api.ResourceMemory, 10, 19, 20),
			},
			test: "not enough cpu and memory resource",
		},
		{
			pod: &api.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			node: &api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0)}},
			fits: true,
			wErr: nil,
			test: "no resources/port/host requested always fits on GPU machine",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 3, Memory: 1, NvidiaGPU: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 10, NvidiaGPU: 1})),
			node:    &api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0)}},
			fits:    false,
			wErr:    nil,
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(api.ResourceNvidiaGPU, 1, 1, 1)},
			test:    "not enough GPU resource",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 3, Memory: 1, NvidiaGPU: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 10, NvidiaGPU: 0})),
			node: &api.Node{Status: api.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0)}},
			fits: true,
			wErr: nil,
			test: "enough GPU resource",
		},
		{
			pod: &api.Pod{
				Spec: api.PodSpec{
					NodeName: "machine2",
				},
			},
			nodeInfo: schedulercache.NewNodeInfo(),
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: "machine1"},
				Status:     api.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0)},
			},
			fits:    false,
			wErr:    nil,
			reasons: []algorithm.PredicateFailureReason{ErrPodNotMatchHostName},
			test:    "host not match",
		},
		{
			pod:      newPodWithPort(123),
			nodeInfo: schedulercache.NewNodeInfo(newPodWithPort(123)),
			node: &api.Node{
				ObjectMeta: api.ObjectMeta{Name: "machine1"},
				Status:     api.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0)},
			},
			fits:    false,
			wErr:    nil,
			reasons: []algorithm.PredicateFailureReason{ErrPodNotFitsHostPorts},
			test:    "hostport conflict",
		},
	}
	for _, test := range resourceTests {
		test.nodeInfo.SetNode(test.node)
		fits, reasons, err := GeneralPredicates(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, test.reasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, test.reasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestInterPodAffinity(t *testing.T) {
	podLabel := map[string]string{"service": "securityscan"}
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	podLabel2 := map[string]string{"security": "S1"}
	node1 := api.Node{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: labels1}}
	tests := []struct {
		pod  *api.Pod
		pods []*api.Pod
		node *api.Node
		fits bool
		test string
	}{
		{
			pod:  new(api.Pod),
			node: &node1,
			fits: true,
			test: "A pod that has no required pod affinity scheduling rules can schedule onto a node with no existing pods",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
										}]
									},
								"topologyKey": "region"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies with requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using In operator that matches the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "NotIn",
										"values": ["securityscan3", "value3"]
									}]
								},
								"topologyKey": "region"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the pod with requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using not in operator in labelSelector that matches the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"namespaces":["DiffNameSpace"]
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel, Namespace: "ns"}}},
			node: &node1,
			fits: false,
			test: "Does not satisfy the PodAffinity with labelSelector because of diff Namespace",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["antivirusscan", "value2"]
									}]
								}
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: false,
			test: "Doesn't satisfy the PodAffinity because of unmatching labelSelector with the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [
								{
									"labelSelector": {
										"matchExpressions": [{
											"key": "service",
											"operator": "Exists"
										}, {
											"key": "wrongkey",
											"operator": "DoesNotExist"
										}]
									},
									"topologyKey": "region"
								}, {
									"labelSelector": {
										"matchExpressions": [{
											"key": "service",
											"operator": "In",
											"values": ["securityscan"]
										}, {
											"key": "service",
											"operator": "NotIn",
											"values": ["WrongValue"]
										}]
									},
									"topologyKey": "region"
								}
							]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity with different label Operators in multiple RequiredDuringSchedulingIgnoredDuringExecution ",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [
								{
									"labelSelector": {
										"matchExpressions": [{
											"key": "service",
											"operator": "Exists"
										}, {
											"key": "wrongkey",
											"operator": "DoesNotExist"
										}]
									},
									"topologyKey": "region"
								}, {
									"labelSelector": {
										"matchExpressions": [{
											"key": "service",
											"operator": "In",
											"values": ["securityscan2"]
										}, {
											"key": "service",
											"operator": "NotIn",
											"values": ["WrongValue"]
										}]
									},
									"topologyKey": "region"
								}
							]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: false,
			test: "The labelSelector requirements(items of matchExpressions) are ANDed, the pod cannot schedule onto the node because one of the matchExpression item don't match.",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "region"
							}]
						},
						"podAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["antivirusscan", "value2"]
									}]
								},
								"topologyKey": "node"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity and PodAntiAffinity with the existing pod",
		},
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		//{
		//	 pod: &api.Pod{
		//		ObjectMeta: api.ObjectMeta{
		//			Labels: podLabel2,
		//			Annotations: map[string]string{
		//				api.AffinityAnnotationKey: `
		//				{"podAffinity": {
		//					"requiredDuringSchedulingRequiredDuringExecution": [
		//						{
		//							"labelSelector": {
		//								"matchExpressions": [{
		//									"key": "service",
		//									"operator": "Exists"
		//								}, {
		//									"key": "wrongkey",
		//									"operator": "DoesNotExist"
		//								}]
		//							},
		//							"topologyKey": "region"
		//						}, {
		//							"labelSelector": {
		//								"matchExpressions": [{
		//									"key": "service",
		//									"operator": "In",
		//									"values": ["securityscan"]
		//								}, {
		//									"key": "service",
		//									"operator": "NotIn",
		//									"values": ["WrongValue"]
		//								}]
		//							},
		//							"topologyKey": "region"
		//						}
		//					]
		//				}}`,
		//			},
		//		},
		//	},
		//	pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podlabel}}},
		//	node: &node1,
		//	fits: true,
		//	test: "satisfies the PodAffinity with different Label Operators in multiple RequiredDuringSchedulingRequiredDuringExecution ",
		//},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "region"
							}]
						},
						"podAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["antivirusscan", "value2"]
									}]
								},
								"topologyKey": "node"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"},
				ObjectMeta: api.ObjectMeta{Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"PodAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["antivirusscan", "value2"]
									}]
								},
								"topologyKey": "node"
							}]
						}}`,
					}},
			}},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity and PodAntiAffinity and PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel2,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "region"
							}]
						},
						"podAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "zone"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: false,
			test: "satisfies the PodAffinity but doesn't satisfies the PodAntiAffinity with the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "region"
							}]
						},
						"podAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["antivirusscan", "value2"]
									}]
								},
								"topologyKey": "node"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"},
				ObjectMeta: api.ObjectMeta{Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"PodAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "zone"
							}]
						}}`,
					}},
			}},
			node: &node1,
			fits: false,
			test: "satisfies the PodAffinity and PodAntiAffinity but doesn't satisfies PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "NotIn",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "region"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine2"}, ObjectMeta: api.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: false,
			test: "pod matches its own Label in PodAffinity and that matches the existing pod Labels",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel,
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"},
				ObjectMeta: api.ObjectMeta{Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"PodAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["securityscan", "value2"]
									}]
								},
								"topologyKey": "zone"
							}]
						}}`,
					}},
			}},
			node: &node1,
			fits: false,
			test: "verify that PodAntiAffinity from existing pod is respected when pod has no AntiAffinity constraints. doesn't satisfy PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabel,
				},
			},
			pods: []*api.Pod{{Spec: api.PodSpec{NodeName: "machine1"},
				ObjectMeta: api.ObjectMeta{Labels: podLabel,
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
                        {"PodAntiAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": [{
                                "labelSelector": {
                                    "matchExpressions": [{
                                        "key": "service",
                                        "operator": "NotIn",
                                        "values": ["securityscan", "value2"]
                                    }]
                                },
                                "topologyKey": "zone"
                            }]
                        }}`,
					}},
			}},
			node: &node1,
			fits: true,
			test: "verify that PodAntiAffinity from existing pod is respected when pod has no AntiAffinity constraints. satisfy PodAntiAffinity symmetry with the existing pod",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch}

	for _, test := range tests {
		node := test.node
		var podsOnNode []*api.Pod
		for _, pod := range test.pods {
			if pod.Spec.NodeName == node.Name {
				podsOnNode = append(podsOnNode, pod)
			}
		}

		fit := PodAffinityChecker{
			info:           FakeNodeInfo(*node),
			podLister:      algorithm.FakePodLister(test.pods),
			failureDomains: priorityutil.Topologies{DefaultKeys: strings.Split(api.DefaultFailureDomains, ",")},
		}
		nodeInfo := schedulercache.NewNodeInfo(podsOnNode...)
		nodeInfo.SetNode(test.node)
		nodeInfoMap := map[string]*schedulercache.NodeInfo{test.node.Name: nodeInfo}
		fits, reasons, err := fit.InterPodAffinityMatches(test.pod, PredicateMetadata(test.pod, nodeInfoMap), nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected %v got %v", test.test, test.fits, fits)
		}
	}
}

func TestInterPodAffinityWithMultipleNodes(t *testing.T) {
	podLabelA := map[string]string{
		"foo": "bar",
	}
	labelRgChina := map[string]string{
		"region": "China",
	}
	labelRgChinaAzAz1 := map[string]string{
		"region": "China",
		"az":     "az1",
	}
	labelRgIndia := map[string]string{
		"region": "India",
	}
	tests := []struct {
		pod   *api.Pod
		pods  []*api.Pod
		nodes []api.Node
		fits  map[string]bool
		test  string
	}{
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "foo",
										"operator": "In",
										"values": ["bar"]
									}]
								},
								"topologyKey": "region"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{
				{Spec: api.PodSpec{NodeName: "machine1"}, ObjectMeta: api.ObjectMeta{Labels: podLabelA}},
			},
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "machine1", Labels: labelRgChina}},
				{ObjectMeta: api.ObjectMeta{Name: "machine2", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: api.ObjectMeta{Name: "machine3", Labels: labelRgIndia}},
			},
			fits: map[string]bool{
				"machine1": true,
				"machine2": true,
				"machine3": false,
			},
			test: "A pod can be scheduled onto all the nodes that have the same topology key & label value with one of them has an existing pod that match the affinity rules",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{
							"nodeAffinity": {
								"requiredDuringSchedulingIgnoredDuringExecution": {
									"nodeSelectorTerms": [{
										"matchExpressions": [{
											"key": "hostname",
											"operator": "NotIn",
											"values": ["h1"]
										}]
									}]
								}
							},
							"podAffinity": {
								"requiredDuringSchedulingIgnoredDuringExecution": [{
									"labelSelector": {
										"matchExpressions": [{
											"key": "foo",
											"operator": "In",
											"values": ["abc"]
										}]
									},
									"topologyKey": "region"
								}]
							}
						}`,
					},
				},
			},
			pods: []*api.Pod{
				{Spec: api.PodSpec{NodeName: "nodeA"}, ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
				{Spec: api.PodSpec{NodeName: "nodeB"}, ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "def"}}},
			},
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "nodeA", Labels: map[string]string{"region": "r1", "hostname": "h1"}}},
				{ObjectMeta: api.ObjectMeta{Name: "nodeB", Labels: map[string]string{"region": "r1", "hostname": "h2"}}},
			},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": true,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA does not satisfy node affinity rule, but has an existing pod that match the inter pod affinity rule. The pod can be scheduled onto nodeB.",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector": {
									"matchExpressions": [{
										"key": "foo",
										"operator": "In",
										"values": ["bar"]
									}]
								},
								"topologyKey": "zone"
							}]
						}}`,
					},
				},
			},
			pods: []*api.Pod{},
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "nodeA", Labels: map[string]string{"zone": "az1", "hostname": "h1"}}},
				{ObjectMeta: api.ObjectMeta{Name: "nodeB", Labels: map[string]string{"zone": "az2", "hostname": "h2"}}},
			},
			fits: map[string]bool{
				"nodeA": true,
				"nodeB": true,
			},
			test: "The affinity rule is to schedule all of the pods of this collection to the same zone. The first pod of the collection " +
				"should not be blocked from being scheduled onto any node, even there's no existing pod that match the rule anywhere.",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{
							"podAntiAffinity": {
								"requiredDuringSchedulingIgnoredDuringExecution": [{
									"labelSelector": {
										"matchExpressions": [{
											"key": "foo",
											"operator": "In",
											"values": ["abc"]
										}]
									},
									"topologyKey": "region"
								}]
							}
						}`,
					},
				},
			},
			pods: []*api.Pod{
				{Spec: api.PodSpec{NodeName: "nodeA"}, ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
			},
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "nodeA", Labels: map[string]string{"region": "r1", "hostname": "nodeA"}}},
				{ObjectMeta: api.ObjectMeta{Name: "nodeB", Labels: map[string]string{"region": "r1", "hostname": "nodeB"}}},
			},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA and nodeB.",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.AffinityAnnotationKey: `
						{
							"podAntiAffinity": {
								"requiredDuringSchedulingIgnoredDuringExecution": [{
									"labelSelector": {
										"matchExpressions": [{
											"key": "foo",
											"operator": "In",
											"values": ["abc"]
										}]
									},
									"topologyKey": "region"
								}]
							}
						}`,
					},
				},
			},
			pods: []*api.Pod{
				{Spec: api.PodSpec{NodeName: "nodeA"}, ObjectMeta: api.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
			},
			nodes: []api.Node{
				{ObjectMeta: api.ObjectMeta{Name: "nodeA", Labels: labelRgChina}},
				{ObjectMeta: api.ObjectMeta{Name: "nodeB", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: api.ObjectMeta{Name: "nodeC", Labels: labelRgIndia}},
			},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
				"nodeC": true,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA and nodeB but can be schedulerd onto nodeC",
		},
	}
	affinityExpectedFailureReasons := []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch}
	selectorExpectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeSelectorNotMatch}

	for _, test := range tests {
		nodeListInfo := FakeNodeListInfo(test.nodes)
		for _, node := range test.nodes {
			var podsOnNode []*api.Pod
			for _, pod := range test.pods {
				if pod.Spec.NodeName == node.Name {
					podsOnNode = append(podsOnNode, pod)
				}
			}

			testFit := PodAffinityChecker{
				info:           nodeListInfo,
				podLister:      algorithm.FakePodLister(test.pods),
				failureDomains: priorityutil.Topologies{DefaultKeys: strings.Split(api.DefaultFailureDomains, ",")},
			}
			nodeInfo := schedulercache.NewNodeInfo(podsOnNode...)
			nodeInfo.SetNode(&node)
			nodeInfoMap := map[string]*schedulercache.NodeInfo{node.Name: nodeInfo}
			fits, reasons, err := testFit.InterPodAffinityMatches(test.pod, PredicateMetadata(test.pod, nodeInfoMap), nodeInfo)
			if err != nil {
				t.Errorf("%s: unexpected error %v", test.test, err)
			}
			if !fits && !reflect.DeepEqual(reasons, affinityExpectedFailureReasons) {
				t.Errorf("%s: unexpected failure reasons: %v", test.test, reasons)
			}
			affinity, err := api.GetAffinityFromPodAnnotations(test.pod.ObjectMeta.Annotations)
			if err != nil {
				t.Errorf("%s: unexpected error: %v", test.test, err)
			}
			if affinity != nil && affinity.NodeAffinity != nil {
				nodeInfo := schedulercache.NewNodeInfo()
				nodeInfo.SetNode(&node)
				nodeInfoMap := map[string]*schedulercache.NodeInfo{node.Name: nodeInfo}
				fits2, reasons, err := PodSelectorMatches(test.pod, PredicateMetadata(test.pod, nodeInfoMap), nodeInfo)
				if err != nil {
					t.Errorf("%s: unexpected error: %v", test.test, err)
				}
				if !fits2 && !reflect.DeepEqual(reasons, selectorExpectedFailureReasons) {
					t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, selectorExpectedFailureReasons)
				}
				fits = fits && fits2
			}

			if fits != test.fits[node.Name] {
				t.Errorf("%s: expected %v for %s got %v", test.test, test.fits[node.Name], node.Name, fits)
			}
		}
	}
}

func TestPodToleratesTaints(t *testing.T) {
	podTolerateTaintsTests := []struct {
		pod  *api.Pod
		node api.Node
		fits bool
		test string
	}{
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod0",
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user1",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: false,
			test: "a pod having no tolerations can't be scheduled onto a node with nonempty taints",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod1",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user1",
							"effect": "NoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod1:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user1",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: true,
			test: "a pod which can be scheduled on a dedicated node assigned to user1 with effect NoSchedule",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "dedicated",
							"operator": "Equal",
							"value": "user2",
							"effect": "NoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user1",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: false,
			test: "a pod which can't be scheduled on a dedicated node assigned to user2 with effect NoSchedule",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "foo",
							"operator": "Exists",
							"effect": "NoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "foo",
							"value": "bar",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: true,
			test: "a pod can be scheduled onto the node, with a toleration uses operator Exists that tolerates the taints on the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "dedicated",
							"operator": "Equal",
							"value": "user2",
							"effect": "NoSchedule"
						}, {
							"key": "foo",
							"operator": "Exists",
							"effect": "NoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user2",
							"effect": "NoSchedule"
						}, {
							"key": "foo",
							"value": "bar",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: true,
			test: "a pod has multiple tolerations, node has multiple taints, all the taints are tolerated, pod can be scheduled onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "foo",
							"operator": "Equal",
							"value": "bar",
							"effect": "PreferNoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "foo",
							"value": "bar",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: false,
			test: "a pod has a toleration that keys and values match the taint on the node, but (non-empty) effect doesn't match, " +
				"can't be scheduled onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "foo",
							"operator": "Equal",
							"value": "bar"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "foo",
							"value": "bar",
							"effect": "NoSchedule"
						}]`,
					},
				},
			},
			fits: true,
			test: "The pod has a toleration that keys and values match the taint on the node, the effect of toleration is empty, " +
				"and the effect of taint is NoSchedule. Pod can be scheduled onto the node",
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: "pod2",
					Annotations: map[string]string{
						api.TolerationsAnnotationKey: `
						[{
							"key": "dedicated",
							"operator": "Equal",
							"value": "user2",
							"effect": "NoSchedule"
						}]`,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{{Image: "pod2:V1"}},
				},
			},
			node: api.Node{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						api.TaintsAnnotationKey: `
						[{
							"key": "dedicated",
							"value": "user1",
							"effect": "PreferNoSchedule"
						}]`,
					},
				},
			},
			fits: true,
			test: "The pod has a toleration that key and value don't match the taint on the node, " +
				"but the effect of taint on node is PreferNochedule. Pod can be scheduled onto the node",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrTaintsTolerationsNotMatch}

	for _, test := range podTolerateTaintsTests {
		nodeInfo := schedulercache.NewNodeInfo()
		nodeInfo.SetNode(&test.node)
		fits, reasons, err := PodToleratesNodeTaints(test.pod, PredicateMetadata(test.pod, nil), nodeInfo)
		if err != nil {
			t.Errorf("%s, unexpected error: %v", test.test, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s, unexpected failure reason: %v, want: %v", test.test, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s, expected: %v got %v", test.test, test.fits, fits)
		}
	}
}

func makeEmptyNodeInfo(node *api.Node) *schedulercache.NodeInfo {
	nodeInfo := schedulercache.NewNodeInfo()
	nodeInfo.SetNode(node)
	return nodeInfo
}

func TestPodSchedulesOnNodeWithMemoryPressureCondition(t *testing.T) {
	// specify best-effort pod
	bestEffortPod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					// no requirements -> best effort pod
					Resources: api.ResourceRequirements{},
				},
			},
		},
	}

	// specify non-best-effort pod
	nonBestEffortPod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					// at least one requirement -> burstable pod
					Resources: api.ResourceRequirements{
						Requests: makeAllocatableResources(100, 100, 100, 100, 0),
					},
				},
			},
		},
	}

	// specify a node with no memory pressure condition on
	noMemoryPressureNode := &api.Node{
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:   "Ready",
					Status: "True",
				},
			},
		},
	}

	// specify a node with memory pressure condition on
	memoryPressureNode := &api.Node{
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:   "MemoryPressure",
					Status: "True",
				},
			},
		},
	}

	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		name     string
	}{
		{
			pod:      bestEffortPod,
			nodeInfo: makeEmptyNodeInfo(noMemoryPressureNode),
			fits:     true,
			name:     "best-effort pod schedulable on node without memory pressure condition on",
		},
		{
			pod:      bestEffortPod,
			nodeInfo: makeEmptyNodeInfo(memoryPressureNode),
			fits:     false,
			name:     "best-effort pod not schedulable on node with memory pressure condition on",
		},
		{
			pod:      nonBestEffortPod,
			nodeInfo: makeEmptyNodeInfo(memoryPressureNode),
			fits:     true,
			name:     "non best-effort pod schedulable on node with memory pressure condition on",
		},
		{
			pod:      nonBestEffortPod,
			nodeInfo: makeEmptyNodeInfo(noMemoryPressureNode),
			fits:     true,
			name:     "non best-effort pod schedulable on node without memory pressure condition on",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeUnderMemoryPressure}

	for _, test := range tests {
		fits, reasons, err := CheckNodeMemoryPressurePredicate(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.name, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected %v got %v", test.name, test.fits, fits)
		}
	}
}

func TestPodSchedulesOnNodeWithDiskPressureCondition(t *testing.T) {
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
				},
			},
		},
	}

	// specify a node with no disk pressure condition on
	noPressureNode := &api.Node{
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:   "Ready",
					Status: "True",
				},
			},
		},
	}

	// specify a node with pressure condition on
	pressureNode := &api.Node{
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:   "DiskPressure",
					Status: "True",
				},
			},
		},
	}

	tests := []struct {
		pod      *api.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		name     string
	}{
		{
			pod:      pod,
			nodeInfo: makeEmptyNodeInfo(noPressureNode),
			fits:     true,
			name:     "pod schedulable on node without pressure condition on",
		},
		{
			pod:      pod,
			nodeInfo: makeEmptyNodeInfo(pressureNode),
			fits:     false,
			name:     "pod not schedulable on node with pressure condition on",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeUnderDiskPressure}

	for _, test := range tests {
		fits, reasons, err := CheckNodeDiskPressurePredicate(test.pod, PredicateMetadata(test.pod, nil), test.nodeInfo)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.name, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.name, reasons, expectedFailureReasons)
		}
		if fits != test.fits {
			t.Errorf("%s: expected %v got %v", test.name, test.fits, fits)
		}
	}
}
