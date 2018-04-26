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
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
)

var (
	extendedResourceA = v1.ResourceName("example.com/aaa")
	extendedResourceB = v1.ResourceName("example.com/bbb")
	hugePageResourceA = v1helper.HugePageResourceName(resource.MustParse("2Mi"))
)

func makeResources(milliCPU, memory, nvidiaGPUs, pods, extendedA, storage, hugePageA int64) v1.NodeResources {
	return v1.NodeResources{
		Capacity: v1.ResourceList{
			v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
			v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
			v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
			v1.ResourceNvidiaGPU:        *resource.NewQuantity(nvidiaGPUs, resource.DecimalSI),
			extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
			v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
			hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
		},
	}
}

func makeAllocatableResources(milliCPU, memory, nvidiaGPUs, pods, extendedA, storage, hugePageA int64) v1.ResourceList {
	return v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(pods, resource.DecimalSI),
		v1.ResourceNvidiaGPU:        *resource.NewQuantity(nvidiaGPUs, resource.DecimalSI),
		extendedResourceA:           *resource.NewQuantity(extendedA, resource.DecimalSI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(storage, resource.BinarySI),
		hugePageResourceA:           *resource.NewQuantity(hugePageA, resource.BinarySI),
	}
}

func newResourcePod(usage ...schedulercache.Resource) *v1.Pod {
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

func newResourceInitPod(pod *v1.Pod, usage ...schedulercache.Resource) *v1.Pod {
	pod.Spec.InitContainers = newResourcePod(usage...).Spec.Containers
	return pod
}

func PredicateMetadata(p *v1.Pod, nodeInfo map[string]*schedulercache.NodeInfo) algorithm.PredicateMetadata {
	pm := PredicateMetadataFactory{schedulertesting.FakePodLister{p}}
	return pm.GetMetadata(p, nodeInfo)
}

func TestPodFitsResources(t *testing.T) {
	enoughPodsTests := []struct {
		pod                      *v1.Pod
		nodeInfo                 *schedulercache.NodeInfo
		fits                     bool
		test                     string
		reasons                  []algorithm.PredicateFailureReason
		ignoredExtendedResources sets.String
	}{
		{
			pod: &v1.Pod{},
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
				NewInsufficientResourceError(v1.ResourceCPU, 1, 10, 10),
				NewInsufficientResourceError(v1.ResourceMemory, 1, 20, 20),
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 3, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to init container cpu",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 3, Memory: 1}, schedulercache.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to highest init container cpu",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 3}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to init container memory",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 3, 19, 20)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}), schedulercache.Resource{MilliCPU: 1, Memory: 3}, schedulercache.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			test:    "too many resources fails due to highest init container memory",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 3, 19, 20)},
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
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 2, 9, 10)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "one resource cpu fits",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 2, 19, 20)},
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
			pod:      newResourcePod(schedulercache.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(newResourcePod(schedulercache.Resource{})),
			fits:     true,
			test:     "extended resource fits",
		},
		{
			pod:      newResourceInitPod(newResourcePod(schedulercache.Resource{}), schedulercache.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(newResourcePod(schedulercache.Resource{})),
			fits:     true,
			test:     "extended resource fits for init container",
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			fits:    false,
			test:    "extended resource capacity enforced",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 10, 0, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			fits:    false,
			test:    "extended resource capacity enforced for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 10, 0, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			fits:    false,
			test:    "extended resource allocatable enforced",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 1, 5, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			fits:    false,
			test:    "extended resource allocatable enforced for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 1, 5, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits:    false,
			test:    "extended resource allocatable enforced for multiple containers",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 6, 2, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits: true,
			test: "extended resource allocatable admits multiple init containers",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 6}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits:    false,
			test:    "extended resource allocatable enforced for multiple init containers",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 6, 2, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			test:    "extended resource allocatable enforced for unknown resource",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceB, 1, 0, 0)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			test:    "extended resource allocatable enforced for unknown resource for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(extendedResourceB, 1, 0, 0)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			fits:    false,
			test:    "hugepages resource capacity enforced",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 10, 0, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{}),
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			fits:    false,
			test:    "hugepages resource capacity enforced for init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 10, 0, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}},
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 2}})),
			fits:    false,
			test:    "hugepages resource allocatable enforced for multiple containers",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 6, 2, 5)},
		},
		{
			pod: newResourcePod(
				schedulercache.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 0, Memory: 0})),
			fits: true,
			ignoredExtendedResources: sets.NewString(string(extendedResourceB)),
			test: "skip checking ignored extended resource",
		},
	}

	for _, test := range enoughPodsTests {
		node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 5, 20, 5).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 5, 20, 5)}}
		test.nodeInfo.SetNode(&node)
		RegisterPredicateMetadataProducerWithExtendedResourceOptions(test.ignoredExtendedResources)
		meta := PredicateMetadata(test.pod, nil)
		fits, reasons, err := PodFitsResources(test.pod, meta, test.nodeInfo)
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
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 10, Memory: 20})),
			fits:    false,
			test:    "even without specified resources predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 5})),
			fits:    false,
			test:    "even if both resources fit predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "even for equal edge case predicate fails when there's no space for additional pod",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 1}), schedulercache.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			test:    "even for equal edge case predicate fails when there's no space for additional pod due to init container",
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
	}
	for _, test := range notEnoughPodsTests {
		node := v1.Node{Status: v1.NodeStatus{Capacity: v1.ResourceList{}, Allocatable: makeAllocatableResources(10, 20, 0, 1, 0, 0, 0)}}
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

	storagePodsTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 10, Memory: 10})),
			fits: false,
			test: "due to container scratch disk",
			reasons: []algorithm.PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceCPU, 1, 10, 10),
			},
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 2, Memory: 10})),
			fits: true,
			test: "pod fit",
		},
		{
			pod: newResourcePod(schedulercache.Resource{EphemeralStorage: 25}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 2, Memory: 2})),
			fits: false,
			test: "storage ephemeral local storage request exceeds allocatable",
			reasons: []algorithm.PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceEphemeralStorage, 25, 0, 20),
			},
		},
		{
			pod: newResourcePod(schedulercache.Resource{EphemeralStorage: 10}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 2, Memory: 2})),
			fits: true,
			test: "pod fits",
		},
	}

	for _, test := range storagePodsTests {
		node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 5, 20, 5).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 5, 20, 5)}}
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
		pod  *v1.Pod
		node *v1.Node
		fits bool
		test string
	}{
		{
			pod:  &v1.Pod{},
			node: &v1.Node{},
			fits: true,
			test: "no host specified",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "foo",
				},
			},
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "foo",
				},
			},
			fits: true,
			test: "host matches",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "bar",
				},
			},
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
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

func newPod(host string, hostPortInfos ...string) *v1.Pod {
	networkPorts := []v1.ContainerPort{}
	for _, portInfo := range hostPortInfos {
		splited := strings.Split(portInfo, "/")
		hostPort, _ := strconv.Atoi(splited[2])

		networkPorts = append(networkPorts, v1.ContainerPort{
			HostIP:   splited[1],
			HostPort: int32(hostPort),
			Protocol: v1.Protocol(splited[0]),
		})
	}
	return &v1.Pod{
		Spec: v1.PodSpec{
			NodeName: host,
			Containers: []v1.Container{
				{
					Ports: networkPorts,
				},
			},
		},
	}
}

func TestPodFitsHostPorts(t *testing.T) {
	tests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		fits     bool
		test     string
	}{
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(),
			fits:     true,
			test:     "nothing running",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/9090")),
			fits: true,
			test: "other port",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			fits: false,
			test: "same udp port",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			fits: false,
			test: "same tcp port",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.2/8080")),
			fits: true,
			test: "different host ip",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			fits: true,
			test: "different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8000", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			fits: false,
			test: "second udp port conflict",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8081")),
			fits: false,
			test: "first tcp port conflict",
		},
		{
			pod: newPod("m1", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			fits: false,
			test: "first tcp port conflict due to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "TCP/10.0.10.10/8001", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			fits: false,
			test: "TCP hostPort conflict due to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			fits: false,
			test: "second tcp port conflict to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			fits: true,
			test: "second different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulercache.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001", "UDP/0.0.0.0/8001")),
			fits: false,
			test: "UDP hostPort conflict due to 0.0.0.0 hostIP",
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

func TestGCEDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "foo",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
						PDName: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&v1.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&v1.Pod{}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state"},
		{&v1.Pod{Spec: volState}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state"},
		{&v1.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state"},
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
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: "foo",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
						VolumeID: "bar",
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&v1.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&v1.Pod{}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state"},
		{&v1.Pod{Spec: volState}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state"},
		{&v1.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state"},
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
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
						CephMonitors: []string{"a", "b"},
						RBDPool:      "foo",
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					RBD: &v1.RBDVolumeSource{
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
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&v1.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&v1.Pod{}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state"},
		{&v1.Pod{Spec: volState}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state"},
		{&v1.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state"},
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

func TestISCSIDiskConflicts(t *testing.T) {
	volState := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					ISCSI: &v1.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1:3260",
						IQN:          "iqn.2016-12.server:storage.target01",
						FSType:       "ext4",
						Lun:          0,
					},
				},
			},
		},
	}
	volState2 := v1.PodSpec{
		Volumes: []v1.Volume{
			{
				VolumeSource: v1.VolumeSource{
					ISCSI: &v1.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1:3260",
						IQN:          "iqn.2017-12.server:storage.target01",
						FSType:       "ext4",
						Lun:          0,
					},
				},
			},
		},
	}
	tests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		isOk     bool
		test     string
	}{
		{&v1.Pod{}, schedulercache.NewNodeInfo(), true, "nothing"},
		{&v1.Pod{}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "one state"},
		{&v1.Pod{Spec: volState}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), false, "same state"},
		{&v1.Pod{Spec: volState2}, schedulercache.NewNodeInfo(&v1.Pod{Spec: volState}), true, "different state"},
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

// TODO: Add test case for RequiredDuringSchedulingRequiredDuringExecution after it's implemented.
func TestPodFitsSelector(t *testing.T) {
	tests := []struct {
		pod    *v1.Pod
		labels map[string]string
		fits   bool
		test   string
	}{
		{
			pod:  &v1.Pod{},
			fits: true,
			test: "no selector",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
				},
			},
			fits: false,
			test: "missing labels",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"bar", "value2"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "kernel-version",
												Operator: v1.NodeSelectorOpGt,
												Values:   []string{"0204"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "mem-type",
												Operator: v1.NodeSelectorOpNotIn,
												Values:   []string{"DDR", "DDR2"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "GPU",
												Operator: v1.NodeSelectorOpExists,
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: nil,
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{},
									},
								},
							},
						},
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
			pod: &v1.Pod{},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with no Affinity will schedule onto a node",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: nil,
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "GPU",
												Operator: v1.NodeSelectorOpExists,
											}, {
												Key:      "GPU",
												Operator: v1.NodeSelectorOpNotIn,
												Values:   []string{"AMD", "INTER"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "GPU",
												Operator: v1.NodeSelectorOpExists,
											}, {
												Key:      "GPU",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"AMD", "INTER"},
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"bar", "value2"},
											},
										},
									},
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "diffkey",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"wrong", "value2"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			test: "Pod with multiple NodeSelectorTerms ORed in affinity, matches the node's labels and will schedule onto the node",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpExists,
											},
										},
									},
								},
							},
						},
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
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeSelector: map[string]string{
						"foo": "bar",
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpExists,
											},
										},
									},
								},
							},
						},
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
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpNotIn,
												Values:   []string{"invalid value: ___@#$%^"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: false,
			test: "Pod with an invalid value in Affinity term won't be scheduled onto the node",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeSelectorNotMatch}

	for _, test := range tests {
		node := v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: test.labels}}
		nodeInfo := schedulercache.NewNodeInfo()
		nodeInfo.SetNode(&node)

		fits, reasons, err := PodMatchNodeSelector(test.pod, PredicateMetadata(test.pod, nil), nodeInfo)
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
		pod      *v1.Pod
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
		node := v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: label}}
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
	node1 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: labels1}}
	node2 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: labels2}}
	node3 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: labels3}}
	node4 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine4", Labels: labels4}}
	node5 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine5", Labels: labels4}}
	tests := []struct {
		pod      *v1.Pod
		pods     []*v1.Pod
		services []*v1.Service
		node     *v1.Node
		labels   []string
		fits     bool
		test     string
	}{
		{
			pod:    new(v1.Pod),
			node:   &node1,
			fits:   true,
			labels: []string{"region"},
			test:   "nothing scheduled",
		},
		{
			pod:    &v1.Pod{Spec: v1.PodSpec{NodeSelector: map[string]string{"region": "r1"}}},
			node:   &node1,
			fits:   true,
			labels: []string{"region"},
			test:   "pod with region label match",
		},
		{
			pod:    &v1.Pod{Spec: v1.PodSpec{NodeSelector: map[string]string{"region": "r2"}}},
			node:   &node1,
			fits:   false,
			labels: []string{"region"},
			test:   "pod with region label mismatch",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on same node",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine2"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service pod on different node, region match",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service pod on different node, region mismatch",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns2"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "service in different namespace, region mismatch",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns2"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			fits:     true,
			labels:   []string{"region"},
			test:     "pod in different namespace, region mismatch",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine3"}, ObjectMeta: metav1.ObjectMeta{Labels: selector, Namespace: "ns1"}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}, ObjectMeta: metav1.ObjectMeta{Namespace: "ns1"}}},
			fits:     false,
			labels:   []string{"region"},
			test:     "service and pod in same namespace, region mismatch",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine2"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node1,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			fits:     false,
			labels:   []string{"region", "zone"},
			test:     "service pod on different node, multiple labels, not all match",
		},
		{
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Labels: selector}},
			pods:     []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine5"}, ObjectMeta: metav1.ObjectMeta{Labels: selector}}},
			node:     &node4,
			services: []*v1.Service{{Spec: v1.ServiceSpec{Selector: selector}}},
			fits:     true,
			labels:   []string{"region", "zone"},
			test:     "service pod on different node, multiple labels, all match",
		},
	}
	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrServiceAffinityViolated}
	for _, test := range tests {
		testIt := func(skipPrecompute bool) {
			nodes := []v1.Node{node1, node2, node3, node4, node5}
			nodeInfo := schedulercache.NewNodeInfo()
			nodeInfo.SetNode(test.node)
			nodeInfoMap := map[string]*schedulercache.NodeInfo{test.node.Name: nodeInfo}
			// Reimplementing the logic that the scheduler implements: Any time it makes a predicate, it registers any precomputations.
			predicate, precompute := NewServiceAffinityPredicate(schedulertesting.FakePodLister(test.pods), schedulertesting.FakeServiceLister(test.services), FakeNodeListInfo(nodes), test.labels)
			// Register a precomputation or Rewrite the precomputation to a no-op, depending on the state we want to test.
			RegisterPredicateMetadataProducer("ServiceAffinityMetaProducer", func(pm *predicateMetadata) {
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
	oneVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ovp"},
					},
				},
			},
		},
	}
	ebsPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol",
						},
					},
				},
			},
		},
	}
	splitPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someNonEBSVol",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "someEBSVol",
						},
					},
				},
			},
		},
	}
	twoVolPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp1"},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "tvp2"},
					},
				},
			},
		},
	}
	splitVolsPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "svp"},
					},
				},
			},
		},
	}
	nonApplicablePod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{},
					},
				},
			},
		},
	}
	deletedPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "deletedPVC",
						},
					},
				},
			},
		},
	}
	twoDeletedPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "deletedPVC",
						},
					},
				},
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherDeletedPVC",
						},
					},
				},
			},
		},
	}
	deletedPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "pvcWithDeletedPV",
						},
					},
				},
			},
		},
	}
	// deletedPVPod2 is a different pod than deletedPVPod but using the same PVC
	deletedPVPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "pvcWithDeletedPV",
						},
					},
				},
			},
		},
	}
	// anotherDeletedPVPod is a different pod than deletedPVPod and uses another PVC
	anotherDeletedPVPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherPVCWithDeletedPV",
						},
					},
				},
			},
		},
	}
	emptyPod := &v1.Pod{
		Spec: v1.PodSpec{},
	}
	unboundPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVC",
						},
					},
				},
			},
		},
	}
	// Different pod than unboundPVCPod, but using the same unbound PVC
	unboundPVCPod2 := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "unboundPVC",
						},
					},
				},
			},
		},
	}

	// pod with unbound PVC that's different to unboundPVC
	anotherUnboundPVCPod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "anotherUnboundPVC",
						},
					},
				},
			},
		},
	}

	tests := []struct {
		newPod       *v1.Pod
		existingPods []*v1.Pod
		maxVols      int
		fits         bool
		test         string
	}{
		{
			newPod:       oneVolPod,
			existingPods: []*v1.Pod{twoVolPod, oneVolPod},
			maxVols:      4,
			fits:         true,
			test:         "fits when node capacity >= new pod's EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod},
			maxVols:      2,
			fits:         false,
			test:         "doesn't fit when node capacity < new pod's EBS volumes",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{twoVolPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores non-EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			maxVols:      3,
			fits:         true,
			test:         "existing pods' counts ignore non-EBS volumes",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{splitVolsPod, nonApplicablePod, emptyPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count considers PVCs backed by EBS volumes",
		},
		{
			newPod:       splitPVCPod,
			existingPods: []*v1.Pod{splitVolsPod, oneVolPod},
			maxVols:      3,
			fits:         true,
			test:         "new pod's count ignores PVCs not backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, ebsPVCPod},
			maxVols:      3,
			fits:         false,
			test:         "existing pods' counts considers PVCs backed by EBS volumes",
		},
		{
			newPod:       twoVolPod,
			existingPods: []*v1.Pod{oneVolPod, twoVolPod, ebsPVCPod},
			maxVols:      4,
			fits:         true,
			test:         "already-mounted EBS volumes are always ok to allow",
		},
		{
			newPod:       splitVolsPod,
			existingPods: []*v1.Pod{oneVolPod, oneVolPod, ebsPVCPod},
			maxVols:      3,
			fits:         true,
			test:         "the same EBS volumes are not counted multiple times",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVCPod},
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PVC is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, twoDeletedPVCPod},
			maxVols:      3,
			fits:         false,
			test:         "pod with missing two PVCs is counted towards the PV limit twice",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			maxVols:      2,
			fits:         false,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			maxVols:      3,
			fits:         true,
			test:         "pod with missing PV is counted towards the PV limit",
		},
		{
			newPod:       deletedPVPod2,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			maxVols:      2,
			fits:         true,
			test:         "two pods missing the same PV are counted towards the PV limit only once",
		},
		{
			newPod:       anotherDeletedPVPod,
			existingPods: []*v1.Pod{oneVolPod, deletedPVPod},
			maxVols:      2,
			fits:         false,
			test:         "two pods missing different PVs are counted towards the PV limit twice",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			maxVols:      2,
			fits:         false,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       ebsPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			maxVols:      3,
			fits:         true,
			test:         "pod with unbound PVC is counted towards the PV limit",
		},
		{
			newPod:       unboundPVCPod2,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			maxVols:      2,
			fits:         true,
			test:         "the same unbound PVC in multiple pods is counted towards the PV limit only once",
		},
		{
			newPod:       anotherUnboundPVCPod,
			existingPods: []*v1.Pod{oneVolPod, unboundPVCPod},
			maxVols:      2,
			fits:         false,
			test:         "two different unbound PVCs are counted towards the PV limit as two volumes",
		},
	}

	pvInfo := FakePersistentVolumeInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someEBSVol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{
					AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{VolumeID: "ebsVol"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNonEBSVol"},
			Spec: v1.PersistentVolumeSpec{
				PersistentVolumeSource: v1.PersistentVolumeSource{},
			},
		},
	}

	pvcInfo := FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someEBSVol"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "someEBSVol"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "someNonEBSVol"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "someNonEBSVol"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "pvcWithDeletedPV"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "pvcWithDeletedPV"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherPVCWithDeletedPV"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "anotherPVCWithDeletedPV"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "unboundPVC"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "anotherUnboundPVC"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: ""},
		},
	}

	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}

	for _, test := range tests {
		os.Setenv(KubeMaxPDVols, strconv.Itoa(test.maxVols))
		pred := NewMaxPDVolumeCountPredicate(EBSVolumeFilterType, pvInfo, pvcInfo)
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

func TestRunGeneralPredicates(t *testing.T) {
	resourceTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulercache.NodeInfo
		node     *v1.Node
		fits     bool
		test     string
		wErr     error
		reasons  []algorithm.PredicateFailureReason
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0, 0, 0)},
			},
			fits: true,
			wErr: nil,
			test: "no resources/port/host requested always fits",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 8, Memory: 10}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 19})),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0, 0, 0)},
			},
			fits: false,
			wErr: nil,
			reasons: []algorithm.PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceCPU, 8, 5, 10),
				NewInsufficientResourceError(v1.ResourceMemory, 10, 19, 20),
			},
			test: "not enough cpu and memory resource",
		},
		{
			pod: &v1.Pod{},
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 9, Memory: 19})),
			node: &v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0, 0, 0)}},
			fits: true,
			wErr: nil,
			test: "no resources/port/host requested always fits on GPU machine",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 3, Memory: 1, NvidiaGPU: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 10, NvidiaGPU: 1})),
			node:    &v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0, 0, 0)}},
			fits:    false,
			wErr:    nil,
			reasons: []algorithm.PredicateFailureReason{NewInsufficientResourceError(v1.ResourceNvidiaGPU, 1, 1, 1)},
			test:    "not enough GPU resource",
		},
		{
			pod: newResourcePod(schedulercache.Resource{MilliCPU: 3, Memory: 1, NvidiaGPU: 1}),
			nodeInfo: schedulercache.NewNodeInfo(
				newResourcePod(schedulercache.Resource{MilliCPU: 5, Memory: 10, NvidiaGPU: 0})),
			node: &v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 1, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 1, 32, 0, 0, 0)}},
			fits: true,
			wErr: nil,
			test: "enough GPU resource",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "machine2",
				},
			},
			nodeInfo: schedulercache.NewNodeInfo(),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0, 0, 0)},
			},
			fits:    false,
			wErr:    nil,
			reasons: []algorithm.PredicateFailureReason{ErrPodNotMatchHostName},
			test:    "host not match",
		},
		{
			pod:      newPodWithPort(123),
			nodeInfo: schedulercache.NewNodeInfo(newPodWithPort(123)),
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "machine1"},
				Status:     v1.NodeStatus{Capacity: makeResources(10, 20, 0, 32, 0, 0, 0).Capacity, Allocatable: makeAllocatableResources(10, 20, 0, 32, 0, 0, 0)},
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

// TODO: Add test case for RequiredDuringSchedulingRequiredDuringExecution after it's implemented.
func TestInterPodAffinity(t *testing.T) {
	podLabel := map[string]string{"service": "securityscan"}
	labels1 := map[string]string{
		"region": "r1",
		"zone":   "z11",
	}
	podLabel2 := map[string]string{"security": "S1"}
	node1 := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: labels1}}
	tests := []struct {
		pod                  *v1.Pod
		pods                 []*v1.Pod
		node                 *v1.Node
		fits                 bool
		test                 string
		expectFailureReasons []algorithm.PredicateFailureReason
	}{
		{
			pod:  new(v1.Pod),
			node: &node1,
			fits: true,
			test: "A pod that has no required pod affinity scheduling rules can schedule onto a node with no existing pods",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies with requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using In operator that matches the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"securityscan3", "value3"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the pod with requiredDuringSchedulingIgnoredDuringExecution in PodAffinity using not in operator in labelSelector that matches the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									Namespaces: []string{"DiffNameSpace"},
								},
							},
						},
					},
				},
			},
			pods:                 []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel, Namespace: "ns"}}},
			node:                 &node1,
			fits:                 false,
			test:                 "Does not satisfy the PodAffinity with labelSelector because of diff Namespace",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrPodAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			pods:                 []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node:                 &node1,
			fits:                 false,
			test:                 "Doesn't satisfy the PodAffinity because of unmatching labelSelector with the existing pod",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrPodAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpExists,
											}, {
												Key:      "wrongkey",
												Operator: metav1.LabelSelectorOpDoesNotExist,
											},
										},
									},
									TopologyKey: "region",
								}, {
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan"},
											}, {
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"WrongValue"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity with different label Operators in multiple RequiredDuringSchedulingIgnoredDuringExecution ",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpExists,
											}, {
												Key:      "wrongkey",
												Operator: metav1.LabelSelectorOpDoesNotExist,
											},
										},
									},
									TopologyKey: "region",
								}, {
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan2"},
											}, {
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"WrongValue"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods:                 []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node:                 &node1,
			fits:                 false,
			test:                 "The labelSelector requirements(items of matchExpressions) are ANDed, the pod cannot schedule onto the node because one of the matchExpression item don't match.",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrPodAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity and PodAntiAffinity with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						NodeName: "machine1",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"antivirusscan", "value2"},
												},
											},
										},
										TopologyKey: "node",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{Labels: podLabel},
				},
			},
			node: &node1,
			fits: true,
			test: "satisfies the PodAffinity and PodAntiAffinity and PodAntiAffinity symmetry with the existing pod",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel2,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "zone",
								},
							},
						},
					},
				},
			},
			pods:                 []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node:                 &node1,
			fits:                 false,
			test:                 "satisfies the PodAffinity but doesn't satisfies the PodAntiAffinity with the existing pod",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"antivirusscan", "value2"},
											},
										},
									},
									TopologyKey: "node",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{
						NodeName: "machine1",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"securityscan", "value2"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{Labels: podLabel},
				},
			},
			node:                 &node1,
			fits:                 false,
			test:                 "satisfies the PodAffinity and PodAntiAffinity but doesn't satisfies PodAntiAffinity symmetry with the existing pod",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrExistingPodsAntiAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel,
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "service",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"securityscan", "value2"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods:                 []*v1.Pod{{Spec: v1.PodSpec{NodeName: "machine2"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabel}}},
			node:                 &node1,
			fits:                 false,
			test:                 "pod matches its own Label in PodAffinity and that matches the existing pod Labels",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrPodAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel,
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{NodeName: "machine1",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"securityscan", "value2"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{Labels: podLabel},
				},
			},
			node:                 &node1,
			fits:                 false,
			test:                 "verify that PodAntiAffinity from existing pod is respected when pod has no AntiAffinity constraints. doesn't satisfy PodAntiAffinity symmetry with the existing pod",
			expectFailureReasons: []algorithm.PredicateFailureReason{ErrPodAffinityNotMatch, ErrExistingPodsAntiAffinityRulesNotMatch},
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabel,
				},
			},
			pods: []*v1.Pod{
				{
					Spec: v1.PodSpec{NodeName: "machine1",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "service",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"securityscan", "value2"},
												},
											},
										},
										TopologyKey: "zone",
									},
								},
							},
						},
					},
					ObjectMeta: metav1.ObjectMeta{Labels: podLabel},
				},
			},
			node: &node1,
			fits: true,
			test: "verify that PodAntiAffinity from existing pod is respected when pod has no AntiAffinity constraints. satisfy PodAntiAffinity symmetry with the existing pod",
		},
	}

	for _, test := range tests {
		node := test.node
		var podsOnNode []*v1.Pod
		for _, pod := range test.pods {
			if pod.Spec.NodeName == node.Name {
				podsOnNode = append(podsOnNode, pod)
			}
		}

		fit := PodAffinityChecker{
			info:      FakeNodeInfo(*node),
			podLister: schedulertesting.FakePodLister(test.pods),
		}
		nodeInfo := schedulercache.NewNodeInfo(podsOnNode...)
		nodeInfo.SetNode(test.node)
		nodeInfoMap := map[string]*schedulercache.NodeInfo{test.node.Name: nodeInfo}
		fits, reasons, _ := fit.InterPodAffinityMatches(test.pod, PredicateMetadata(test.pod, nodeInfoMap), nodeInfo)
		if !fits && !reflect.DeepEqual(reasons, test.expectFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.test, reasons, test.expectFailureReasons)
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
	labelRgUS := map[string]string{
		"region": "US",
	}

	tests := []struct {
		pod                               *v1.Pod
		pods                              []*v1.Pod
		nodes                             []v1.Node
		nodesExpectAffinityFailureReasons [][]algorithm.PredicateFailureReason
		fits                              map[string]bool
		test                              string
		nometa                            bool
	}{
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{NodeName: "machine1"}, ObjectMeta: metav1.ObjectMeta{Labels: podLabelA}},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "machine1", Labels: labelRgChina}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine2", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "machine3", Labels: labelRgIndia}},
			},
			fits: map[string]bool{
				"machine1": true,
				"machine2": true,
				"machine3": false,
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{nil, nil, {ErrPodAffinityNotMatch, ErrPodAffinityRulesNotMatch}},
			test: "A pod can be scheduled onto all the nodes that have the same topology key & label value with one of them has an existing pod that match the affinity rules",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "hostname",
												Operator: v1.NodeSelectorOpNotIn,
												Values:   []string{"h1"},
											},
										},
									},
								},
							},
						},
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"abc"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{NodeName: "nodeA"}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
				{Spec: v1.PodSpec{NodeName: "nodeB"}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "def"}}},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: map[string]string{"region": "r1", "hostname": "h1"}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: map[string]string{"region": "r1", "hostname": "h2"}}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{nil, nil},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": true,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA does not satisfy node affinity rule, but has an existing pod that match the inter pod affinity rule. The pod can be scheduled onto nodeB.",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAffinity: &v1.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
									TopologyKey: "zone",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: map[string]string{"zone": "az1", "hostname": "h1"}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: map[string]string{"zone": "az2", "hostname": "h2"}}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{nil, nil},
			fits: map[string]bool{
				"nodeA": true,
				"nodeB": true,
			},
			test: "The affinity rule is to schedule all of the pods of this collection to the same zone. The first pod of the collection " +
				"should not be blocked from being scheduled onto any node, even there's no existing pod that match the rule anywhere.",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"abc"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{NodeName: "nodeA"}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: map[string]string{"region": "r1", "hostname": "nodeA"}}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: map[string]string{"region": "r1", "hostname": "nodeB"}}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch}, {ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch}},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA and nodeB.",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"abc"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{NodeName: "nodeA"}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "abc"}}},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: labelRgChina}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: labelRgIndia}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch}, {ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch}, nil},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
				"nodeC": true,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA and nodeB but can be schedulerd onto nodeC",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "123"}},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{Spec: v1.PodSpec{NodeName: "nodeA"}, ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar"}}},
				{
					Spec: v1.PodSpec{
						NodeName: "nodeC",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "foo",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"123"},
												},
											},
										},
										TopologyKey: "region",
									},
								},
							},
						},
					},
				},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: labelRgChina}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: labelRgIndia}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeD", Labels: labelRgUS}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{
				{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch},
				{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch},
				{ErrPodAffinityNotMatch, ErrExistingPodsAntiAffinityRulesNotMatch},
				nil,
			},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
				"nodeC": false,
				"nodeD": true,
			},
			test:   "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. NodeC has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA, nodeB and nodeC but can be schedulerd onto nodeD",
			nometa: true,
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Labels:    map[string]string{"foo": "123"},
					Namespace: "NS1",
				},
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						PodAntiAffinity: &v1.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "foo",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
									TopologyKey: "region",
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "NS1",
					},
					Spec: v1.PodSpec{NodeName: "nodeA"},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Namespace: "NS2"},
					Spec: v1.PodSpec{
						NodeName: "nodeC",
						Affinity: &v1.Affinity{
							PodAntiAffinity: &v1.PodAntiAffinity{
								RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
									{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "foo",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"123"},
												},
											},
										},
										TopologyKey: "region",
									},
								},
							},
						},
					},
				},
			},
			nodes: []v1.Node{
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeA", Labels: labelRgChina}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeB", Labels: labelRgChinaAzAz1}},
				{ObjectMeta: metav1.ObjectMeta{Name: "nodeC", Labels: labelRgIndia}},
			},
			nodesExpectAffinityFailureReasons: [][]algorithm.PredicateFailureReason{
				{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch},
				{ErrPodAffinityNotMatch, ErrPodAntiAffinityRulesNotMatch},
				nil,
			},
			fits: map[string]bool{
				"nodeA": false,
				"nodeB": false,
				"nodeC": true,
			},
			test: "NodeA and nodeB have same topologyKey and label value. NodeA has an existing pod that match the inter pod affinity rule. The pod can not be scheduled onto nodeA, nodeB, but can be schedulerd onto nodeC (NodeC has an existing pod that match the inter pod affinity rule but in different namespace)",
		},
	}

	selectorExpectedFailureReasons := []algorithm.PredicateFailureReason{ErrNodeSelectorNotMatch}

	for indexTest, test := range tests {
		nodeListInfo := FakeNodeListInfo(test.nodes)
		for indexNode, node := range test.nodes {
			var podsOnNode []*v1.Pod
			for _, pod := range test.pods {
				if pod.Spec.NodeName == node.Name {
					podsOnNode = append(podsOnNode, pod)
				}
			}

			testFit := PodAffinityChecker{
				info:      nodeListInfo,
				podLister: schedulertesting.FakePodLister(test.pods),
			}
			nodeInfo := schedulercache.NewNodeInfo(podsOnNode...)
			nodeInfo.SetNode(&node)
			nodeInfoMap := map[string]*schedulercache.NodeInfo{node.Name: nodeInfo}

			var meta algorithm.PredicateMetadata

			if !test.nometa {
				meta = PredicateMetadata(test.pod, nodeInfoMap)
			}

			fits, reasons, _ := testFit.InterPodAffinityMatches(test.pod, meta, nodeInfo)
			if !fits && !reflect.DeepEqual(reasons, test.nodesExpectAffinityFailureReasons[indexNode]) {
				t.Errorf("index: %d test: %s unexpected failure reasons: %v expect: %v", indexTest, test.test, reasons, test.nodesExpectAffinityFailureReasons[indexNode])
			}
			affinity := test.pod.Spec.Affinity
			if affinity != nil && affinity.NodeAffinity != nil {
				nodeInfo := schedulercache.NewNodeInfo()
				nodeInfo.SetNode(&node)
				nodeInfoMap := map[string]*schedulercache.NodeInfo{node.Name: nodeInfo}
				fits2, reasons, err := PodMatchNodeSelector(test.pod, PredicateMetadata(test.pod, nodeInfoMap), nodeInfo)
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
		pod  *v1.Pod
		node v1.Node
		fits bool
		test string
	}{
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod0",
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}},
				},
			},
			fits: false,
			test: "a pod having no tolerations can't be scheduled onto a node with nonempty taints",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod1:V1"}},
					Tolerations: []v1.Toleration{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}},
				},
			},
			fits: true,
			test: "a pod which can be scheduled on a dedicated node assigned to user1 with effect NoSchedule",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{{Key: "dedicated", Value: "user1", Effect: "NoSchedule"}},
				},
			},
			fits: false,
			test: "a pod which can't be scheduled on a dedicated node assigned to user2 with effect NoSchedule",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{{Key: "foo", Operator: "Exists", Effect: "NoSchedule"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{{Key: "foo", Value: "bar", Effect: "NoSchedule"}},
				},
			},
			fits: true,
			test: "a pod can be scheduled onto the node, with a toleration uses operator Exists that tolerates the taints on the node",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{
						{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"},
						{Key: "foo", Operator: "Exists", Effect: "NoSchedule"},
					},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "dedicated", Value: "user2", Effect: "NoSchedule"},
						{Key: "foo", Value: "bar", Effect: "NoSchedule"},
					},
				},
			},
			fits: true,
			test: "a pod has multiple tolerations, node has multiple taints, all the taints are tolerated, pod can be scheduled onto the node",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{{Key: "foo", Operator: "Equal", Value: "bar", Effect: "PreferNoSchedule"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "foo", Value: "bar", Effect: "NoSchedule"},
					},
				},
			},
			fits: false,
			test: "a pod has a toleration that keys and values match the taint on the node, but (non-empty) effect doesn't match, " +
				"can't be scheduled onto the node",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{{Key: "foo", Operator: "Equal", Value: "bar"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "foo", Value: "bar", Effect: "NoSchedule"},
					},
				},
			},
			fits: true,
			test: "The pod has a toleration that keys and values match the taint on the node, the effect of toleration is empty, " +
				"and the effect of taint is NoSchedule. Pod can be scheduled onto the node",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers:  []v1.Container{{Image: "pod2:V1"}},
					Tolerations: []v1.Toleration{{Key: "dedicated", Operator: "Equal", Value: "user2", Effect: "NoSchedule"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "dedicated", Value: "user1", Effect: "PreferNoSchedule"},
					},
				},
			},
			fits: true,
			test: "The pod has a toleration that key and value don't match the taint on the node, " +
				"but the effect of taint on node is PreferNochedule. Pod can be scheduled onto the node",
		},
		{
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{Image: "pod2:V1"}},
				},
			},
			node: v1.Node{
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "dedicated", Value: "user1", Effect: "PreferNoSchedule"},
					},
				},
			},
			fits: true,
			test: "The pod has no toleration, " +
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

func makeEmptyNodeInfo(node *v1.Node) *schedulercache.NodeInfo {
	nodeInfo := schedulercache.NewNodeInfo()
	nodeInfo.SetNode(node)
	return nodeInfo
}

func TestPodSchedulesOnNodeWithMemoryPressureCondition(t *testing.T) {
	// specify best-effort pod
	bestEffortPod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					// no requirements -> best effort pod
					Resources: v1.ResourceRequirements{},
				},
			},
		},
	}

	// specify non-best-effort pod
	nonBestEffortPod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
					// at least one requirement -> burstable pod
					Resources: v1.ResourceRequirements{
						Requests: makeAllocatableResources(100, 100, 100, 100, 0, 0, 0),
					},
				},
			},
		},
	}

	// specify a node with no memory pressure condition on
	noMemoryPressureNode := &v1.Node{
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   "Ready",
					Status: "True",
				},
			},
		},
	}

	// specify a node with memory pressure condition on
	memoryPressureNode := &v1.Node{
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   "MemoryPressure",
					Status: "True",
				},
			},
		},
	}

	tests := []struct {
		pod      *v1.Pod
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
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "container",
					Image:           "image",
					ImagePullPolicy: "Always",
				},
			},
		},
	}

	// specify a node with no disk pressure condition on
	noPressureNode := &v1.Node{
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   "Ready",
					Status: "True",
				},
			},
		},
	}

	// specify a node with pressure condition on
	pressureNode := &v1.Node{
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   "DiskPressure",
					Status: "True",
				},
			},
		},
	}

	tests := []struct {
		pod      *v1.Pod
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

func TestNodeConditionPredicate(t *testing.T) {
	tests := []struct {
		node        *v1.Node
		schedulable bool
	}{
		// node1 considered
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node1"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}},
			schedulable: true,
		},
		// node2 ignored - node not Ready
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node2"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}}},
			schedulable: false,
		},
		// node3 ignored - node out of disk
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node3"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			schedulable: false,
		},

		// node4 considered
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node4"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},
			schedulable: true,
		},
		// node5 ignored - node out of disk
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node5"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			schedulable: false,
		},
		// node6 considered
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node6"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},
			schedulable: true,
		},
		// node7 ignored - node out of disk, node not Ready
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node7"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			schedulable: false,
		},
		// node8 ignored - node not Ready
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node8"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},
			schedulable: false,
		},
		// node9 ignored - node unschedulable
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node9"}, Spec: v1.NodeSpec{Unschedulable: true}},
			schedulable: false,
		},
		// node10 considered
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node10"}, Spec: v1.NodeSpec{Unschedulable: false}},
			schedulable: true,
		},
		// node11 considered
		{
			node:        &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node11"}},
			schedulable: true,
		},
	}

	for _, test := range tests {
		nodeInfo := makeEmptyNodeInfo(test.node)
		if fit, reasons, err := CheckNodeConditionPredicate(nil, nil, nodeInfo); fit != test.schedulable {
			t.Errorf("%s: expected: %t, got %t; %+v, %v",
				test.node.Name, test.schedulable, fit, reasons, err)
		}
	}
}

func createPodWithVolume(pod, pv, pvc string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: pod, Namespace: "default"},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: pv,
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: pvc,
						},
					},
				},
			},
		},
	}
}

func TestVolumeZonePredicate(t *testing.T) {
	pvInfo := FakePersistentVolumeInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{kubeletapis.LabelZoneRegion: "us-west1-b", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{kubeletapis.LabelZoneRegion: "us-west1-c"}},
		},
	}

	pvcInfo := FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_3", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_3"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_4", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_not_exist"},
		},
	}

	tests := []struct {
		Name string
		Pod  *v1.Pod
		Fits bool
		Node *v1.Node
	}{
		{
			Name: "pod without volume",
			Pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod_1", Namespace: "default"},
			},
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a"},
				},
			},
			Fits: true,
		},
		{
			Name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
			Fits: true,
		},
		{
			Name: "label zone failure domain matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
				},
			},
			Fits: true,
		},
		{
			Name: "label zone region matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneRegion: "us-west1-b", "uselessLabel": "none"},
				},
			},
			Fits: true,
		},
		{
			Name: "label zone region failed match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_2"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneRegion: "no_us-west1-b", "uselessLabel": "none"},
				},
			},
			Fits: false,
		},
		{
			Name: "label zone failure domain failed match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "no_us-west1-a", "uselessLabel": "none"},
				},
			},
			Fits: false,
		},
	}

	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrVolumeZoneConflict}

	for _, test := range tests {
		fit := NewVolumeZonePredicate(pvInfo, pvcInfo, nil)
		node := &schedulercache.NodeInfo{}
		node.SetNode(test.Node)

		fits, reasons, err := fit(test.Pod, nil, node)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.Name, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.Name, reasons, expectedFailureReasons)
		}
		if fits != test.Fits {
			t.Errorf("%s: expected %v got %v", test.Name, test.Fits, fits)
		}

	}
}

func TestVolumeZonePredicateMultiZone(t *testing.T) {
	pvInfo := FakePersistentVolumeInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_2", Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-b", "uselessLabel": "none"}},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_3", Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-c__us-west1-a"}},
		},
	}

	pvcInfo := FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_2", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_2"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_3", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_3"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_4", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_not_exist"},
		},
	}

	tests := []struct {
		Name string
		Pod  *v1.Pod
		Fits bool
		Node *v1.Node
	}{
		{
			Name: "node without labels",
			Pod:  createPodWithVolume("pod_1", "Vol_3", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "host1",
				},
			},
			Fits: true,
		},
		{
			Name: "label zone failure domain matched",
			Pod:  createPodWithVolume("pod_1", "Vol_3", "PVC_3"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
				},
			},
			Fits: true,
		},
		{
			Name: "label zone failure domain failed match",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "host1",
					Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-b", "uselessLabel": "none"},
				},
			},
			Fits: false,
		},
	}

	expectedFailureReasons := []algorithm.PredicateFailureReason{ErrVolumeZoneConflict}

	for _, test := range tests {
		fit := NewVolumeZonePredicate(pvInfo, pvcInfo, nil)
		node := &schedulercache.NodeInfo{}
		node.SetNode(test.Node)

		fits, reasons, err := fit(test.Pod, nil, node)
		if err != nil {
			t.Errorf("%s: unexpected error: %v", test.Name, err)
		}
		if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
			t.Errorf("%s: unexpected failure reasons: %v, want: %v", test.Name, reasons, expectedFailureReasons)
		}
		if fits != test.Fits {
			t.Errorf("%s: expected %v got %v", test.Name, test.Fits, fits)
		}

	}
}

func TestVolumeZonePredicateWithVolumeBinding(t *testing.T) {
	var (
		modeWait = storagev1.VolumeBindingWaitForFirstConsumer

		class0         = "Class_0"
		classWait      = "Class_Wait"
		classImmediate = "Class_Immediate"
	)

	classInfo := FakeStorageClassInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: classImmediate},
		},
		{
			ObjectMeta:        metav1.ObjectMeta{Name: classWait},
			VolumeBindingMode: &modeWait,
		},
	}

	pvInfo := FakePersistentVolumeInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "Vol_1", Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a"}},
		},
	}

	pvcInfo := FakePersistentVolumeClaimInfo{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_1", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "Vol_1"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_NoSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &class0},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_EmptySC", Namespace: "default"},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_WaitSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &classWait},
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "PVC_ImmediateSC", Namespace: "default"},
			Spec:       v1.PersistentVolumeClaimSpec{StorageClassName: &classImmediate},
		},
	}

	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "host1",
			Labels: map[string]string{kubeletapis.LabelZoneFailureDomain: "us-west1-a", "uselessLabel": "none"},
		},
	}

	tests := []struct {
		Name          string
		Pod           *v1.Pod
		Fits          bool
		Node          *v1.Node
		ExpectFailure bool
	}{
		{
			Name: "label zone failure domain matched",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_1"),
			Node: testNode,
			Fits: true,
		},
		{
			Name:          "unbound volume empty storage class",
			Pod:           createPodWithVolume("pod_1", "vol_1", "PVC_EmptySC"),
			Node:          testNode,
			Fits:          false,
			ExpectFailure: true,
		},
		{
			Name:          "unbound volume no storage class",
			Pod:           createPodWithVolume("pod_1", "vol_1", "PVC_NoSC"),
			Node:          testNode,
			Fits:          false,
			ExpectFailure: true,
		},
		{
			Name:          "unbound volume immediate binding mode",
			Pod:           createPodWithVolume("pod_1", "vol_1", "PVC_ImmediateSC"),
			Node:          testNode,
			Fits:          false,
			ExpectFailure: true,
		},
		{
			Name: "unbound volume wait binding mode",
			Pod:  createPodWithVolume("pod_1", "vol_1", "PVC_WaitSC"),
			Node: testNode,
			Fits: true,
		},
	}

	err := utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true")
	if err != nil {
		t.Fatalf("Failed to enable feature gate for VolumeScheduling: %v", err)
	}

	for _, test := range tests {
		fit := NewVolumeZonePredicate(pvInfo, pvcInfo, classInfo)
		node := &schedulercache.NodeInfo{}
		node.SetNode(test.Node)

		fits, _, err := fit(test.Pod, nil, node)
		if !test.ExpectFailure && err != nil {
			t.Errorf("%s: unexpected error: %v", test.Name, err)
		}
		if test.ExpectFailure && err == nil {
			t.Errorf("%s: expected error, got success", test.Name)
		}
		if fits != test.Fits {
			t.Errorf("%s: expected %v got %v", test.Name, test.Fits, fits)
		}
	}

	err = utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for VolumeScheduling: %v", err)
	}
}

func TestGetMaxVols(t *testing.T) {
	previousValue := os.Getenv(KubeMaxPDVols)
	defaultValue := 39

	tests := []struct {
		rawMaxVols string
		expected   int
		test       string
	}{
		{
			rawMaxVols: "invalid",
			expected:   defaultValue,
			test:       "Unable to parse maximum PD volumes value, using default value",
		},
		{
			rawMaxVols: "-2",
			expected:   defaultValue,
			test:       "Maximum PD volumes must be a positive value, using default value",
		},
		{
			rawMaxVols: "40",
			expected:   40,
			test:       "Parse maximum PD volumes value from env",
		},
	}

	for _, test := range tests {
		os.Setenv(KubeMaxPDVols, test.rawMaxVols)
		result := getMaxVols(defaultValue)
		if result != test.expected {
			t.Errorf("%s: expected %v got %v", test.test, test.expected, result)
		}
	}

	os.Unsetenv(KubeMaxPDVols)
	if previousValue != "" {
		os.Setenv(KubeMaxPDVols, previousValue)
	}
}
