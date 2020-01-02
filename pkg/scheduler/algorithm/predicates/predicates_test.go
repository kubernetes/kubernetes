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
	"reflect"
	"strconv"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	extendedResourceA     = v1.ResourceName("example.com/aaa")
	extendedResourceB     = v1.ResourceName("example.com/bbb")
	kubernetesIOResourceA = v1.ResourceName("kubernetes.io/something")
	kubernetesIOResourceB = v1.ResourceName("subdomain.kubernetes.io/something")
	hugePageResourceA     = v1helper.HugePageResourceName(resource.MustParse("2Mi"))
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

func newResourceInitPod(pod *v1.Pod, usage ...schedulernodeinfo.Resource) *v1.Pod {
	pod.Spec.InitContainers = newResourcePod(usage...).Spec.Containers
	return pod
}

func newResourceOverheadPod(pod *v1.Pod, overhead v1.ResourceList) *v1.Pod {
	pod.Spec.Overhead = overhead
	return pod
}
func TestPodFitsResources(t *testing.T) {

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodOverhead, true)()

	enoughPodsTests := []struct {
		pod                      *v1.Pod
		nodeInfo                 *schedulernodeinfo.NodeInfo
		fits                     bool
		name                     string
		reasons                  []PredicateFailureReason
		ignoredExtendedResources sets.String
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 10, Memory: 20})),
			fits: true,
			name: "no resources requested always fits",
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 10, Memory: 20})),
			fits: false,
			name: "too many resources fails",
			reasons: []PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceCPU, 1, 10, 10),
				NewInsufficientResourceError(v1.ResourceMemory, 1, 20, 20),
			},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 3, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			name:    "too many resources fails due to init container cpu",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 3, Memory: 1}, schedulernodeinfo.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 8, Memory: 19})),
			fits:    false,
			name:    "too many resources fails due to highest init container cpu",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 3, 8, 10)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 1, Memory: 3}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			name:    "too many resources fails due to init container memory",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 3, 19, 20)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 1, Memory: 3}, schedulernodeinfo.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 19})),
			fits:    false,
			name:    "too many resources fails due to highest init container memory",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 3, 19, 20)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 19})),
			fits: true,
			name: "init container fits because it's the max, not sum, of containers and init containers",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}, schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 19})),
			fits: true,
			name: "multiple init containers fit because it's the max, not sum, of containers and init containers",
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 5})),
			fits: true,
			name: "both resources fit",
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 2, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 9, Memory: 5})),
			fits:    false,
			name:    "one resource memory fits",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceCPU, 2, 9, 10)},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 2}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			name:    "one resource cpu fits",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourceMemory, 2, 19, 20)},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			fits: true,
			name: "equal edge case",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 4, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			fits: true,
			name: "equal edge case for init container",
		},
		{
			pod:      newResourcePod(schedulernodeinfo.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(newResourcePod(schedulernodeinfo.Resource{})),
			fits:     true,
			name:     "extended resource fits",
		},
		{
			pod:      newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}), schedulernodeinfo.Resource{ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(newResourcePod(schedulernodeinfo.Resource{})),
			fits:     true,
			name:     "extended resource fits for init container",
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			fits:    false,
			name:    "extended resource capacity enforced",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 10, 0, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 0}})),
			fits:    false,
			name:    "extended resource capacity enforced for init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 10, 0, 5)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			fits:    false,
			name:    "extended resource allocatable enforced",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 1, 5, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 5}})),
			fits:    false,
			name:    "extended resource allocatable enforced for init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 1, 5, 5)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits:    false,
			name:    "extended resource allocatable enforced for multiple containers",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 6, 2, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}},
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits: true,
			name: "extended resource allocatable admits multiple init containers",
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 6}},
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 3}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{extendedResourceA: 2}})),
			fits:    false,
			name:    "extended resource allocatable enforced for multiple init containers",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceA, 6, 2, 5)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			name:    "extended resource allocatable enforced for unknown resource",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceB, 1, 0, 0)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			name:    "extended resource allocatable enforced for unknown resource for init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(extendedResourceB, 1, 0, 0)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{kubernetesIOResourceA: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			name:    "kubernetes.io resource capacity enforced",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(kubernetesIOResourceA, 10, 0, 0)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{kubernetesIOResourceB: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0})),
			fits:    false,
			name:    "kubernetes.io resource capacity enforced for init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(kubernetesIOResourceB, 10, 0, 0)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			fits:    false,
			name:    "hugepages resource capacity enforced",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 10, 0, 5)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{}),
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 10}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 0}})),
			fits:    false,
			name:    "hugepages resource capacity enforced for init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 10, 0, 5)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}},
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 3}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0, ScalarResources: map[v1.ResourceName]int64{hugePageResourceA: 2}})),
			fits:    false,
			name:    "hugepages resource allocatable enforced for multiple containers",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(hugePageResourceA, 6, 2, 5)},
		},
		{
			pod: newResourcePod(
				schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1, ScalarResources: map[v1.ResourceName]int64{extendedResourceB: 1}}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 0, Memory: 0})),
			fits:                     true,
			ignoredExtendedResources: sets.NewString(string(extendedResourceB)),
			name:                     "skip checking ignored extended resource",
		},
		{
			pod: newResourceOverheadPod(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceList{v1.ResourceCPU: resource.MustParse("3m"), v1.ResourceMemory: resource.MustParse("13")},
			),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 5})),
			fits:                     true,
			ignoredExtendedResources: sets.NewString(string(extendedResourceB)),
			name:                     "resources + pod overhead fits",
		},
		{
			pod: newResourceOverheadPod(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
				v1.ResourceList{v1.ResourceCPU: resource.MustParse("1m"), v1.ResourceMemory: resource.MustParse("15")},
			),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 5})),
			fits:                     false,
			ignoredExtendedResources: sets.NewString(string(extendedResourceB)),
			name:                     "requests + overhead does not fit for memory",
			reasons: []PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceMemory, 16, 5, 20),
			},
		},
	}

	for _, test := range enoughPodsTests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 5, 20, 5).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 5, 20, 5)}}
			test.nodeInfo.SetNode(&node)
			fits, reasons, err := PodFitsResourcesPredicate(test.pod, GetResourceRequest(test.pod), test.ignoredExtendedResources, test.nodeInfo)
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

	notEnoughPodsTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulernodeinfo.NodeInfo
		fits     bool
		name     string
		reasons  []PredicateFailureReason
	}{
		{
			pod: &v1.Pod{},
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 10, Memory: 20})),
			fits:    false,
			name:    "even without specified resources predicate fails when there's no space for additional pod",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 5})),
			fits:    false,
			name:    "even if both resources fit predicate fails when there's no space for additional pod",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			name:    "even for equal edge case predicate fails when there's no space for additional pod",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
		{
			pod: newResourceInitPod(newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 1}), schedulernodeinfo.Resource{MilliCPU: 5, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 5, Memory: 19})),
			fits:    false,
			name:    "even for equal edge case predicate fails when there's no space for additional pod due to init container",
			reasons: []PredicateFailureReason{NewInsufficientResourceError(v1.ResourcePods, 1, 1, 1)},
		},
	}
	for _, test := range notEnoughPodsTests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: v1.ResourceList{}, Allocatable: makeAllocatableResources(10, 20, 1, 0, 0, 0)}}
			test.nodeInfo.SetNode(&node)
			fits, reasons, err := PodFitsResourcesPredicate(test.pod, GetResourceRequest(test.pod), nil, test.nodeInfo)

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

	storagePodsTests := []struct {
		pod      *v1.Pod
		nodeInfo *schedulernodeinfo.NodeInfo
		fits     bool
		name     string
		reasons  []PredicateFailureReason
	}{
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 10, Memory: 10})),
			fits: false,
			name: "due to container scratch disk",
			reasons: []PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceCPU, 1, 10, 10),
			},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{MilliCPU: 1, Memory: 1}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 2, Memory: 10})),
			fits: true,
			name: "pod fit",
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{EphemeralStorage: 25}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 2, Memory: 2})),
			fits: false,
			name: "storage ephemeral local storage request exceeds allocatable",
			reasons: []PredicateFailureReason{
				NewInsufficientResourceError(v1.ResourceEphemeralStorage, 25, 0, 20),
			},
		},
		{
			pod: newResourcePod(schedulernodeinfo.Resource{EphemeralStorage: 10}),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newResourcePod(schedulernodeinfo.Resource{MilliCPU: 2, Memory: 2})),
			fits: true,
			name: "pod fits",
		},
	}

	for _, test := range storagePodsTests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{Status: v1.NodeStatus{Capacity: makeResources(10, 20, 32, 5, 20, 5).Capacity, Allocatable: makeAllocatableResources(10, 20, 32, 5, 20, 5)}}
			test.nodeInfo.SetNode(&node)
			fits, reasons, err := PodFitsResourcesPredicate(test.pod, GetResourceRequest(test.pod), nil, test.nodeInfo)
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

func TestPodFitsHost(t *testing.T) {
	tests := []struct {
		pod  *v1.Pod
		node *v1.Node
		fits bool
		name string
	}{
		{
			pod:  &v1.Pod{},
			node: &v1.Node{},
			fits: true,
			name: "no host specified",
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
			name: "host matches",
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
			name: "host doesn't match",
		},
	}
	expectedFailureReasons := []PredicateFailureReason{ErrPodNotMatchHostName}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nodeInfo := schedulernodeinfo.NewNodeInfo()
			nodeInfo.SetNode(test.node)
			fits, reasons, err := PodFitsHost(test.pod, nil, nodeInfo)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
				t.Errorf("unexpected failure reasons: %v, want: %v", reasons, expectedFailureReasons)
			}
			if fits != test.fits {
				t.Errorf("unexpected difference: expected: %v got %v", test.fits, fits)
			}
		})
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
		nodeInfo *schedulernodeinfo.NodeInfo
		fits     bool
		name     string
	}{
		{
			pod:      &v1.Pod{},
			nodeInfo: schedulernodeinfo.NewNodeInfo(),
			fits:     true,
			name:     "nothing running",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/9090")),
			fits: true,
			name: "other port",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			fits: false,
			name: "same udp port",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			fits: false,
			name: "same tcp port",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.2/8080")),
			fits: true,
			name: "different host ip",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8080")),
			fits: true,
			name: "different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8000", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "UDP/127.0.0.1/8080")),
			fits: false,
			name: "second udp port conflict",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8080"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001", "UDP/127.0.0.1/8081")),
			fits: false,
			name: "first tcp port conflict",
		},
		{
			pod: newPod("m1", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			fits: false,
			name: "first tcp port conflict due to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "TCP/10.0.10.10/8001", "TCP/0.0.0.0/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/127.0.0.1/8001")),
			fits: false,
			name: "TCP hostPort conflict due to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "TCP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			fits: false,
			name: "second tcp port conflict to 0.0.0.0 hostIP",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001")),
			fits: true,
			name: "second different protocol",
		},
		{
			pod: newPod("m1", "UDP/127.0.0.1/8001"),
			nodeInfo: schedulernodeinfo.NewNodeInfo(
				newPod("m1", "TCP/0.0.0.0/8001", "UDP/0.0.0.0/8001")),
			fits: false,
			name: "UDP hostPort conflict due to 0.0.0.0 hostIP",
		},
	}
	expectedFailureReasons := []PredicateFailureReason{ErrPodNotFitsHostPorts}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fits, reasons, err := PodFitsHostPorts(test.pod, nil, test.nodeInfo)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
				t.Errorf("unexpected failure reasons: %v, want: %v", reasons, expectedFailureReasons)
			}
			if test.fits != fits {
				t.Errorf("expected %v, saw %v", test.fits, fits)
			}
		})
	}
}

// TODO: Add test case for RequiredDuringSchedulingRequiredDuringExecution after it's implemented.
func TestPodFitsSelector(t *testing.T) {
	tests := []struct {
		pod      *v1.Pod
		labels   map[string]string
		nodeName string
		fits     bool
		name     string
	}{
		{
			pod:  &v1.Pod{},
			fits: true,
			name: "no selector",
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
			name: "missing labels",
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
			name: "same labels",
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
			name: "node labels are superset",
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
			name: "node labels are subset",
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
			name: "Pod with matchExpressions using In operator that matches the existing node",
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
			name: "Pod with matchExpressions using Gt operator that matches the existing node",
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
			name: "Pod with matchExpressions using NotIn operator that matches the existing node",
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
			name: "Pod with matchExpressions using Exists operator that matches the existing node",
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
			name: "Pod with affinity that don't match node's labels won't schedule onto the node",
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
			name: "Pod with a nil []NodeSelectorTerm in affinity, can't match the node's labels and won't schedule onto the node",
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
			name: "Pod with an empty []NodeSelectorTerm in affinity, can't match the node's labels and won't schedule onto the node",
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
			name: "Pod with empty MatchExpressions is not a valid value will match no objects and won't schedule onto the node",
		},
		{
			pod: &v1.Pod{},
			labels: map[string]string{
				"foo": "bar",
			},
			fits: true,
			name: "Pod with no Affinity will schedule onto a node",
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
			name: "Pod with Affinity but nil NodeSelector will schedule onto a node",
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
			name: "Pod with multiple matchExpressions ANDed that matches the existing node",
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
			name: "Pod with multiple matchExpressions ANDed that doesn't match the existing node",
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
			name: "Pod with multiple NodeSelectorTerms ORed in affinity, matches the node's labels and will schedule onto the node",
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
			name: "Pod with an Affinity and a PodSpec.NodeSelector(the old thing that we are deprecating) " +
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
			name: "Pod with an Affinity matches node's labels but the PodSpec.NodeSelector(the old thing that we are deprecating) " +
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
			name: "Pod with an invalid value in Affinity term won't be scheduled onto the node",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_1",
			fits:     true,
			name:     "Pod with matchFields using In operator that matches the existing node",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_2",
			fits:     false,
			name:     "Pod with matchFields using In operator that does not match the existing node",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
									},
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_2",
			labels:   map[string]string{"foo": "bar"},
			fits:     true,
			name:     "Pod with two terms: matchFields does not match, but matchExpressions matches",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_2",
			labels:   map[string]string{"foo": "bar"},
			fits:     false,
			name:     "Pod with one term: matchFields does not match, but matchExpressions matches",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"bar"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_1",
			labels:   map[string]string{"foo": "bar"},
			fits:     true,
			name:     "Pod with one term: both matchFields and matchExpressions match",
		},
		{
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      api.ObjectNameField,
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"node_1"},
											},
										},
									},
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "foo",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"not-match-to-bar"},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			nodeName: "node_2",
			labels:   map[string]string{"foo": "bar"},
			fits:     false,
			name:     "Pod with two terms: both matchFields and matchExpressions do not match",
		},
	}
	expectedFailureReasons := []PredicateFailureReason{ErrNodeSelectorNotMatch}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			node := v1.Node{ObjectMeta: metav1.ObjectMeta{
				Name:   test.nodeName,
				Labels: test.labels,
			}}
			nodeInfo := schedulernodeinfo.NewNodeInfo()
			nodeInfo.SetNode(&node)

			fits, reasons, err := PodMatchNodeSelector(test.pod, nil, nodeInfo)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
				t.Errorf("unexpected failure reasons: %v, want: %v", reasons, expectedFailureReasons)
			}
			if fits != test.fits {
				t.Errorf("expected: %v got %v", test.fits, fits)
			}
		})
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
				NewInsufficientResourceError(v1.ResourceCPU, 8, 5, 10),
				NewInsufficientResourceError(v1.ResourceMemory, 10, 19, 20),
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
			reasons: []PredicateFailureReason{ErrPodNotMatchHostName},
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
			reasons: []PredicateFailureReason{ErrPodNotFitsHostPorts},
			name:    "hostport conflict",
		},
	}
	for _, test := range resourceTests {
		t.Run(test.name, func(t *testing.T) {
			test.nodeInfo.SetNode(test.node)
			fits, reasons, err := GeneralPredicates(test.pod, nil, test.nodeInfo)
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

func TestPodToleratesTaints(t *testing.T) {
	podTolerateTaintsTests := []struct {
		pod  *v1.Pod
		node v1.Node
		fits bool
		name string
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
			name: "A pod having no tolerations can't be scheduled onto a node with nonempty taints",
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
			name: "A pod which can be scheduled on a dedicated node assigned to user1 with effect NoSchedule",
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
			name: "A pod which can't be scheduled on a dedicated node assigned to user2 with effect NoSchedule",
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
			name: "A pod can be scheduled onto the node, with a toleration uses operator Exists that tolerates the taints on the node",
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
			name: "A pod has multiple tolerations, node has multiple taints, all the taints are tolerated, pod can be scheduled onto the node",
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
			name: "A pod has a toleration that keys and values match the taint on the node, but (non-empty) effect doesn't match, " +
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
			name: "The pod has a toleration that keys and values match the taint on the node, the effect of toleration is empty, " +
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
			name: "The pod has a toleration that key and value don't match the taint on the node, " +
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
			name: "The pod has no toleration, " +
				"but the effect of taint on node is PreferNochedule. Pod can be scheduled onto the node",
		},
	}
	expectedFailureReasons := []PredicateFailureReason{ErrTaintsTolerationsNotMatch}

	for _, test := range podTolerateTaintsTests {
		t.Run(test.name, func(t *testing.T) {
			nodeInfo := schedulernodeinfo.NewNodeInfo()
			nodeInfo.SetNode(&test.node)
			fits, reasons, err := PodToleratesNodeTaints(test.pod, nil, nodeInfo)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if !fits && !reflect.DeepEqual(reasons, expectedFailureReasons) {
				t.Errorf("unexpected failure reason: %v, want: %v", reasons, expectedFailureReasons)
			}
			if fits != test.fits {
				t.Errorf("expected: %v got %v", test.fits, fits)
			}
		})
	}
}
