/*
Copyright 2025 The Kubernetes Authors.

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

package swap_test

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/kubelet/util/swap"
	"k8s.io/utils/ptr"
)

const gb = 1024 * 1024 * 1024

func getTestPod(memoryRequest int64, toSetLimit bool) v1.Pod {
	pod := v1.Pod{Spec: v1.PodSpec{Containers: []v1.Container{{Resources: v1.ResourceRequirements{}}}}}

	if memoryRequest > 0 {
		pod.Spec.Containers[0].Resources.Requests = v1.ResourceList{v1.ResourceMemory: resource.MustParse(fmt.Sprintf("%d", memoryRequest))}

		if toSetLimit {
			pod.Spec.Containers[0].Resources.Limits = v1.ResourceList{v1.ResourceMemory: resource.MustParse(fmt.Sprintf("%d", memoryRequest))}
		}
	}

	return pod
}

type nodeResources struct {
	nodeMemoryCapacity uint64
	nodeSwapCapacity   uint64
}

var nodeWithSwap = nodeResources{
	nodeMemoryCapacity: 512 * gb,
	nodeSwapCapacity:   8 * gb,
}

var nodeWithNoSwap = nodeResources{
	nodeMemoryCapacity: 512 * gb,
	nodeSwapCapacity:   0,
}

func TestLimitedSwapCalculator_IsContainerEligibleForSwap(t *testing.T) {
	tests := []struct {
		name       string
		resources  nodeResources
		isCgroupV1 bool
		// pod calculated is the first container in the pod's spec
		pod                   v1.Pod
		expectEligibleForSwap bool
	}{
		{
			name:                  "container with no request, expected not to be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(0, false),
			expectEligibleForSwap: false,
		},
		{
			name:                  "container with memory request equal to limit, expected not to be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(1234, true),
			expectEligibleForSwap: false,
		},
		{
			name:                  "with cgroup v1, expected not to be eligible for swap",
			resources:             nodeWithSwap,
			isCgroupV1:            true,
			pod:                   getTestPod(1234, false),
			expectEligibleForSwap: false,
		},
		{
			name:                  "a container with no memory limit, expected to be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(1234, false),
			expectEligibleForSwap: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := swap.NewLimitedSwapCalculator(tt.resources.nodeMemoryCapacity, tt.resources.nodeSwapCapacity, tt.isCgroupV1)
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}
			eligibleForSwap, err := l.IsContainerEligibleForSwap(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}

			if eligibleForSwap != tt.expectEligibleForSwap {
				t.Errorf("IsContainerEligibleForSwap() eligibleForSwap = %v, expectEligibleForSwap %v", eligibleForSwap, tt.expectEligibleForSwap)
			}
		})
	}
}

func TestLimitedSwapCalculator_IsPodEligibleForSwap(t *testing.T) {
	tests := []struct {
		name       string
		resources  nodeResources
		isCgroupV1 bool
		// pod calculated is the first container in the pod's spec
		pod                   v1.Pod
		expectEligibleForSwap bool
	}{
		{
			name:                  "best-effort QoS pod, expected not to be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(0, false),
			expectEligibleForSwap: false,
		},
		{
			name:                  "guaranteed QoS pod, expected not to be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(1234, true),
			expectEligibleForSwap: false,
		},
		{
			name:      "critical pod, expected not to be eligible for swap",
			resources: nodeWithSwap,
			pod: func() v1.Pod {
				pod := getTestPod(1234, false)
				pod.Spec.Priority = ptr.To(scheduling.SystemCriticalPriority)
				return pod
			}(),
			expectEligibleForSwap: false,
		},
		{
			name:                  "node with no swap, expected not to be eligible for swap",
			resources:             nodeWithNoSwap,
			pod:                   getTestPod(1234, false),
			expectEligibleForSwap: false,
		},
		{
			name:                  "a pod that should be eligible for swap",
			resources:             nodeWithSwap,
			pod:                   getTestPod(1234, false),
			expectEligibleForSwap: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := swap.NewLimitedSwapCalculator(tt.resources.nodeMemoryCapacity, tt.resources.nodeSwapCapacity, tt.isCgroupV1)
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}
			eligibleForSwap, err := l.IsPodEligibleForSwap(tt.pod)
			if err != nil {
				t.Errorf("IsPodEligibleForSwap() error = %v", err)
				return
			}

			if eligibleForSwap != tt.expectEligibleForSwap {
				t.Errorf("IsPodEligibleForSwap() eligibleForSwap = %v, expectEligibleForSwap %v", eligibleForSwap, tt.expectEligibleForSwap)
			}
		})
	}
}

func TestLimitedSwapCalculator_CalcContainerSwapLimit(t *testing.T) {
	tests := []struct {
		name       string
		resources  nodeResources
		isCgroupV1 bool
		// pod calculated is the first container in the pod's spec
		pod             v1.Pod
		expectSwapLimit int64
	}{
		{
			name:            "container with no request, expected not to be eligible for swap",
			resources:       nodeWithSwap,
			pod:             getTestPod(0, false),
			expectSwapLimit: 0,
		},
		{
			name:            "expect zero limit on nodes with no swap",
			resources:       nodeWithNoSwap,
			pod:             getTestPod(123, false),
			expectSwapLimit: 0,
		},
		{
			name:            "container eligible for swap, ensure correct calculation",
			resources:       nodeWithSwap,
			pod:             getTestPod(5*gb, false),
			expectSwapLimit: int64((float64(5*gb) / float64(nodeWithSwap.nodeMemoryCapacity)) * float64(nodeWithSwap.nodeSwapCapacity)),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := swap.NewLimitedSwapCalculator(tt.resources.nodeMemoryCapacity, tt.resources.nodeSwapCapacity, tt.isCgroupV1)
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}
			swapLimit, err := l.CalcContainerSwapLimit(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}

			if swapLimit != tt.expectSwapLimit {
				t.Errorf("IsContainerEligibleForSwap() swapLimit = %v, expectSwapLimit %v", swapLimit, tt.expectSwapLimit)
			}
		})
	}
}

func TestLimitedSwapCalculator_CalcPodSwapLimit(t *testing.T) {
	tests := []struct {
		name       string
		resources  nodeResources
		isCgroupV1 bool
		// pod calculated is the first container in the pod's spec
		pod             v1.Pod
		expectSwapLimit int64
	}{
		{
			name:            "container with no request, expected not to be eligible for swap",
			resources:       nodeWithSwap,
			pod:             getTestPod(0, false),
			expectSwapLimit: 0,
		},
		{
			name:            "container eligible for swap, ensure correct calculation",
			resources:       nodeWithSwap,
			pod:             getTestPod(5*gb, false),
			expectSwapLimit: int64((float64(5*gb) / float64(nodeWithSwap.nodeMemoryCapacity)) * float64(nodeWithSwap.nodeSwapCapacity)),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l, err := swap.NewLimitedSwapCalculator(tt.resources.nodeMemoryCapacity, tt.resources.nodeSwapCapacity, tt.isCgroupV1)
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}
			swapLimit, err := l.CalcContainerSwapLimit(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewLimitedSwapCalculator() error = %v", err)
			}

			if swapLimit != tt.expectSwapLimit {
				t.Errorf("IsContainerEligibleForSwap() swapLimit = %v, expectSwapLimit %v", swapLimit, tt.expectSwapLimit)
			}
		})
	}
}

func TestNoSwapCalculator_IsContainerEligibleForSwap(t *testing.T) {
	const expectEligibleForSwap = false

	tests := []struct {
		name string
		// pod calculated is the first container in the pod's spec
		pod v1.Pod
	}{
		{
			name: "container with no request, expected not to be eligible for swap",
			pod:  getTestPod(0, false),
		},
		{
			name: "container with memory request equal to limit, expected not to be eligible for swap",
			pod:  getTestPod(1234, true),
		},
		{
			name: "with cgroup v1, expected not to be eligible for swap",
			pod:  getTestPod(1234, false),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := swap.NewNoSwapCalculator()
			eligibleForSwap, err := l.IsContainerEligibleForSwap(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewNoSwapCalculator() error = %v", err)
			}

			if eligibleForSwap != expectEligibleForSwap {
				t.Errorf("IsContainerEligibleForSwap() eligibleForSwap = %v, expectEligibleForSwap %v", eligibleForSwap, expectEligibleForSwap)
			}
		})
	}
}

func TestNoSwapCalculator_IsPodEligibleForSwap(t *testing.T) {
	const expectEligibleForSwap = false

	tests := []struct {
		name string
		// pod calculated is the first container in the pod's spec
		pod v1.Pod
	}{
		{
			name: "container with no request, expected not to be eligible for swap",
			pod: func() v1.Pod {
				pod := getTestPod(1234, false)
				pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{Resources: v1.ResourceRequirements{Requests: v1.ResourceList{}}})
				return pod
			}(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := swap.NewNoSwapCalculator()
			eligibleForSwap, err := l.IsPodEligibleForSwap(tt.pod)
			if err != nil {
				t.Errorf("NewNoSwapCalculator() error = %v", err)
			}

			if eligibleForSwap != expectEligibleForSwap {
				t.Errorf("IsContainerEligibleForSwap() eligibleForSwap = %v, expectEligibleForSwap %v", eligibleForSwap, expectEligibleForSwap)
			}
		})
	}
}

func TestNoSwapCalculator_CalcContainerSwapLimit(t *testing.T) {
	const expectSwapLimit = int64(0)

	tests := []struct {
		name string
		// pod calculated is the first container in the pod's spec
		pod v1.Pod
	}{
		{
			name: "container with no request, expected not to be eligible for swap",
			pod:  getTestPod(0, false),
		},
		{
			name: "container eligible for swap, ensure correct calculation",
			pod:  getTestPod(5*gb, false),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := swap.NewNoSwapCalculator()
			swapLimit, err := l.CalcContainerSwapLimit(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewNoSwapCalculator() error = %v", err)
			}

			if swapLimit != expectSwapLimit {
				t.Errorf("IsContainerEligibleForSwap() swapLimit = %v, expectSwapLimit %v", swapLimit, expectSwapLimit)
			}
		})
	}
}

func TestNoSwapCalculator_CalcPodSwapLimit(t *testing.T) {
	const expectSwapLimit = int64(0)

	tests := []struct {
		name       string
		resources  nodeResources
		isCgroupV1 bool
		// pod calculated is the first container in the pod's spec
		pod v1.Pod
	}{
		{
			name:      "container with no request, expected not to be eligible for swap",
			resources: nodeWithSwap,
			pod:       getTestPod(0, false),
		},
		{
			name:      "container eligible for swap, ensure correct calculation",
			resources: nodeWithSwap,
			pod:       getTestPod(5*gb, false),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			l := swap.NewNoSwapCalculator()
			swapLimit, err := l.CalcContainerSwapLimit(tt.pod, tt.pod.Spec.Containers[0])
			if err != nil {
				t.Errorf("NewNoSwapCalculator() error = %v", err)
			}

			if swapLimit != expectSwapLimit {
				t.Errorf("IsContainerEligibleForSwap() swapLimit = %v, expectSwapLimit %v", swapLimit, expectSwapLimit)
			}
		})
	}
}
