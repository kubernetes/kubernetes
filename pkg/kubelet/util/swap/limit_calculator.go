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

package swap

import (
	"fmt"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/types"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
)

func NewLimitCalculator(nodeMemoryCapacity, nodeSwapCapacity uint64, isCgroupV1 bool, swapBehavior string) (LimitCalculator, error) {
	switch swapBehavior {
	case types.LimitedSwap:
		return NewLimitedSwapCalculator(nodeMemoryCapacity, nodeSwapCapacity, isCgroupV1)
	case types.NoSwap:
		return NewNoSwapCalculator(), nil
	default:
		return nil, fmt.Errorf("unknown swap behavior %s", swapBehavior)
	}
}

type LimitCalculator interface {
	IsContainerEligibleForSwap(pod v1.Pod, container v1.Container) (bool, error)
	IsPodEligibleForSwap(pod v1.Pod) (bool, error)
	CalcContainerSwapLimit(pod v1.Pod, container v1.Container) (int64, error)
	CalcPodSwapLimit(pod v1.Pod) (int64, error)
}

func calcPodSwapLimit(limitCalculator LimitCalculator, pod v1.Pod) (int64, error) {
	swapLimit := int64(0)
	for _, container := range pod.Spec.Containers {
		limit, err := limitCalculator.CalcContainerSwapLimit(pod, container)
		if err != nil {
			return 0, err
		}

		swapLimit += limit
	}

	return swapLimit, nil
}

func atLeastOneContainerEligibleForSwap(limitCalculator LimitCalculator, pod v1.Pod) (bool, error) {
	for _, container := range pod.Spec.Containers {
		eligible, err := limitCalculator.IsContainerEligibleForSwap(pod, container)
		if err != nil {
			return false, err
		}

		if eligible {
			return true, nil
		}
	}

	return false, nil
}

type LimitedSwapCalculator struct {
	nodeMemoryCapacity uint64
	nodeSwapCapacity   uint64
	isCgroupV1         bool
}

func NewLimitedSwapCalculator(nodeMemoryCapacity, nodeSwapCapacity uint64, isCgroupV1 bool) (*LimitedSwapCalculator, error) {
	if nodeMemoryCapacity <= 0 {
		return nil, fmt.Errorf("total node memory is 0")
	}

	if nodeSwapCapacity <= 0 {
		klog.InfoS("LimitedSwapCalculator is initialized with total node swap is 0, swap limit will always be set to zero")
	}

	return &LimitedSwapCalculator{
		nodeMemoryCapacity: nodeMemoryCapacity,
		nodeSwapCapacity:   nodeSwapCapacity,
		isCgroupV1:         isCgroupV1,
	}, nil
}

func (l *LimitedSwapCalculator) isPodRuledOutForSwap(pod v1.Pod) bool {
	if qos.GetPodQOS(&pod) != v1.PodQOSBurstable {
		return true
	}
	if types.IsCriticalPod(&pod) {
		return true
	}
	if l.nodeSwapCapacity <= 0 {
		return true
	}

	return false
}

func (l *LimitedSwapCalculator) IsContainerEligibleForSwap(pod v1.Pod, container v1.Container) (bool, error) {
	if l.isPodRuledOutForSwap(pod) {
		return false, nil
	}
	if container.Resources.Requests == nil || container.Resources.Requests.Memory() == nil {
		return false, nil
	}

	containerDoesNotRequestMemory := container.Resources.Requests.Memory().IsZero() && container.Resources.Limits.Memory().IsZero()
	memoryRequestEqualsToLimit := container.Resources.Requests.Memory().Cmp(*container.Resources.Limits.Memory()) == 0

	if containerDoesNotRequestMemory || l.isCgroupV1 || memoryRequestEqualsToLimit {
		return false, nil
	}

	return true, nil
}

func (l *LimitedSwapCalculator) IsPodEligibleForSwap(pod v1.Pod) (bool, error) {
	if l.isPodRuledOutForSwap(pod) {
		return false, nil
	}

	hasContainerEligibleForSwap, err := atLeastOneContainerEligibleForSwap(l, pod)
	if err != nil {
		return false, err
	}

	return hasContainerEligibleForSwap, nil
}

func (l *LimitedSwapCalculator) CalcContainerSwapLimit(pod v1.Pod, container v1.Container) (int64, error) {
	containerEligibleForSwap, err := l.IsContainerEligibleForSwap(pod, container)
	if err != nil {
		return 0, err
	}

	if !containerEligibleForSwap {
		return 0, nil
	}

	containerMemoryRequest := container.Resources.Requests.Memory().Value()
	if containerMemoryRequest > int64(l.nodeMemoryCapacity) {
		return 0, fmt.Errorf("container request %d is larger than total node memory %d", containerMemoryRequest, l.nodeMemoryCapacity)
	}

	containerMemoryProportion := float64(containerMemoryRequest) / float64(l.nodeMemoryCapacity)
	swapAllocation := containerMemoryProportion * float64(l.nodeSwapCapacity)

	return int64(swapAllocation), nil
}

func (l *LimitedSwapCalculator) CalcPodSwapLimit(pod v1.Pod) (int64, error) {
	if !l.isPodRuledOutForSwap(pod) {
		return 0, nil
	}

	return calcPodSwapLimit(l, pod)
}

type NoSwapCalculator struct{}

func NewNoSwapCalculator() *NoSwapCalculator {
	return &NoSwapCalculator{}
}

func (n NoSwapCalculator) IsContainerEligibleForSwap(_ v1.Pod, _ v1.Container) (bool, error) {
	return false, nil
}

func (n NoSwapCalculator) IsPodEligibleForSwap(pod v1.Pod) (bool, error) {
	return false, nil
}

func (n NoSwapCalculator) CalcContainerSwapLimit(_ v1.Pod, _ v1.Container) (int64, error) {
	return 0, nil
}

func (n NoSwapCalculator) CalcPodSwapLimit(pod v1.Pod) (int64, error) {
	return 0, nil
}
