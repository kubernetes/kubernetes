/*
Copyright 2024 The Kubernetes Authors.

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

// Package qos contains helper functions for quality of service.
// For each resource (memory, CPU) Kubelet supports three classes of containers.
// Memory guaranteed containers will receive the highest priority and will get all the resources
// they need.
// Burstable containers will be guaranteed their request and can "burst" and use more resources
// when available.
// Best-Effort containers, which don't specify a request, can use resources only if not being used
// by other pods.

package qos

import (
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
)

// minRegularContainerMemory returns the minimum memory resource quantity
// across all regular containers in pod.Spec.Containers.
// It does not include initContainers (both restartable and non-restartable).
func minRegularContainerMemory(pod *v1.Pod) int64 {
	memoryValue := getEffectiveContainerMemoryRequest(pod, &pod.Spec.Containers[0])
	for i := 1; i < len(pod.Spec.Containers); i++ {
		effectiveMem := getEffectiveContainerMemoryRequest(pod, &pod.Spec.Containers[i])
		if effectiveMem < memoryValue {
			memoryValue = effectiveMem
		}
	}
	return memoryValue
}

// remainingPodMemReqPerContainer calculates the share of the remaining pod memory
// request that container is credited with, by:
// 1. Taking the total pod memory requests
// 2. Subtracting total container memory requests and any DRA memory allocations from pod memory requests
// 3. Dividing the remainder by the number of containers that run at the same time as container.
// This gives us the additional memory request that is not allocated to any
// containers in the pod. This value will be divided equally among the containers
// that compete for it, to calculate oom score adjustment.
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/2837-pod-level-resource-spec/README.md#oom-score-adjustment
// for more details.
// pod.Spec.Resources must be non-nil. Returns 0 if the pod-level memory request is unset.
func remainingPodMemReqPerContainer(pod *v1.Pod, container *v1.Container) int64 {
	var remainingMemory int64
	if pod.Spec.Resources.Requests.Memory().IsZero() {
		return remainingMemory
	}
	numContainers := numCompetingContainers(pod, container)
	if numContainers == 0 {
		return remainingMemory
	}
	// Aggregated requests of all containers (including DRA if enabled).
	opts := resourcehelper.PodResourcesOptions{
		UseExtraNodeAllocatableResources: utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources),
	}
	aggrContainerReqs := resourcehelper.AggregateContainerRequests(pod, opts)
	remainingMemory = pod.Spec.Resources.Requests.Memory().Value() - aggrContainerReqs.Memory().Value()
	remainingMemory = max(remainingMemory, 0)
	remainingMemoryPerContainer := remainingMemory / int64(numContainers)
	return remainingMemoryPerContainer
}

// numCompetingContainers returns the number of containers that run at the same time as
// container, and that therefore share the part of the pod's memory request which no
// container asked for. The regular containers and the restartable init containers all run
// together. A non-restartable init container runs before every regular container and
// before every restartable init container declared after it, so it shares that memory
// only with the restartable init containers declared before it, which is the same set
// AggregateContainerRequests charges to it.
func numCompetingContainers(pod *v1.Pod, container *v1.Container) int {
	for i := range pod.Spec.InitContainers {
		if pod.Spec.InitContainers[i].Name != container.Name {
			continue
		}
		if podutil.IsRestartableInitContainer(&pod.Spec.InitContainers[i]) {
			break
		}
		return 1 + numRestartableInitContainers(pod.Spec.InitContainers[:i])
	}
	return len(pod.Spec.Containers) + numRestartableInitContainers(pod.Spec.InitContainers)
}

// numRestartableInitContainers returns how many of containers are restartable init containers.
func numRestartableInitContainers(containers []v1.Container) int {
	var n int
	for i := range containers {
		if podutil.IsRestartableInitContainer(&containers[i]) {
			n++
		}
	}
	return n
}
