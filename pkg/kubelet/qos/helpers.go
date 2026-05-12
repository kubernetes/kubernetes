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
	resourcehelper "k8s.io/component-helpers/resource"
)

// minRegularContainerMemory returns the minimum memory resource quantity
// across all regular containers in pod.Spec.Containers.
// It does not include initContainers (both restartable and non-restartable).
func minRegularContainerMemory(pod v1.Pod) int64 {
	memoryValue := pod.Spec.Containers[0].Resources.Requests.Memory().Value()
	for _, container := range pod.Spec.Containers[1:] {
		if container.Resources.Requests.Memory().Value() < memoryValue {
			memoryValue = container.Resources.Requests.Memory().Value()
		}
	}
	return memoryValue
}

// remainingPodMemReqPerContainer calculates the remaining pod memory request per
// container by:
// 1. Taking the total pod memory requests
// 2. Subtracting total container memory requests from pod memory requests
// 3. Dividing the remainder by the number of containers.
// This gives us the additional memory request that is not allocated to any
// containers in the pod. This value will be divided equally among all containers to
// calculate oom score adjusment.
// See https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/2837-pod-level-resource-spec/README.md#oom-score-adjustment
// for more details.
func remainingPodMemReqPerContainer(pod *v1.Pod) int64 {
	var remainingMemory int64
	if pod.Spec.Resources.Requests.Memory().IsZero() {
		return remainingMemory
	}

	numContainers := len(pod.Spec.Containers) + len(pod.Spec.InitContainers)

	// Aggregated requests of all containers.
	aggrContainerReqs := resourcehelper.AggregateContainerRequests(pod, resourcehelper.PodResourcesOptions{})

	remainingMemory = pod.Spec.Resources.Requests.Memory().Value() - aggrContainerReqs.Memory().Value()

	remainingMemoryPerContainer := remainingMemory / int64(numContainers)
	return remainingMemoryPerContainer
}
