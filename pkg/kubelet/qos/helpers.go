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

package qos // import "k8s.io/kubernetes/pkg/kubelet/qos"

import (
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// minRegularContainerMemory returns the minimum memory resource quantity
// across all regular containers in pod.Spec.Containers.
// It does not include initContainers (both restartable and non-restartable).
func minRegularContainerMemory(pod v1.Pod) int64 {
	var containerStatuses map[string]*v1.ContainerStatus
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		containerStatuses = make(map[string]*v1.ContainerStatus, len(pod.Status.ContainerStatuses))
		for i := range pod.Status.ContainerStatuses {
			containerStatuses[pod.Status.ContainerStatuses[i].Name] = &pod.Status.ContainerStatuses[i]
		}
	}
	memoryValue := int64(0)
	for _, container := range pod.Spec.Containers {
		memoryValue = container.Resources.Requests.Memory().Value()
		if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
			cs, found := containerStatuses[container.Name]
			if found {
				if pod.Status.Resize == v1.PodResizeStatusInfeasible {
					memoryValue = cs.AllocatedResources.Memory().Value()
				} else if cs.AllocatedResources.Memory().Value() > memoryValue {
					memoryValue = cs.AllocatedResources.Memory().Value()
				}
			}
		}

		// min memory quantity
		if memoryValue > container.Resources.Requests.Memory().Value() {
			memoryValue = container.Resources.Requests.Memory().Value()
		}
	}
	return memoryValue
}
