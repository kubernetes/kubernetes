/*
Copyright 2019 The Kubernetes Authors.

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

package priorities

import (
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	leastExtendedResourcePriority = &ResourceAllocationPriority{"LeastResourceAllocationExtended", leastResourceScorerExtended}

	// LeastExtendedRequestedPriorityMap is a priority function that favors nodes with fewer requested resources.
	// It calculates the percentage of CPU, memory, EphemeralStorage as well as Extended resources requested by pods scheduled on the node, and
	// prioritizes based on the minimum of the average of the fraction of requested to allocable.
	//
	// Details: (cpu((allocable-sum(requested))*10/allocable)
	//         + memory((allocable-sum(requested))*10/allocable)
	//         + ephemeralStorage((allocable-sum(requested))*10/allocable)
	//         + sigmaOfAllExtendedResources((allocable-sum(requested))*10/allocable)) / numOfResources
	LeastExtendedRequestedPriorityMap = leastExtendedResourcePriority.PriorityMap
)

func leastResourceScorerExtended(requested, allocable *schedulernodeinfo.Resource, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	var result float32
	numOfResources := 2.0
	// CPU and memory
	result += leastRequestedScoreExtended(float32(requested.MilliCPU), float32(allocable.MilliCPU))
	result += leastRequestedScoreExtended(float32(requested.Memory), float32(allocable.Memory))
	// EphemeralStorage
	if allocable.EphemeralStorage != 0 {
		result += leastRequestedScoreExtended(float32(requested.EphemeralStorage), float32(allocable.EphemeralStorage))
		numOfResources += 1
	}
	// ScalarResources
	for key, value := range allocable.ScalarResources {
		result += leastRequestedScoreExtended(float32(requested.ScalarResources[key]), float32(value))
	}
	numOfResources += float64(len(allocable.ScalarResources))
	return int64(float32(result) / float32(numOfResources))
}

// The unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest.
// The more unused resources the higher the score is.
func leastRequestedScoreExtended(requested, capacity float32) float32 {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		return 0
	}

	return ((capacity - requested) * schedulerapi.MaxPriority) / capacity
}
