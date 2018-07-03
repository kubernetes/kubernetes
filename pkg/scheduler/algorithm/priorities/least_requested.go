/*
Copyright 2016 The Kubernetes Authors.

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
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
)

var (
	leastResourcePriority = &ResourceAllocationPriority{"LeastResourceAllocation", leastResourceScorer}

	// LeastRequestedPriorityMap is a priority function that favors nodes with fewer requested resources.
	// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and
	// prioritizes based on the minimum of the average of the fraction of requested to capacity.
	//
	// Details:
	// (cpu((capacity-sum(requested))*10/capacity) + memory((capacity-sum(requested))*10/capacity))/2
	LeastRequestedPriorityMap = leastResourcePriority.PriorityMap
)

func leastResourceScorer(requested, allocable *schedulercache.Resource, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	return (leastRequestedScore(requested.MilliCPU, allocable.MilliCPU) +
		leastRequestedScore(requested.Memory, allocable.Memory)) / 2
}

// The unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest.
// The more unused resources the higher the score is.
func leastRequestedScore(requested, capacity int64) int64 {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		return 0
	}

	return ((capacity - requested) * int64(schedulerapi.MaxPriority)) / capacity
}
