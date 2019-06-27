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
	"math"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

var (
	balancedResourcePriority = &ResourceAllocationPriority{"BalancedResourceAllocation", balancedResourceScorer}

	// BalancedResourceAllocationMap favors nodes with balanced resource usage rate.
	// BalancedResourceAllocationMap should **NOT** be used alone, and **MUST** be used together
	// with LeastRequestedPriority. It calculates the difference between the cpu and memory fraction
	// of capacity, and prioritizes the host based on how close the two metrics are to each other.
	// Detail: score = 10 - variance(cpuFraction,memoryFraction,volumeFraction)*10. The algorithm is partly inspired by:
	// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced
	// Resource Utilization"
	BalancedResourceAllocationMap = balancedResourcePriority.PriorityMap
)

func balancedResourceScorer(requested, allocable *schedulernodeinfo.Resource, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	cpuFraction := fractionOfCapacity(requested.MilliCPU, allocable.MilliCPU)
	memoryFraction := fractionOfCapacity(requested.Memory, allocable.Memory)
	// This to find a node which has most balanced CPU, memory and volume usage.
	if cpuFraction >= 1 || memoryFraction >= 1 {
		// if requested >= capacity, the corresponding host should never be preferred.
		return 0
	}

	if includeVolumes && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && allocatableVolumes > 0 {
		volumeFraction := float64(requestedVolumes) / float64(allocatableVolumes)
		if volumeFraction >= 1 {
			// if requested >= capacity, the corresponding host should never be preferred.
			return 0
		}
		// Compute variance for all the three fractions.
		mean := (cpuFraction + memoryFraction + volumeFraction) / float64(3)
		variance := float64((((cpuFraction - mean) * (cpuFraction - mean)) + ((memoryFraction - mean) * (memoryFraction - mean)) + ((volumeFraction - mean) * (volumeFraction - mean))) / float64(3))
		// Since the variance is between positive fractions, it will be positive fraction. 1-variance lets the
		// score to be higher for node which has least variance and multiplying it with 10 provides the scaling
		// factor needed.
		return int64((1 - variance) * float64(schedulerapi.MaxPriority))
	}

	// Upper and lower boundary of difference between cpuFraction and memoryFraction are -1 and 1
	// respectively. Multiplying the absolute value of the difference by 10 scales the value to
	// 0-10 with 0 representing well balanced allocation and 10 poorly balanced. Subtracting it from
	// 10 leads to the score which also scales from 0 to 10 while 10 representing well balanced.
	diff := math.Abs(cpuFraction - memoryFraction)
	return int64((1 - diff) * float64(schedulerapi.MaxPriority))
}

func fractionOfCapacity(requested, capacity int64) float64 {
	if capacity == 0 {
		return 1
	}
	return float64(requested) / float64(capacity)
}
