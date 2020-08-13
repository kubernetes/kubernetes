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

package noderesources

import (
	"context"
	"fmt"
	"math"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// BalancedAllocation is a score plugin that calculates the difference between the cpu and memory fraction
// of capacity, and prioritizes the host based on how close the two metrics are to each other.
type BalancedAllocation struct {
	handle framework.FrameworkHandle
	resourceAllocationScorer
}

var _ = framework.ScorePlugin(&BalancedAllocation{})

// BalancedAllocationName is the name of the plugin used in the plugin registry and configurations.
const BalancedAllocationName = "NodeResourcesBalancedAllocation"

// Name returns name of the plugin. It is used in logs, etc.
func (ba *BalancedAllocation) Name() string {
	return BalancedAllocationName
}

// Score invoked at the score extension point.
func (ba *BalancedAllocation) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := ba.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	// ba.score favors nodes with balanced resource usage rate.
	// It calculates the difference between the cpu and memory fraction of capacity,
	// and prioritizes the host based on how close the two metrics are to each other.
	// Detail: score = (1 - variance(cpuFraction,memoryFraction,volumeFraction)) * MaxNodeScore. The algorithm is partly inspired by:
	// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced
	// Resource Utilization"
	return ba.score(pod, nodeInfo)
}

// ScoreExtensions of the Score plugin.
func (ba *BalancedAllocation) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

// NewBalancedAllocation initializes a new plugin and returns it.
func NewBalancedAllocation(_ runtime.Object, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &BalancedAllocation{
		handle: h,
		resourceAllocationScorer: resourceAllocationScorer{
			BalancedAllocationName,
			balancedResourceScorer,
			defaultRequestedRatioResources,
		},
	}, nil
}

// todo: use resource weights in the scorer function
func balancedResourceScorer(requested, allocable resourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64 {
	cpuFraction := fractionOfCapacity(requested[v1.ResourceCPU], allocable[v1.ResourceCPU])
	memoryFraction := fractionOfCapacity(requested[v1.ResourceMemory], allocable[v1.ResourceMemory])
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
		// score to be higher for node which has least variance and multiplying it with `MaxNodeScore` provides the scaling
		// factor needed.
		return int64((1 - variance) * float64(framework.MaxNodeScore))
	}

	// Upper and lower boundary of difference between cpuFraction and memoryFraction are -1 and 1
	// respectively. Multiplying the absolute value of the difference by `MaxNodeScore` scales the value to
	// 0-MaxNodeScore with 0 representing well balanced allocation and `MaxNodeScore` poorly balanced. Subtracting it from
	// `MaxNodeScore` leads to the score which also scales from 0 to `MaxNodeScore` while `MaxNodeScore` representing well balanced.
	diff := math.Abs(cpuFraction - memoryFraction)
	return int64((1 - diff) * float64(framework.MaxNodeScore))
}

func fractionOfCapacity(requested, capacity int64) float64 {
	if capacity == 0 {
		return 1
	}
	return float64(requested) / float64(capacity)
}
