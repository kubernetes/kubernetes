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
	"fmt"
	"math"

	"k8s.io/api/core/v1"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

// This is a reasonable size range of all container images. 90%ile of images on dockerhub drops into this range.
const (
	mb         int64 = 1024 * 1024
	minImgSize int64 = 23 * mb
	maxImgSize int64 = 1000 * mb
)

// Also used in most/least_requested nad metadata.
// TODO: despaghettify it
func getNonZeroRequests(pod *v1.Pod) *schedulercache.Resource {
	result := &schedulercache.Resource{}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		result.MilliCPU += cpu
		result.Memory += memory
	}
	return result
}

func calculateBalancedResourceAllocation(pod *v1.Pod, podRequests *schedulercache.Resource, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	allocatableResources := nodeInfo.AllocatableResource()
	totalResources := *podRequests
	totalResources.MilliCPU += nodeInfo.NonZeroRequest().MilliCPU
	totalResources.Memory += nodeInfo.NonZeroRequest().Memory

	cpuFraction := fractionOfCapacity(totalResources.MilliCPU, allocatableResources.MilliCPU)
	memoryFraction := fractionOfCapacity(totalResources.Memory, allocatableResources.Memory)
	score := int(0)
	if cpuFraction >= 1 || memoryFraction >= 1 {
		// if requested >= capacity, the corresponding host should never be preferred.
		score = 0
	} else {
		// Upper and lower boundary of difference between cpuFraction and memoryFraction are -1 and 1
		// respectively. Multilying the absolute value of the difference by 10 scales the value to
		// 0-10 with 0 representing well balanced allocation and 10 poorly balanced. Subtracting it from
		// 10 leads to the score which also scales from 0 to 10 while 10 representing well balanced.
		diff := math.Abs(cpuFraction - memoryFraction)
		score = int((1 - diff) * float64(schedulerapi.MaxPriority))
	}
	if glog.V(10) {
		// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		glog.V(10).Infof(
			"%v -> %v: Balanced Resource Allocation, capacity %d millicores %d memory bytes, total request %d millicores %d memory bytes, score %d",
			pod.Name, node.Name,
			allocatableResources.MilliCPU, allocatableResources.Memory,
			totalResources.MilliCPU, totalResources.Memory,
			score,
		)
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: score,
	}, nil
}

func fractionOfCapacity(requested, capacity int64) float64 {
	if capacity == 0 {
		return 1
	}
	return float64(requested) / float64(capacity)
}

// BalancedResourceAllocationMap favors nodes with balanced resource usage rate.
// BalancedResourceAllocationMap should **NOT** be used alone, and **MUST** be used together with LeastRequestedPriority.
// It calculates the difference between the cpu and memory fracion of capacity, and prioritizes the host based on how
// close the two metrics are to each other.
// Detail: score = 10 - abs(cpuFraction-memoryFraction)*10. The algorithm is partly inspired by:
// "Wei Huang et al. An Energy Efficient Virtual Machine Placement Algorithm with Balanced Resource Utilization"
func BalancedResourceAllocationMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	var nonZeroRequest *schedulercache.Resource
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		nonZeroRequest = priorityMeta.nonZeroRequest
	} else {
		// We couldn't parse metadatat - fallback to computing it.
		nonZeroRequest = getNonZeroRequests(pod)
	}
	return calculateBalancedResourceAllocation(pod, nonZeroRequest, nodeInfo)
}
