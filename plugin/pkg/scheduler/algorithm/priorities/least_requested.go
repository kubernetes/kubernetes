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

	"k8s.io/api/core/v1"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

// LeastRequestedPriority is a priority function that favors nodes with fewer requested resources.
// It calculates the percentage of memory and CPU requested by pods scheduled on the node, and prioritizes
// based on the minimum of the average of the fraction of requested to capacity.
// Details: cpu((capacity - sum(requested)) * 10 / capacity) + memory((capacity - sum(requested)) * 10 / capacity) / 2
func LeastRequestedPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	var nonZeroRequest *schedulercache.Resource
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		nonZeroRequest = priorityMeta.nonZeroRequest
	} else {
		// We couldn't parse metadata - fallback to computing it.
		nonZeroRequest = getNonZeroRequests(pod)
	}
	return calculateUnusedPriority(pod, nonZeroRequest, nodeInfo)
}

// The unused capacity is calculated on a scale of 0-10
// 0 being the lowest priority and 10 being the highest.
// The more unused resources the higher the score is.
func calculateUnusedScore(requested int64, capacity int64, node string) int64 {
	if capacity == 0 {
		return 0
	}
	if requested > capacity {
		glog.V(10).Infof("Combined requested resources %d from existing pods exceeds capacity %d on node %s",
			requested, capacity, node)
		return 0
	}
	return ((capacity - requested) * int64(schedulerapi.MaxPriority)) / capacity
}

// Calculates host priority based on the amount of unused resources.
// 'node' has information about the resources on the node.
// 'pods' is a list of pods currently scheduled on the node.
func calculateUnusedPriority(pod *v1.Pod, podRequests *schedulercache.Resource, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	allocatableResources := nodeInfo.AllocatableResource()
	totalResources := *podRequests
	totalResources.MilliCPU += nodeInfo.NonZeroRequest().MilliCPU
	totalResources.Memory += nodeInfo.NonZeroRequest().Memory

	cpuScore := calculateUnusedScore(totalResources.MilliCPU, allocatableResources.MilliCPU, node.Name)
	memoryScore := calculateUnusedScore(totalResources.Memory, allocatableResources.Memory, node.Name)
	if glog.V(10) {
		// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		glog.V(10).Infof(
			"%v -> %v: Least Requested Priority, capacity %d millicores %d memory bytes, total request %d millicores %d memory bytes, score %d CPU %d memory",
			pod.Name, node.Name,
			allocatableResources.MilliCPU, allocatableResources.Memory,
			totalResources.MilliCPU, totalResources.Memory,
			cpuScore, memoryScore,
		)
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int((cpuScore + memoryScore) / 2),
	}, nil
}
