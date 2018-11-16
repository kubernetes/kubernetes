/*
Copyright 2017 The Kubernetes Authors.

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
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"

	"k8s.io/klog"
)

// ResourceLimitsPriorityMap is a priority function that increases score of input node by 1 if the node satisfies
// input pod's resource limits. In detail, this priority function works as follows: If a node does not publish its
// allocatable resources (cpu and memory both), the node score is not affected. If a pod does not specify
// its cpu and memory limits both, the node score is not affected. If one or both of cpu and memory limits
// of the pod are satisfied, the node is assigned a score of 1.
// Rationale of choosing the lowest score of 1 is that this is mainly selected to break ties between nodes that have
// same scores assigned by one of least and most requested priority functions.
func ResourceLimitsPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	allocatableResources := nodeInfo.AllocatableResource()

	// compute pod limits
	podLimits := getResourceLimits(pod)

	cpuScore := computeScore(podLimits.MilliCPU, allocatableResources.MilliCPU)
	memScore := computeScore(podLimits.Memory, allocatableResources.Memory)

	score := int(0)
	if cpuScore == 1 || memScore == 1 {
		score = 1
	}

	if klog.V(10) {
		// We explicitly don't do klog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		klog.Infof(
			"%v -> %v: Resource Limits Priority, allocatable %d millicores %d memory bytes, pod limits %d millicores %d memory bytes, score %d",
			pod.Name, node.Name,
			allocatableResources.MilliCPU, allocatableResources.Memory,
			podLimits.MilliCPU, podLimits.Memory,
			score,
		)
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: score,
	}, nil
}

// computeScore returns 1 if limit value is less than or equal to allocatable
// value, otherwise it returns 0.
func computeScore(limit, allocatable int64) int64 {
	if limit != 0 && allocatable != 0 && limit <= allocatable {
		return 1
	}
	return 0
}

// getResourceLimits computes resource limits for input pod.
// The reason to create this new function is to be consistent with other
// priority functions because most or perhaps all priority functions work
// with schedulercache.Resource.
// TODO: cache it as part of metadata passed to priority functions.
func getResourceLimits(pod *v1.Pod) *schedulercache.Resource {
	result := &schedulercache.Resource{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Limits)
	}

	// take max_resource(sum_pod, any_init_container)
	for _, container := range pod.Spec.InitContainers {
		result.SetMaxResource(container.Resources.Limits)
	}

	return result
}
