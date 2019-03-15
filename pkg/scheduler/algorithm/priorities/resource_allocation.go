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
	"strings"

	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// ResourceAllocationPriority contains information to calculate resource allocation priority.
type ResourceAllocationPriority struct {
	Name   string
	scorer func(requested, allocable *schedulernodeinfo.Resource, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64
}

// PriorityMap priorities nodes according to the resource allocations on the node.
// It will use `scorer` function to calculate the score.
func (r *ResourceAllocationPriority) PriorityMap(
	pod *v1.Pod,
	meta interface{},
	nodeInfo *schedulernodeinfo.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}
	allocatable := nodeInfo.AllocatableResource()

	var requested schedulernodeinfo.Resource
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		requested = *priorityMeta.nonZeroRequest
	} else {
		// We couldn't parse metadata - fallback to computing it.
		requested = *getNonZeroRequests(pod)
	}

	requested.MilliCPU += nodeInfo.NonZeroRequest().MilliCPU
	requested.Memory += nodeInfo.NonZeroRequest().Memory

	// If we are using a extended resources priority scorer,
	// then we need to get requests of resources besides CPU and memory
	if strings.Contains(r.Name, "Extended") {
		getOtherRequestsFromPod(pod, &requested)
		getOtherRequestsOnNode(nodeInfo, &requested)
	}

	var score int64
	// Check if the pod has volumes and this could be added to scorer function for balanced resource allocation.
	if len(pod.Spec.Volumes) >= 0 && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && nodeInfo.TransientInfo != nil {
		score = r.scorer(&requested, &allocatable, true, nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes, nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount)
	} else {
		score = r.scorer(&requested, &allocatable, false, 0, 0)
	}

	if klog.V(10) {
		if len(pod.Spec.Volumes) >= 0 && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && nodeInfo.TransientInfo != nil {
			klog.Infof(
				"%v -> %v: %v, capacity %d millicores %d memory bytes, %d volumes, total request %d millicores %d memory bytes %d volumes, score %d",
				pod.Name, node.Name, r.Name,
				allocatable.MilliCPU, allocatable.Memory, nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount,
				requested.MilliCPU, requested.Memory,
				nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes,
				score,
			)
		} else {
			klog.Infof(
				"%v -> %v: %v, capacity %d millicores %d memory bytes, total request %d millicores %d memory bytes, score %d",
				pod.Name, node.Name, r.Name,
				allocatable.MilliCPU, allocatable.Memory,
				requested.MilliCPU, requested.Memory,
				score,
			)
		}
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int(score),
	}, nil
}

func getNonZeroRequests(pod *v1.Pod) *schedulernodeinfo.Resource {
	result := &schedulernodeinfo.Resource{}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		result.MilliCPU += cpu
		result.Memory += memory
	}
	return result
}

// getOtherRequestsFromPod gets requests besides CPU and memory from the pod
func getOtherRequestsFromPod(pod *v1.Pod, requested *schedulernodeinfo.Resource) {
	if requested.ScalarResources == nil {
		requested.ScalarResources = make(map[v1.ResourceName]int64)
	}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		// Ephemeral storage
		if value, ok := container.Resources.Requests[v1.ResourceEphemeralStorage]; ok {
			requested.EphemeralStorage += value.Value()
		}
		// Other scalar resources
		for key, value := range container.Resources.Requests {
			if key != v1.ResourceCPU && key != v1.ResourceMemory && key != v1.ResourceEphemeralStorage {
				requested.ScalarResources[key] += value.Value()
			}
		}
	}
}

// getOtherRequestsOnNode gets requests besides CPU and memory based on the node info
func getOtherRequestsOnNode(nodeInfo *schedulernodeinfo.NodeInfo, requested *schedulernodeinfo.Resource) {
	if requested.ScalarResources == nil {
		requested.ScalarResources = make(map[v1.ResourceName]int64)
	}
	requested.EphemeralStorage += nodeInfo.RequestedResource().EphemeralStorage
	for key, value := range nodeInfo.RequestedResource().ScalarResources {
		if key != v1.ResourceCPU && key != v1.ResourceMemory {
			requested.ScalarResources[key] += value
		}
	}
}
