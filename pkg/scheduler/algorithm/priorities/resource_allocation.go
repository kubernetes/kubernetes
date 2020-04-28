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

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// ResourceAllocationPriority contains information to calculate resource allocation priority.
type ResourceAllocationPriority struct {
	Name                string
	scorer              func(requested, allocable ResourceToValueMap, includeVolumes bool, requestedVolumes int, allocatableVolumes int) int64
	resourceToWeightMap ResourceToWeightMap
}

// ResourceToWeightMap contains resource name and weight.
type ResourceToWeightMap map[v1.ResourceName]int64

// ResourceToValueMap contains resource name and score.
type ResourceToValueMap map[v1.ResourceName]int64

// DefaultRequestedRatioResources is used to set default requestToWeight map for CPU and memory
var DefaultRequestedRatioResources = ResourceToWeightMap{v1.ResourceMemory: 1, v1.ResourceCPU: 1}

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
	if r.resourceToWeightMap == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("resources not found")
	}
	requested := make(ResourceToValueMap, len(r.resourceToWeightMap))
	allocatable := make(ResourceToValueMap, len(r.resourceToWeightMap))
	for resource := range r.resourceToWeightMap {
		allocatable[resource], requested[resource] = calculateResourceAllocatableRequest(nodeInfo, pod, resource)
	}
	var score int64

	// Check if the pod has volumes and this could be added to scorer function for balanced resource allocation.
	if len(pod.Spec.Volumes) >= 0 && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && nodeInfo.TransientInfo != nil {
		score = r.scorer(requested, allocatable, true, nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes, nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount)
	} else {
		score = r.scorer(requested, allocatable, false, 0, 0)
	}
	if klog.V(10) {
		if len(pod.Spec.Volumes) >= 0 && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) && nodeInfo.TransientInfo != nil {
			klog.Infof(
				"%v -> %v: %v, map of allocatable resources %v, map of requested resources %v , allocatable volumes %d, requested volumes %d, score %d",
				pod.Name, node.Name, r.Name,
				allocatable, requested, nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount,
				nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes,
				score,
			)
		} else {
			klog.Infof(
				"%v -> %v: %v, map of allocatable resources %v, map of requested resources %v ,score %d,",
				pod.Name, node.Name, r.Name,
				allocatable, requested, score,
			)

		}
	}

	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int(score),
	}, nil
}

// calculateResourceAllocatableRequest returns resources Allocatable and Requested values
func calculateResourceAllocatableRequest(nodeInfo *schedulernodeinfo.NodeInfo, pod *v1.Pod, resource v1.ResourceName) (int64, int64) {
	allocatable := nodeInfo.AllocatableResource()
	requested := nodeInfo.RequestedResource()
	podRequest := calculatePodResourceRequest(pod, resource)
	switch resource {
	case v1.ResourceCPU:
		return allocatable.MilliCPU, (nodeInfo.NonZeroRequest().MilliCPU + podRequest)
	case v1.ResourceMemory:
		return allocatable.Memory, (nodeInfo.NonZeroRequest().Memory + podRequest)

	case v1.ResourceEphemeralStorage:
		return allocatable.EphemeralStorage, (requested.EphemeralStorage + podRequest)
	default:
		if v1helper.IsScalarResourceName(resource) {
			return allocatable.ScalarResources[resource], (requested.ScalarResources[resource] + podRequest)
		}
	}
	if klog.V(10) {
		klog.Infof("requested resource %v not considered for node score calculation",
			resource,
		)
	}
	return 0, 0
}

// calculatePodResourceRequest returns the total non-zero requests. If Overhead is defined for the pod and the
// PodOverhead feature is enabled, the Overhead is added to the result.
// podResourceRequest = max(sum(podSpec.Containers), podSpec.InitContainers) + overHead
func calculatePodResourceRequest(pod *v1.Pod, resource v1.ResourceName) int64 {
	var podRequest int64
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		value := priorityutil.GetNonzeroRequestForResource(resource, &container.Resources.Requests)
		podRequest += value
	}

	for i := range pod.Spec.InitContainers {
		initContainer := &pod.Spec.InitContainers[i]
		value := priorityutil.GetNonzeroRequestForResource(resource, &initContainer.Resources.Requests)
		if podRequest < value {
			podRequest = value
		}
	}

	// If Overhead is being utilized, add to the total requests for the pod
	if pod.Spec.Overhead != nil && utilfeature.DefaultFeatureGate.Enabled(features.PodOverhead) {
		if quantity, found := pod.Spec.Overhead[resource]; found {
			podRequest += quantity.Value()
		}
	}

	return podRequest
}
