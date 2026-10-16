/*
Copyright The Kubernetes Authors.

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

package helper

import (
	"fmt"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/resource"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// ResourceRequestsOptions contains feature gate flags for resource request computation.
type ResourceRequestsOptions struct {
	EnablePodLevelResources                            bool
	EnableDRAExtendedResource                          bool
	EnableDRANodeAllocatableResources                  bool
	EnableInPlacePodVerticalScalingSchedulerPreemption bool
}

// ShouldDelegateResourceToDRA checks if the given resource should be delegated to the DRA plugin.
// It returns true if:
//  1. The resource is not a scalar resource in the node's allocatable (not provided by device plugin)
//  2. A device class mapping exists for the resource in the cache (when draManager is available)
func ShouldDelegateResourceToDRA(rName v1.ResourceName, nodeInfo fwk.NodeInfo, draManager fwk.SharedDRAManager, opts ResourceRequestsOptions) bool {
	if !opts.EnableDRAExtendedResource {
		return false
	}

	if nodeInfo != nil {
		if allocatable := nodeInfo.GetAllocatable().GetScalarResources()[rName]; allocatable > 0 {
			return false
		}
	}

	if draManager == nil {
		return false
	}

	// If draManager is available, check the cache for a mapping
	cache := draManager.DeviceClassResolver()
	return cache.GetDeviceClass(rName) != nil
}

// ComputePodResourceRequest returns a framework.Resource that covers the largest
// width in each resource dimension. Because init-containers run sequentially, we collect
// the max in each dimension iteratively. In contrast, we sum the resource vectors for
// regular containers since they run simultaneously.
//
// # The resources defined for Overhead should be added to the calculated Resource request sum
//
// Example:
//
// Pod:
//
//	InitContainers
//	  IC1:
//	    CPU: 2
//	    Memory: 1G
//	  IC2:
//	    CPU: 2
//	    Memory: 3G
//	Containers
//	  C1:
//	    CPU: 2
//	    Memory: 1G
//	  C2:
//	    CPU: 1
//	    Memory: 1G
//
// Result: CPU: 3, Memory: 3G
func ComputePodResourceRequest(pod *v1.Pod, opts ResourceRequestsOptions) framework.Resource {
	// pod hasn't scheduled yet so we don't need to worry about InPlacePodVerticalScalingEnabled
	reqs := resource.PodRequests(pod, resource.PodResourcesOptions{
		// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
		SkipPodLevelResources:                    !opts.EnablePodLevelResources,
		UseDRANodeAllocatableResourceClaimStatus: opts.EnableDRANodeAllocatableResources,
	})
	var result framework.Resource
	result.SetMaxResource(reqs)
	return result
}

// InsufficientResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type InsufficientResource struct {
	ResourceName v1.ResourceName
	// We explicitly have a parameter for reason to avoid formatting a message on the fly
	// for common resources, which is expensive for cluster autoscaler simulations.
	Reason    string
	Requested int64
	Used      int64
	Capacity  int64
	// Unresolvable indicates whether this node could be schedulable for the pod by the preemption,
	// which is determined by comparing the node's size and the pod's request.
	Unresolvable bool
}

// Fits checks if node have enough resources to host the pod.
func Fits(pod *v1.Pod, nodeInfo fwk.NodeInfo, draManager fwk.SharedDRAManager, opts ResourceRequestsOptions) []InsufficientResource {
	req := ComputePodResourceRequest(pod, opts)
	return FitsRequest(&req, nodeInfo, nil, nil, draManager, opts, pod)
}

// FitsRequest checks if the node has enough resources for an already computed pod
// request, skipping the ignored extended resources and the resources DRA provides.
func FitsRequest(podRequest *framework.Resource, nodeInfo fwk.NodeInfo, ignoredExtendedResources, ignoredResourceGroups sets.Set[string], draManager fwk.SharedDRAManager, opts ResourceRequestsOptions, pod *v1.Pod) []InsufficientResource {
	insufficientResources := make([]InsufficientResource, 0, 4)

	deltaPods := 1
	if isPodResizeOnNode(opts, nodeInfo, pod) {
		deltaPods = 0
	}
	allowedPodNumber := nodeInfo.GetAllocatable().GetAllowedPodNumber()
	if len(nodeInfo.GetPods())+deltaPods > allowedPodNumber {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourcePods,
			Reason:       "Too many pods",
			Requested:    1,
			Used:         int64(len(nodeInfo.GetPods())),
			Capacity:     int64(allowedPodNumber),
		})
	}

	if podRequest.MilliCPU == 0 &&
		podRequest.Memory == 0 &&
		podRequest.EphemeralStorage == 0 &&
		len(podRequest.ScalarResources) == 0 {
		return insufficientResources
	}

	deltaMilliCPU, deltaMemory, deltaEphemeralStorage, deltaScalarResources := adjustDeltasForPodResize(opts, podRequest, nodeInfo, pod)

	if podRequest.MilliCPU > 0 && deltaMilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-nodeInfo.GetRequested().GetMilliCPU()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceCPU,
			Reason:       "Insufficient cpu",
			Requested:    podRequest.MilliCPU,
			Used:         nodeInfo.GetRequested().GetMilliCPU(),
			Capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			Unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && deltaMemory > (nodeInfo.GetAllocatable().GetMemory()-nodeInfo.GetRequested().GetMemory()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceMemory,
			Reason:       "Insufficient memory",
			Requested:    podRequest.Memory,
			Used:         nodeInfo.GetRequested().GetMemory(),
			Capacity:     nodeInfo.GetAllocatable().GetMemory(),
			Unresolvable: podRequest.Memory > nodeInfo.GetAllocatable().GetMemory(),
		})
	}
	if podRequest.EphemeralStorage > 0 &&
		deltaEphemeralStorage > (nodeInfo.GetAllocatable().GetEphemeralStorage()-nodeInfo.GetRequested().GetEphemeralStorage()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceEphemeralStorage,
			Reason:       "Insufficient ephemeral-storage",
			Requested:    podRequest.EphemeralStorage,
			Used:         nodeInfo.GetRequested().GetEphemeralStorage(),
			Capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			Unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
	}

	for rName, rQuant := range deltaScalarResources {
		// Skip in case request quantity is zero
		if podRequest.ScalarResources[rName] == 0 {
			continue
		}

		if v1helper.IsExtendedResourceName(rName) {
			// If this resource is one of the extended resources that should be ignored, we will skip checking it.
			// rName is guaranteed to have a slash due to API validation.
			var rNamePrefix string
			if ignoredResourceGroups.Len() > 0 {
				rNamePrefix = strings.Split(string(rName), "/")[0]
			}
			if ignoredExtendedResources.Has(string(rName)) || ignoredResourceGroups.Has(rNamePrefix) {
				continue
			}
		}

		if ShouldDelegateResourceToDRA(rName, nodeInfo, draManager, opts) {
			continue
		}
		if podRequest.ScalarResources[rName] > 0 && rQuant > (nodeInfo.GetAllocatable().GetScalarResources()[rName]-nodeInfo.GetRequested().GetScalarResources()[rName]) {
			insufficientResources = append(insufficientResources, InsufficientResource{
				ResourceName: rName,
				Reason:       fmt.Sprintf("Insufficient %v", rName),
				Requested:    podRequest.ScalarResources[rName],
				Used:         nodeInfo.GetRequested().GetScalarResources()[rName],
				Capacity:     nodeInfo.GetAllocatable().GetScalarResources()[rName],
				Unresolvable: podRequest.ScalarResources[rName] > nodeInfo.GetAllocatable().GetScalarResources()[rName],
			})
		}
	}

	return insufficientResources
}

func isPodResizeOnNode(opts ResourceRequestsOptions, nodeInfo fwk.NodeInfo, pod *v1.Pod) bool {
	return opts.EnableInPlacePodVerticalScalingSchedulerPreemption && pod != nil && resource.IsPodResizeDeferred(pod) && pod.Spec.NodeName == nodeInfo.Node().Name
}

// adjustDeltasForPodResize calculates the resource deltas to evaluate for a pod.
// A deferred resize pod on its node is already counted in nodeInfo at max(desired, allocated, actual),
// so its delta is 0 and we only check whether the node is overcommitted.
func adjustDeltasForPodResize(opts ResourceRequestsOptions, podRequest *framework.Resource, nodeInfo fwk.NodeInfo, pod *v1.Pod) (int64, int64, int64, map[v1.ResourceName]int64) {
	if !isPodResizeOnNode(opts, nodeInfo, pod) {
		return podRequest.MilliCPU, podRequest.Memory, podRequest.EphemeralStorage, podRequest.ScalarResources
	}

	adjustedScalars := make(map[v1.ResourceName]int64, len(podRequest.ScalarResources))
	for rName := range podRequest.ScalarResources {
		adjustedScalars[rName] = 0
	}
	return 0, 0, 0, adjustedScalars
}
