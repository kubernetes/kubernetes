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
	// pod hasn't scheduled yet so we don't need to worry about UseStatusResources
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

	// A deferred resize pod on its node is already counted in nodeInfo at max(desired, allocated, actual),
	// so it adds nothing and we only check whether the node is overcommitted. This relies on
	// refreshResizingPodFromCache having already confirmed that the pod is present on this node in the snapshot.
	alreadyCounted := isPodResizeOnNode(opts, nodeInfo, pod)
	added := podRequest
	if alreadyCounted {
		added = &framework.Resource{}
	}

	allowedPodNumber := nodeInfo.GetAllocatable().GetAllowedPodNumber()
	if !alreadyCounted && len(nodeInfo.GetPods()) >= allowedPodNumber {
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

	requested := nodeInfo.GetRequested()
	if alreadyCounted {
		requested = effectiveRequestedForResize(nodeInfo, pod)
	}

	if podRequest.MilliCPU > 0 && added.MilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-requested.GetMilliCPU()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceCPU,
			Reason:       "Insufficient cpu",
			Requested:    podRequest.MilliCPU,
			Used:         requested.GetMilliCPU(),
			Capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			Unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && added.Memory > (nodeInfo.GetAllocatable().GetMemory()-requested.GetMemory()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceMemory,
			Reason:       "Insufficient memory",
			Requested:    podRequest.Memory,
			Used:         requested.GetMemory(),
			Capacity:     nodeInfo.GetAllocatable().GetMemory(),
			Unresolvable: podRequest.Memory > nodeInfo.GetAllocatable().GetMemory(),
		})
	}
	if podRequest.EphemeralStorage > 0 &&
		added.EphemeralStorage > (nodeInfo.GetAllocatable().GetEphemeralStorage()-requested.GetEphemeralStorage()) {
		insufficientResources = append(insufficientResources, InsufficientResource{
			ResourceName: v1.ResourceEphemeralStorage,
			Reason:       "Insufficient ephemeral-storage",
			Requested:    podRequest.EphemeralStorage,
			Used:         requested.GetEphemeralStorage(),
			Capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			Unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
	}

	for rName, rQuant := range podRequest.ScalarResources {
		// Skip in case request quantity is zero
		if rQuant == 0 {
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
		if added.ScalarResources[rName] > (nodeInfo.GetAllocatable().GetScalarResources()[rName] - requested.GetScalarResources()[rName]) {
			insufficientResources = append(insufficientResources, InsufficientResource{
				ResourceName: rName,
				Reason:       fmt.Sprintf("Insufficient %v", rName),
				Requested:    rQuant,
				Used:         requested.GetScalarResources()[rName],
				Capacity:     nodeInfo.GetAllocatable().GetScalarResources()[rName],
				Unresolvable: rQuant > nodeInfo.GetAllocatable().GetScalarResources()[rName],
			})
		}
	}

	return insufficientResources
}

func isPodResizeOnNode(opts ResourceRequestsOptions, nodeInfo fwk.NodeInfo, pod *v1.Pod) bool {
	return opts.EnableInPlacePodVerticalScalingSchedulerPreemption && pod != nil && len(pod.Spec.NodeName) > 0 && nodeInfo.Node() != nil && pod.Spec.NodeName == nodeInfo.Node().Name && resource.IsPodResizeDeferred(pod)
}

// effectiveRequestedForResize returns nodeInfo.GetRequested() with other deferred resize pods on
// the node counted at max(allocated, actual) rather than max(desired, allocated, actual), matching
// Kubelet's resize admission calculation. The evaluated pod itself stays at max(desired, allocated, actual).
func effectiveRequestedForResize(nodeInfo fwk.NodeInfo, pod *v1.Pod) fwk.Resource {
	var adjusted *framework.Resource
	for _, pi := range nodeInfo.GetPods() {
		other := pi.GetPod()
		if other.UID == pod.UID || !resource.IsPodResizeDeferred(other) {
			continue
		}
		if adjusted == nil {
			adjusted = nodeInfo.GetRequested().(*framework.Resource).Clone()
		}
		withDesired := pi.(*framework.PodInfo).CalculateResource().Resource
		withoutDesired := podInfoWithAllocatedResources(pi).CalculateResource().Resource
		adjusted.MilliCPU -= withDesired.GetMilliCPU() - withoutDesired.GetMilliCPU()
		adjusted.Memory -= withDesired.GetMemory() - withoutDesired.GetMemory()
		adjusted.EphemeralStorage -= withDesired.GetEphemeralStorage() - withoutDesired.GetEphemeralStorage()
		for rName, rQuant := range withDesired.GetScalarResources() {
			adjusted.ScalarResources[rName] -= rQuant - withoutDesired.GetScalarResources()[rName]
		}
	}
	if adjusted == nil {
		return nodeInfo.GetRequested()
	}
	return adjusted
}

// podInfoWithAllocatedResources returns a PodInfo whose spec requests are replaced with the
// allocated resources from status, so CalculateResource() evaluates max(allocated, actual)
// instead of max(desired, allocated, actual). CalculateResource takes the desired term from
// spec: container requests, or pod-level requests when set. Pod-level status only replaces the
// allocated and actual terms, so both spec levels are replaced.
func podInfoWithAllocatedResources(pi fwk.PodInfo) *framework.PodInfo {
	pod := pi.GetPod().DeepCopy()

	// Status.AllocatedResources and Status.Resources.Requests are kubelet pod-level aggregates
	// that already include pod.Spec.Overhead and DRA node-allocatable quantities. Subtract
	// overhead before copying into pod-level spec requests and on Status.Resources.Requests, and
	// clear Status.AllocatedResources so PodRequests does not pick un-adjusted status aggregates
	// before adding pod.Spec.Overhead back.
	if pod.Spec.Resources != nil {
		if pod.Status.AllocatedResources != nil {
			for rName := range pod.Spec.Resources.Requests {
				if qty, ok := pod.Status.AllocatedResources[rName]; ok {
					if overhead, hasOverhead := pod.Spec.Overhead[rName]; hasOverhead {
						qty.Sub(overhead)
						if qty.Sign() < 0 {
							qty.Set(0)
						}
					}
					pod.Spec.Resources.Requests[rName] = qty
				} else {
					delete(pod.Spec.Resources.Requests, rName)
				}
			}
		}
		if pod.Status.Resources != nil {
			for rName, qty := range pod.Status.Resources.Requests {
				if overhead, hasOverhead := pod.Spec.Overhead[rName]; hasOverhead {
					qty.Sub(overhead)
					if qty.Sign() < 0 {
						qty.Set(0)
					}
					pod.Status.Resources.Requests[rName] = qty
				}
			}
		}
	}
	pod.Status.AllocatedResources = nil

	allocByContainer := make(map[string]v1.ResourceList, len(pod.Status.ContainerStatuses)+len(pod.Status.InitContainerStatuses))
	for _, cs := range pod.Status.InitContainerStatuses {
		if cs.AllocatedResources != nil || cs.State.Running != nil {
			allocByContainer[cs.Name] = cs.AllocatedResources
		}
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.AllocatedResources != nil || cs.State.Running != nil {
			allocByContainer[cs.Name] = cs.AllocatedResources
		}
	}
	for i := range pod.Spec.InitContainers {
		if alloc, ok := allocByContainer[pod.Spec.InitContainers[i].Name]; ok {
			pod.Spec.InitContainers[i].Resources.Requests = alloc
		}
	}
	for i := range pod.Spec.Containers {
		if alloc, ok := allocByContainer[pod.Spec.Containers[i].Name]; ok {
			pod.Spec.Containers[i].Resources.Requests = alloc
		}
	}

	return &framework.PodInfo{Pod: pod}
}
