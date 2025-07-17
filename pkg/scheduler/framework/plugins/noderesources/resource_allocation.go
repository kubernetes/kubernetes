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

package noderesources

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"

	resourcehelper "k8s.io/component-helpers/resource"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// scorer is decorator for resourceAllocationScorer
type scorer func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer

// resourceAllocationScorer contains information to calculate resource allocation score.
type resourceAllocationScorer struct {
	Name                            string
	enableInPlacePodVerticalScaling bool
	enablePodLevelResources         bool
	// used to decide whether to use Requested or NonZeroRequested for
	// cpu and memory.
	useRequested bool
	scorer       func(requested, allocable []int64) int64
	resources    []config.ResourceSpec
}

// score will use `scorer` function to calculate the score.
func (r *resourceAllocationScorer) score(
	ctx context.Context,
	pod *v1.Pod,
	nodeInfo fwk.NodeInfo,
	podRequests []int64) (int64, *fwk.Status) {
	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()

	// resources not set, nothing scheduled,
	if len(r.resources) == 0 {
		return 0, fwk.NewStatus(fwk.Error, "resources not found")
	}

	requested := make([]int64, len(r.resources))
	allocatable := make([]int64, len(r.resources))
	for i := range r.resources {
		alloc, req := r.calculateResourceAllocatableRequest(logger, nodeInfo, v1.ResourceName(r.resources[i].Name), podRequests[i])
		// Only fill the extended resource entry when it's non-zero.
		if alloc == 0 {
			continue
		}
		allocatable[i] = alloc
		requested[i] = req
	}

	score := r.scorer(requested, allocatable)

	if loggerV := logger.V(10); loggerV.Enabled() { // Serializing these maps is costly.
		loggerV.Info("Listed internal info for allocatable resources, requested resources and score", "pod",
			klog.KObj(pod), "node", klog.KObj(node), "resourceAllocationScorer", r.Name,
			"allocatableResource", allocatable, "requestedResource", requested, "resourceScore", score,
		)
	}

	return score, nil
}

// calculateResourceAllocatableRequest returns 2 parameters:
// - 1st param: quantity of allocatable resource on the node.
// - 2nd param: aggregated quantity of requested resource on the node.
// Note: if it's an extended resource, and the pod doesn't request it, (0, 0) is returned.
func (r *resourceAllocationScorer) calculateResourceAllocatableRequest(logger klog.Logger, nodeInfo fwk.NodeInfo, resource v1.ResourceName, podRequest int64) (int64, int64) {
	requested := nodeInfo.GetNonZeroRequested()
	if r.useRequested {
		requested = nodeInfo.GetRequested()
	}

	// If it's an extended resource, and the pod doesn't request it. We return (0, 0)
	// as an implication to bypass scoring on this resource.
	if podRequest == 0 && schedutil.IsScalarResourceName(resource) {
		return 0, 0
	}
	switch resource {
	case v1.ResourceCPU:
		return nodeInfo.GetAllocatable().GetMilliCPU(), (requested.GetMilliCPU() + podRequest)
	case v1.ResourceMemory:
		return nodeInfo.GetAllocatable().GetMemory(), (requested.GetMemory() + podRequest)
	case v1.ResourceEphemeralStorage:
		return nodeInfo.GetAllocatable().GetEphemeralStorage(), (nodeInfo.GetRequested().GetEphemeralStorage() + podRequest)
	default:
		if _, exists := nodeInfo.GetAllocatable().GetScalarResources()[resource]; exists {
			return nodeInfo.GetAllocatable().GetScalarResources()[resource], (nodeInfo.GetRequested().GetScalarResources()[resource] + podRequest)
		}
	}
	logger.V(10).Info("Requested resource is omitted for node score calculation", "resourceName", resource)
	return 0, 0
}

// calculatePodResourceRequest returns the total non-zero requests. If Overhead is defined for the pod
// the Overhead is added to the result.
func (r *resourceAllocationScorer) calculatePodResourceRequest(pod *v1.Pod, resourceName v1.ResourceName) int64 {

	opts := resourcehelper.PodResourcesOptions{
		UseStatusResources: r.enableInPlacePodVerticalScaling,
		// SkipPodLevelResources is set to false when PodLevelResources feature is enabled.
		SkipPodLevelResources: !r.enablePodLevelResources,
	}

	if !r.useRequested {
		opts.NonMissingContainerRequests = v1.ResourceList{
			v1.ResourceCPU:    *resource.NewMilliQuantity(schedutil.DefaultMilliCPURequest, resource.DecimalSI),
			v1.ResourceMemory: *resource.NewQuantity(schedutil.DefaultMemoryRequest, resource.DecimalSI),
		}
	}

	requests := resourcehelper.PodRequests(pod, opts)

	quantity := requests[resourceName]
	if resourceName == v1.ResourceCPU {
		return quantity.MilliValue()
	}
	return quantity.Value()
}

func (r *resourceAllocationScorer) calculatePodResourceRequestList(pod *v1.Pod, resources []config.ResourceSpec) []int64 {
	podRequests := make([]int64, len(resources))
	for i := range resources {
		podRequests[i] = r.calculatePodResourceRequest(pod, v1.ResourceName(resources[i].Name))
	}
	return podRequests
}

func (r *resourceAllocationScorer) isBestEffortPod(podRequests []int64) bool {
	for _, request := range podRequests {
		if request != 0 {
			return false
		}
	}
	return true
}
