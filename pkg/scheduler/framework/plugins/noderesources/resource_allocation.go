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
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	resourceapi "k8s.io/api/resource/v1"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources/extended"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// scorer is decorator for resourceAllocationScorer
type scorer func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer

// resourceAllocationScorer contains information to calculate resource allocation score.
type resourceAllocationScorer struct {
	Name                            string
	enableInPlacePodVerticalScaling bool
	enablePodLevelResources         bool
	enableDRAExtendedResource       bool
	// used to decide whether to use Requested or NonZeroRequested for
	// cpu and memory.
	useRequested bool
	scorer       func(requested, allocable []int64) int64
	resources    []config.ResourceSpec
	draFeatures  structured.Features
	draManager   fwk.SharedDRAManager
	celCache     *cel.Cache
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
		alloc, req := r.calculateResourceAllocatableRequest(ctx, nodeInfo, v1.ResourceName(r.resources[i].Name), podRequests[i])
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
func (r *resourceAllocationScorer) calculateResourceAllocatableRequest(ctx context.Context, nodeInfo fwk.NodeInfo, resource v1.ResourceName, podRequest int64) (int64, int64) {
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
		allocatable, exists := nodeInfo.GetAllocatable().GetScalarResources()[resource]
		if allocatable == 0 && r.enableDRAExtendedResource {
			// Allocatable 0 means that this resource is not handled by device plugin.
			// Calculate allocatable and requested for resources backed by DRA.
			allocatable, allocated := r.calculateDRAExtendedResourceAllocatableRequest(ctx, nodeInfo.Node(), resource)
			if allocatable > 0 {
				return allocatable, allocated + podRequest
			}
		}
		if exists {
			return allocatable, (nodeInfo.GetRequested().GetScalarResources()[resource] + podRequest)
		}
	}
	klog.FromContext(ctx).V(10).Info("Requested resource is omitted for node score calculation", "resourceName", resource)
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

// calculateDRAExtendedResourceAllocatableRequest calculates allocatable and allocated
// quantities for extended resources backed by DRA.
func (r *resourceAllocationScorer) calculateDRAExtendedResourceAllocatableRequest(ctx context.Context, node *v1.Node, resource v1.ResourceName) (int64, int64) {
	logger := klog.FromContext(ctx)
	// Get device class mapping to find the device class for this resource
	deviceClassMapping, err := extended.DeviceClassMapping(r.draManager)
	if err != nil {
		logger.Error(err, "Failed to get device class mapping for DRA extended resource scoring")
		return 0, 0
	}

	deviceClassName, exists := deviceClassMapping[resource]
	if !exists {
		logger.Error(nil, "Extended resource not found in device class mapping", "resource", resource)
		return 0, 0
	}

	deviceClass, err := r.draManager.DeviceClasses().Get(deviceClassName)
	if err != nil {
		logger.Error(err, "Failed to get device class for DRA extended resource scoring", "resource", resource, "deviceClass", deviceClassName)
		return 0, 0
	}

	capacity, allocated, err := r.calculateDRAResourceTotals(ctx, node, deviceClass)
	if err != nil {
		logger.Error(err, "Failed to calculate DRA resource capacity and allocated", "node", node.Name, "resource", resource, "deviceClass", deviceClassName)
		return 0, 0
	}

	logger.V(7).Info("DRA extended resource calculation", "node", node.Name, "resource", resource, "deviceClass", deviceClassName, "capacity", capacity, "allocated", allocated)
	return capacity, allocated
}

// calculateDRAResourceTotals computes the total capacity and total allocated count of devices
// matching the specified Device Class on the given node. It queries the DRA manager for resource
// slices and allocated devices, filters devices by class and driver, and returns the counts.
// Returns an error if resource information cannot be retrieved or if node matching fails.
//
// Parameters:
//
//	ctx         - context for cancellation and deadlines
//	node        - the node to evaluate device resources on
//	deviceClass - the device class to filter devices by
//
// Returns:
//
//	totalCapacity  - total number of devices matching the device class on the node
//	totalAllocated - number of devices currently allocated from the matching set
//	error          - any error encountered during processing
func (r *resourceAllocationScorer) calculateDRAResourceTotals(ctx context.Context, node *v1.Node, deviceClass *resourceapi.DeviceClass) (int64, int64, error) {
	allocatedState, err := r.draManager.ResourceClaims().GatherAllocatedState()
	if err != nil {
		return 0, 0, err
	}

	resourceSlices, err := r.draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return 0, 0, err
	}

	var totalCapacity, totalAllocated int64
	for _, slice := range resourceSlices {
		driver := slice.Spec.Driver
		pool := slice.Spec.Pool.Name
		var devices []resourceapi.Device
		// Handle per-device node selection vs slice-level node selection
		if ptr.Deref(slice.Spec.PerDeviceNodeSelection, false) {
			devices = []resourceapi.Device{}
			// When per-device node selection is enabled, check each device individually
			for _, device := range slice.Spec.Devices {
				// Check if this specific device matches the node
				deviceMatches, err := structured.NodeMatches(r.draFeatures, node, ptr.Deref(device.NodeName, ""), ptr.Deref(device.AllNodes, false), device.NodeSelector)
				if err != nil {
					return 0, 0, err
				}
				if deviceMatches {
					devices = append(devices, device)
				}
			}
		} else {
			// When per-device node selection is disabled, check slice-level node selection first
			matches, err := structured.NodeMatches(r.draFeatures, node, ptr.Deref(slice.Spec.NodeName, ""), ptr.Deref(slice.Spec.AllNodes, false), slice.Spec.NodeSelector)
			if err != nil {
				return 0, 0, err
			}
			if !matches {
				// Skip this slice as it doesn't match the node
				continue
			}
			devices = slice.Spec.Devices
		}
		// Count devices that match the device class
		for _, device := range devices {
			matches, err := r.deviceMatchesClass(ctx, device, deviceClass, driver)
			if err != nil {
				return 0, 0, err
			}
			if matches {
				totalCapacity++
				// Count allocated devices (both fully allocated and partially consumed)
				deviceID := structured.MakeDeviceID(driver, pool, device.Name)
				if structured.IsDeviceAllocated(deviceID, allocatedState) {
					totalAllocated++
				}
			}
		}
	}

	return totalCapacity, totalAllocated, nil
}

// deviceMatchesClass checks if a device matches the selectors of a device class.
// Note: This method assumes the device class has ExtendedResourceName set, as filtering
// should be done by the caller to ensure we only process DRA resources meant for extended
// resource scoring.
func (r *resourceAllocationScorer) deviceMatchesClass(ctx context.Context, device resourceapi.Device, deviceClass *resourceapi.DeviceClass, driver string) (bool, error) {
	// If no selectors are defined, all devices match
	if len(deviceClass.Spec.Selectors) == 0 {
		return true, nil
	}

	// All selectors must match for the device to be considered a match
	for _, selector := range deviceClass.Spec.Selectors {
		if selector.CEL == nil {
			continue
		}

		// Use cached CEL compilation for performance
		result := r.celCache.GetOrCompile(selector.CEL.Expression)
		if result.Error != nil {
			return false, result.Error
		}

		// Evaluate the expression against the device
		celDevice := cel.Device{
			Driver:     driver,
			Attributes: device.Attributes,
			Capacity:   device.Capacity,
		}

		matches, _, err := result.DeviceMatches(ctx, celDevice)
		if err != nil || !matches {
			return false, nil
		}
	}

	return true, nil
}
