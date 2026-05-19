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
	"strings"
	"sync"

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
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// scorer is decorator for resourceAllocationScorer
type scorer func(args *config.NodeResourcesFitArgs) *resourceAllocationScorer

// DRACaches holds various caches used for DRA-related computations
type DRACaches struct {
	// celCache is a cache for compiled CEL expressions used in device class selectors.
	celCache *cel.Cache
	// Cache for DeviceMatches results to avoid expensive repeated evaluations
	deviceMatchCache sync.Map // map[deviceMatchCacheKey]bool
	// Cache for NodeMatches results to avoid expensive repeated node selector evaluations
	nodeMatchCache sync.Map // map[nodeMatchCacheKey]bool
}

// resourceAllocationScorer contains information to calculate resource allocation score.
type resourceAllocationScorer struct {
	Name                                          string
	enableInPlacePodVerticalScaling               bool
	enablePodLevelResources                       bool
	enableDRAExtendedResource                     bool
	enableInPlacePodLevelResourcesVerticalScaling bool
	// used to decide whether to use Requested or NonZeroRequested for
	// cpu and memory.
	useRequested bool
	scorer       func(requested, allocable []int64) int64
	resources    []config.ResourceSpec
	draFeatures  structured.Features
	draManager   fwk.SharedDRAManager
	// Caches for DRA-related computations
	DRACaches
}

// buildNodeMatchCacheKey creates a string cache key for node matching results
// Using a string key is significantly faster than struct keys with sync.Map
func buildNodeMatchCacheKey(nodeName string, nodeNameToMatch string, allNodesMatch bool, nodeSelectorHash string) string {
	// Pre-allocate sufficient capacity to avoid reallocation
	var b strings.Builder
	b.Grow(len(nodeName) + len(nodeNameToMatch) + len(nodeSelectorHash) + 4)

	b.WriteString(nodeName)
	b.WriteByte('|')
	b.WriteString(nodeNameToMatch)
	b.WriteByte('|')
	if allNodesMatch {
		b.WriteByte('1')
	} else {
		b.WriteByte('0')
	}
	b.WriteByte('|')
	b.WriteString(nodeSelectorHash)

	return b.String()
}

// buildDeviceMatchCacheKey creates a string cache key for device matching results
// Using a string key is significantly faster than struct keys with sync.Map
// This concatenates expression|driver|poolName|deviceName with pipe separators
func buildDeviceMatchCacheKey(expression string, driver string, poolName string, deviceName string) string {
	// Pre-allocate sufficient capacity to avoid reallocation
	var b strings.Builder
	b.Grow(len(expression) + len(driver) + len(poolName) + len(deviceName) + 3)

	b.WriteString(expression)
	b.WriteByte('|')
	b.WriteString(driver)
	b.WriteByte('|')
	b.WriteString(poolName)
	b.WriteByte('|')
	b.WriteString(deviceName)

	return b.String()
}

// nodeMatches is a cached wrapper around structured.NodeMatches
func (r *resourceAllocationScorer) nodeMatches(node *v1.Node, nodeNameToMatch string, allNodesMatch bool, nodeSelector *v1.NodeSelector) (bool, error) {

	var nodeName string
	if node != nil {
		nodeName = node.Name
	}

	nodeSelectorStr := nodeSelector.String()
	key := buildNodeMatchCacheKey(nodeName, nodeNameToMatch, allNodesMatch, nodeSelectorStr)

	// Check cache first
	if matches, ok := r.nodeMatchCache.Load(key); ok {
		return matches.(bool), nil
	}

	// Call the original function
	matches, err := structured.NodeMatches(r.draFeatures, node, nodeNameToMatch, allNodesMatch, nodeSelector)

	// Cache the result (even if there was an error, to avoid repeated failures)
	if err == nil {
		r.nodeMatchCache.Store(key, matches)
	}

	return matches, err
}

// score will use `scorer` function to calculate the score.
func (r *resourceAllocationScorer) score(
	ctx context.Context,
	pod *v1.Pod,
	nodeInfo fwk.NodeInfo,
	podRequests []int64,
	draPreScoreState *draPreScoreState,
) (int64, *fwk.Status) {
	logger := klog.FromContext(ctx)
	node := nodeInfo.Node()

	// resources not set, nothing scheduled,
	if len(r.resources) == 0 {
		return 0, fwk.NewStatus(fwk.Error, "resources not found")
	}

	requested := make([]int64, len(r.resources))
	allocatable := make([]int64, len(r.resources))
	for i := range r.resources {
		alloc, req := r.calculateResourceAllocatableRequest(ctx, nodeInfo, v1.ResourceName(r.resources[i].Name), podRequests[i], draPreScoreState)
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
func (r *resourceAllocationScorer) calculateResourceAllocatableRequest(
	ctx context.Context,
	nodeInfo fwk.NodeInfo,
	resource v1.ResourceName,
	podRequest int64,
	draPreScoreState *draPreScoreState,
) (int64, int64) {
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
		if allocatable == 0 && r.enableDRAExtendedResource && draPreScoreState != nil {
			// Allocatable 0 means that this resource is not handled by device plugin.
			// Calculate allocatable and requested for resources backed by DRA.
			allocatable, allocated := r.calculateDRAExtendedResourceAllocatableRequest(ctx, nodeInfo.Node(), resource, draPreScoreState)
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
		InPlacePodLevelResourcesVerticalScalingEnabled: r.enableInPlacePodLevelResourcesVerticalScaling,
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

// getDRAPreScoredParams returns the DRA allocated state and resource slices for DRA extended resource scoring.
func getDRAPreScoredParams(draManager fwk.SharedDRAManager, resources []config.ResourceSpec) (*draPreScoreState, *fwk.Status) {
	anyBackedByDRA := false
	for _, resource := range resources {
		resourceName := v1.ResourceName(resource.Name)
		if !schedutil.IsDRAExtendedResourceName(resourceName) {
			continue
		}
		deviceClass := draManager.DeviceClassResolver().GetDeviceClass(resourceName)
		if deviceClass != nil {
			anyBackedByDRA = true
			break
		}
	}
	// There's no point in returning DRA data as there are no resources backed by DRA.
	if !anyBackedByDRA {
		return nil, nil
	}

	allocatedState, err := draManager.ResourceClaims().GatherAllocatedState()
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	resourceSlices, err := draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return nil, fwk.AsStatus(err)
	}

	return &draPreScoreState{
		allocatedState: allocatedState,
		resourceSlices: resourceSlices,
	}, nil
}

// calculateDRAExtendedResourceAllocatableRequest calculates allocatable and allocated
// quantities for extended resources backed by DRA.
func (r *resourceAllocationScorer) calculateDRAExtendedResourceAllocatableRequest(
	ctx context.Context,
	node *v1.Node,
	resource v1.ResourceName,
	draPreScoreState *draPreScoreState,
) (int64, int64) {
	logger := klog.FromContext(ctx)
	deviceClass := r.draManager.DeviceClassResolver().GetDeviceClass(resource)
	if deviceClass == nil {
		// This resource is not backed by DRA.
		logger.V(7).Info("Extended resource not found in device class mapping", "resource", resource)
		return 0, 0
	}

	capacity, allocated, err := r.calculateDRAResourceTotals(ctx, node, deviceClass, draPreScoreState.allocatedState, draPreScoreState.resourceSlices)
	if err != nil {
		logger.Error(err, "Failed to calculate DRA resource capacity and allocated", "node", node.Name, "resource", resource, "deviceClass", deviceClass.Name)
		return 0, 0
	}

	logger.V(7).Info("DRA extended resource calculation", "node", node.Name, "resource", resource, "deviceClass", deviceClass.Name, "capacity", capacity, "allocated", allocated)
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
func (r *resourceAllocationScorer) calculateDRAResourceTotals(ctx context.Context, node *v1.Node, deviceClass *resourceapi.DeviceClass, allocatedState *structured.AllocatedState, resourceSlices []*resourceapi.ResourceSlice,
) (int64, int64, error) {
	var totalCapacity, totalAllocated int64
	nodeName := node.Name

	for _, slice := range resourceSlices {
		// Early filtering: check if slice applies to this node
		perDeviceNodeSelection := ptr.Deref(slice.Spec.PerDeviceNodeSelection, false)

		var devices []resourceapi.Device

		if perDeviceNodeSelection {
			// Per-device node selection: filter devices individually
			devices = make([]resourceapi.Device, 0, len(slice.Spec.Devices))
			for _, device := range slice.Spec.Devices {
				deviceNodeName := ptr.Deref(device.NodeName, "")
				deviceAllNodes := ptr.Deref(device.AllNodes, false)

				// Fast path: check AllNodes or exact name match first
				if deviceAllNodes || (deviceNodeName != "" && deviceNodeName == nodeName) {
					devices = append(devices, device)
					continue
				}

				// Slow path: only if we have a node selector
				if device.NodeSelector != nil {
					deviceMatches, err := r.nodeMatches(node, deviceNodeName, deviceAllNodes, device.NodeSelector)
					if err != nil {
						return 0, 0, err
					}
					if deviceMatches {
						devices = append(devices, device)
					}
				}
			}
		} else {
			// Slice-level node selection
			sliceNodeName := ptr.Deref(slice.Spec.NodeName, "")
			sliceAllNodes := ptr.Deref(slice.Spec.AllNodes, false)

			// Fast path: check AllNodes or exact name match first
			if !sliceAllNodes && sliceNodeName != nodeName && slice.Spec.NodeSelector != nil {
				// Need to check node selector
				matches, err := r.nodeMatches(node, sliceNodeName, sliceAllNodes, slice.Spec.NodeSelector)
				if err != nil {
					return 0, 0, err
				}
				if !matches {
					continue // Skip this slice
				}
			} else if !sliceAllNodes && sliceNodeName != "" && sliceNodeName != nodeName {
				// Node name specified but doesn't match
				continue
			}

			devices = slice.Spec.Devices
		}

		// Fast path for device class with no selectors
		if len(deviceClass.Spec.Selectors) == 0 {
			driver := slice.Spec.Driver
			pool := slice.Spec.Pool.Name
			for _, device := range devices {
				totalCapacity++
				deviceID := structured.MakeDeviceID(driver, pool, device.Name)
				if structured.IsDeviceAllocated(deviceID, allocatedState) {
					totalAllocated++
				}
			}
		} else {
			// Slow path: check device class selectors
			driver := slice.Spec.Driver
			pool := slice.Spec.Pool.Name
			for _, device := range devices {
				matches, err := r.deviceMatchesClass(ctx, device, deviceClass, driver, pool)
				if err != nil {
					return 0, 0, err
				}
				if matches {
					totalCapacity++
					deviceID := structured.MakeDeviceID(driver, pool, device.Name)
					if structured.IsDeviceAllocated(deviceID, allocatedState) {
						totalAllocated++
					}
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
func (r *resourceAllocationScorer) deviceMatchesClass(ctx context.Context, device resourceapi.Device, deviceClass *resourceapi.DeviceClass, driver string, poolName string) (bool, error) {
	// If no selectors are defined, all devices match
	if len(deviceClass.Spec.Selectors) == 0 {
		return true, nil
	}

	// Lazily create the CEL device only when needed (first CEL selector that's not cached)
	var celDevice cel.Device
	celDeviceCreated := false

	// All selectors must match for the device to be considered a match
	for _, selector := range deviceClass.Spec.Selectors {
		if selector.CEL == nil {
			continue
		}

		key := buildDeviceMatchCacheKey(selector.CEL.Expression, driver, poolName, device.Name)

		// Check if result is already cached
		if matches, ok := r.deviceMatchCache.Load(key); ok {
			if !matches.(bool) {
				return false, nil
			}
			continue // This selector matches, check the next one
		}

		// Cache miss - need to evaluate CEL expression
		// Create CEL device if we haven't already
		if !celDeviceCreated {
			celDevice = cel.Device{
				Driver:     driver,
				Attributes: device.Attributes,
				Capacity:   device.Capacity,
			}
			celDeviceCreated = true
		}

		// Use cached CEL compilation for performance
		result := r.celCache.GetOrCompile(selector.CEL.Expression)
		if result.Error != nil {
			return false, result.Error
		}

		matches, _, err := result.DeviceMatches(ctx, celDevice)
		if err != nil || !matches {
			return false, err
		}

		// Cache the result for future use
		r.deviceMatchCache.Store(key, matches)
	}

	return true, nil
}
