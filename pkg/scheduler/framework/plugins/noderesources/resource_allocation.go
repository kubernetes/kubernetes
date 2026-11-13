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
	"container/list"
	"context"
	"fmt"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	resourceinformers "k8s.io/client-go/informers/resource/v1"
	"k8s.io/client-go/tools/cache"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

	resourceapi "k8s.io/api/resource/v1"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/dynamic-resource-allocation/structured"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// scorer is decorator for resourceAllocationScorer
type scorer func(strategy *config.ScoringStrategy) *resourceAllocationScorer

// DRACaches belongs to one scoring plugin instance and is shared by scheduling
// cycles and parallel Score workers. It must not be copied after first use.
type DRACaches struct {
	// celCache is a cache for compiled CEL expressions used in device class selectors.
	celCache *cel.Cache
	// Cache for DeviceMatches results to avoid expensive repeated evaluations.
	// Both this cache and sliceMatchResults.results use sync.Map because parallel
	// Score workers share them, and reads are typically more common than writes.
	deviceMatchCache   deviceMatchCache
	resourceSliceStore cache.Store
	gcDone             chan struct{}
}

// sliceMatchResults belongs to one immutable ResourceSlice spec. In-flight readers
// retain their entry even if a newer ResourceSlice generation replaces it in the outer cache.
type sliceMatchResults struct {
	generation int64
	name       string
	element    *list.Element // Protected by deviceMatchCache.mu.
	results    sync.Map      // map[deviceMatchKey]bool
}

func (c *DRACaches) matchResultsForSlice(slice *draapi.ResourceSlice) *sliceMatchResults {
	if slice.UID == "" {
		return nil
	}
	if c.resourceSliceStore != nil {
		obj, exists, err := c.resourceSliceStore.GetByKey(slice.Name)
		current, _ := obj.(*resourceapi.ResourceSlice)
		if err != nil || !exists || current == nil || current.UID != slice.UID || current.Generation != slice.Generation {
			if value, ok := c.deviceMatchCache.Load(slice.UID); ok {
				entry := value.(*sliceMatchResults)
				// An old snapshot must not invalidate a newer, currently eligible entry.
				if err != nil || !exists || current == nil || current.UID != slice.UID || current.Generation != entry.generation {
					c.deviceMatchCache.CompareAndDelete(slice.UID, entry)
				}
			}
			return &sliceMatchResults{generation: slice.Generation, name: slice.Name}
		}
	}
	match, published := c.loadMatchResultsForSlice(slice)
	if published && c.resourceSliceStore != nil {
		// A deletion may occur between the eligibility check and publication.
		obj, exists, err := c.resourceSliceStore.GetByKey(slice.Name)
		current, _ := obj.(*resourceapi.ResourceSlice)
		if err != nil || !exists || current == nil || current.UID != slice.UID || current.Generation != slice.Generation {
			c.deviceMatchCache.CompareAndDelete(slice.UID, match)
		}
	}
	return match
}

func (c *DRACaches) loadMatchResultsForSlice(slice *draapi.ResourceSlice) (*sliceMatchResults, bool) {
	// Test slices may lack UIDs and must not share a cache entry.
	if slice.UID == "" {
		return nil, false
	}
	var fresh *sliceMatchResults
	for {
		value, ok := c.deviceMatchCache.Load(slice.UID)
		if ok && value.(*sliceMatchResults).generation == slice.Generation {
			return value.(*sliceMatchResults), false
		}
		if fresh == nil {
			fresh = &sliceMatchResults{generation: slice.Generation, name: slice.Name}
		}
		if !ok {
			if _, loaded := c.deviceMatchCache.LoadOrStore(slice.UID, fresh); !loaded {
				return fresh, true
			}
		} else {
			// An older snapshot may still be scored, but must not replace newer results.
			if value.(*sliceMatchResults).generation > slice.Generation {
				return fresh, false
			}
			if c.deviceMatchCache.CompareAndSwap(slice.UID, value, fresh) {
				return fresh, true
			}
		}
	}
}

// invalidateDeviceMatchCache preserves results published for a newer generation
// while an older informer notification was still queued.
func (c *DRACaches) invalidateDeviceMatchCache(slice *resourceapi.ResourceSlice) {
	if value, ok := c.deviceMatchCache.Load(slice.UID); ok && value.(*sliceMatchResults).generation <= slice.Generation {
		c.deviceMatchCache.CompareAndDelete(slice.UID, value)
	}
}

// Objectless deletions provide only a name, so only this exceptional path scans
// the cache. Check current state after capturing each candidate to retain entries
// published for a same-name replacement while the notification was pending.
func (c *DRACaches) invalidateDeviceMatchCacheByName(name string) {
	c.deviceMatchCache.values.Range(func(key, value any) bool {
		entry := value.(*sliceMatchResults)
		if entry.name != name {
			return true
		}
		obj, exists, err := c.resourceSliceStore.GetByKey(name)
		current, _ := obj.(*resourceapi.ResourceSlice)
		if err == nil && (!exists || current == nil || current.UID != key || current.Generation != entry.generation) {
			c.deviceMatchCache.CompareAndDelete(key, entry)
		}
		return true
	})
}

func (c *DRACaches) watchResourceSlices(ctx context.Context, informer cache.SharedIndexInformer) error {
	c.resourceSliceStore = informer.GetStore()
	c.gcDone = make(chan struct{})
	registration, err := resourceinformers.ToResourceSliceIndexInformer(informer).AddTypedEventHandler(resourceinformers.ResourceSliceHandlerFuncs{
		UpdateFunc: func(oldSlice, newSlice *resourceapi.ResourceSlice) {
			if oldSlice.UID != newSlice.UID || oldSlice.Generation != newSlice.Generation {
				c.invalidateDeviceMatchCache(oldSlice)
			}
		},
		DeleteFunc: func(deleted resourceinformers.DeletedResourceSlice) {
			if deleted.OptionalObj != nil {
				c.invalidateDeviceMatchCache(deleted.OptionalObj)
			} else {
				c.invalidateDeviceMatchCacheByName(deleted.GetName())
			}
		},
	})
	if err != nil {
		return fmt.Errorf("register ResourceSlice match cache invalidation handler: %w", err)
	}
	go c.runDeviceMatchGC(ctx, func() { _ = informer.RemoveEventHandler(registration) })
	return nil
}

// resourceAllocationScorer contains information to calculate resource allocation score.
type resourceAllocationScorer struct {
	Name                                          string
	enablePodLevelResources                       bool
	enableDRAExtendedResource                     bool
	enableInPlacePodLevelResourcesVerticalScaling bool
	// used to decide whether to use Requested or NonZeroRequested for
	// cpu and memory.
	useRequested bool
	scorer       func(requested, allocated, allocatable []int64) int64
	resources    []config.ResourceSpec
	draFeatures  structured.Features
	draManager   fwk.SharedDRAManager
	DRACaches
}

// Driver and pool are fixed within the ResourceSlice owning these results.
type deviceMatchKey struct {
	expression string
	device     draapi.UniqueString
}

// nodeMatches is not cached: node labels change without any event that could
// invalidate cached selector results (#144106), and evaluating a selector is
// cheap compared to CEL device matching.
func (r *resourceAllocationScorer) nodeMatches(node *v1.Node, nodeNameToMatch string, allNodesMatch bool, nodeSelector *v1.NodeSelector) (bool, error) {
	return structured.NodeMatches(r.draFeatures, node, nodeNameToMatch, allNodesMatch, nodeSelector)
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

	requested, allocated, allocatable := r.calculateNodeAllocatableRequest(ctx, nodeInfo, podRequests, draPreScoreState)

	score := r.scorer(requested, allocated, allocatable)

	if loggerV := logger.V(10); loggerV.Enabled() { // Serializing these maps is costly.
		loggerV.Info("Listed internal info for allocatable resources, requested resources and score", "pod",
			klog.KObj(pod), "node", klog.KObj(node), "resourceAllocationScorer", r.Name,
			"allocatableResource", allocatable, "requestedResource", requested, "resourceScore", score,
		)
	}

	return score, nil
}

func (r *resourceAllocationScorer) calculateNodeAllocatableRequest(
	ctx context.Context,
	nodeInfo fwk.NodeInfo,
	podRequests []int64,
	draPreScoreState *draPreScoreState,
) (requested []int64, allocated []int64, allocatable []int64) {
	requested = make([]int64, len(r.resources))
	allocated = make([]int64, len(r.resources))
	allocatable = make([]int64, len(r.resources))
	for i := range r.resources {
		resource := v1.ResourceName(r.resources[i].Name)
		// If it's an extended resource, and the pod doesn't request it.
		// We don't fill the resource entry as an implication to bypass scoring on it.
		if podRequests[i] == 0 && schedutil.IsScalarResourceName(resource) {
			continue
		}
		nodeAllocatable, nodeAllocated := r.calculateResourceAllocatableRequest(ctx, nodeInfo, resource, draPreScoreState)
		// Only fill the extended resource entry when it's non-zero.
		if nodeAllocatable == 0 {
			continue
		}
		allocatable[i] = nodeAllocatable
		allocated[i] = nodeAllocated
		requested[i] = framework.SaturatingAdd(allocated[i], podRequests[i])
	}
	return requested, allocated, allocatable
}

// calculateResourceAllocatableRequest returns 2 parameters:
// - 1st param: quantity of allocatable resource on the node.
// - 2nd param: aggregated quantity of requested resource on the node.
func (r *resourceAllocationScorer) calculateResourceAllocatableRequest(
	ctx context.Context,
	nodeInfo fwk.NodeInfo,
	resource v1.ResourceName,
	draPreScoreState *draPreScoreState,
) (allocatable int64, allocated int64) {
	requested := nodeInfo.GetNonZeroRequested()
	if r.useRequested {
		requested = nodeInfo.GetRequested()
	}

	switch resource {
	case v1.ResourceCPU:
		return nodeInfo.GetAllocatable().GetMilliCPU(), requested.GetMilliCPU()
	case v1.ResourceMemory:
		return nodeInfo.GetAllocatable().GetMemory(), requested.GetMemory()
	case v1.ResourceEphemeralStorage:
		return nodeInfo.GetAllocatable().GetEphemeralStorage(), nodeInfo.GetRequested().GetEphemeralStorage()
	default:
		allocatable, exists := nodeInfo.GetAllocatable().GetScalarResources()[resource]
		if allocatable == 0 && r.enableDRAExtendedResource && draPreScoreState != nil {
			// Allocatable 0 means that this resource is not handled by device plugin.
			// Calculate allocatable and requested for resources backed by DRA.
			allocatable, allocated := r.calculateDRAExtendedResourceAllocatableRequest(ctx, nodeInfo.Node(), resource, draPreScoreState)
			if allocatable > 0 {
				return allocatable, allocated
			}
		}
		if exists {
			return allocatable, nodeInfo.GetRequested().GetScalarResources()[resource]
		}
	}
	klog.FromContext(ctx).V(10).Info("Requested resource is omitted for node score calculation", "resourceName", resource)
	return 0, 0
}

// calculatePodResourceRequest returns the total non-zero requests. If Overhead is defined for the pod
// the Overhead is added to the result.
func (r *resourceAllocationScorer) calculatePodResourceRequest(pod *v1.Pod, resourceName v1.ResourceName) int64 {

	opts := resourcehelper.PodResourcesOptions{
		UseStatusResources: true,
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
func (r *resourceAllocationScorer) calculateDRAResourceTotals(ctx context.Context, node *v1.Node, deviceClass *resourceapi.DeviceClass, allocatedState *structured.AllocatedState, resourceSlices []*draapi.ResourceSlice,
) (int64, int64, error) {
	var totalCapacity, totalAllocated int64
	nodeName := node.Name

	for _, slice := range resourceSlices {
		// Early filtering: check if slice applies to this node
		perDeviceNodeSelection := ptr.Deref(slice.Spec.PerDeviceNodeSelection, false)

		var devices []draapi.Device

		if perDeviceNodeSelection {
			// Per-device node selection: filter devices individually
			devices = make([]draapi.Device, 0, len(slice.Spec.Devices))
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
			sliceAllNodes := slice.Spec.AllNodes

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
				// TODO (?): use unique strings in structured.IsDeviceAllocated
				deviceID := structured.MakeDeviceID(driver.String(), pool.String(), device.Name.String())
				if structured.IsDeviceAllocated(deviceID, allocatedState) {
					totalAllocated++
				}
			}
		} else {
			// Slow path: check device class selectors
			for _, device := range devices {
				matches, err := r.deviceMatchesClass(ctx, slice, device, deviceClass)
				if err != nil {
					return 0, 0, err
				}
				if matches {
					totalCapacity++
					deviceID := structured.DeviceID{Driver: slice.Spec.Driver, Pool: slice.Spec.Pool.Name, Device: device.Name}
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
func (r *resourceAllocationScorer) deviceMatchesClass(ctx context.Context, slice *draapi.ResourceSlice, device draapi.Device, deviceClass *resourceapi.DeviceClass) (bool, error) {
	// If no selectors are defined, all devices match
	if len(deviceClass.Spec.Selectors) == 0 {
		return true, nil
	}

	match := r.matchResultsForSlice(slice)

	// Lazily create the CEL device only when needed (first CEL selector that's not cached)
	var celDevice cel.Device
	celDeviceCreated := false

	// All selectors must match for the device to be considered a match
	for _, selector := range deviceClass.Spec.Selectors {
		if selector.CEL == nil {
			continue
		}

		// Check if result is already cached
		var key deviceMatchKey
		if match != nil {
			key = deviceMatchKey{expression: selector.CEL.Expression, device: device.Name}
			if matches, ok := match.results.Load(key); ok {
				if !matches.(bool) {
					return false, nil
				}
				continue // This selector matches, check the next one
			}
		}

		// Cache miss - need to evaluate CEL expression
		// Create CEL device if we haven't already
		if !celDeviceCreated {
			celDevice = cel.Device{
				Driver:             slice.Spec.Driver.String(),
				LookupUniqueString: slice.LookupUniqueString,
				Attributes:         device.Attributes,
				Capacity:           device.Capacity,
			}
			celDeviceCreated = true
		}

		// Use cached CEL compilation for performance
		result := r.celCache.GetOrCompile(selector.CEL.Expression)
		if result.Error != nil {
			return false, result.Error
		}

		matches, _, err := result.DeviceMatches(ctx, celDevice)
		if err != nil {
			return false, err
		}

		// Cache the result for future use
		if match != nil {
			match.results.Store(key, matches)
		}
		if !matches {
			return false, nil
		}
	}

	return true, nil
}

// TODO(#141166): use framework.SaturatingAdd for pod and node placement sums
func (r *resourceAllocationScorer) scorePlacement(
	ctx context.Context,
	podGroupInfo fwk.PodGroupInfo,
	podGroupAssignments *fwk.PodGroupAssignments,
) (int64, *fwk.Status) {
	requested := make([]int64, len(r.resources))
	// Calculate requests for the pod group pods scheduled for this placement
	for _, assignment := range podGroupAssignments.ProposedAssignments {
		pod := assignment.GetPod()
		podRequests := r.calculatePodResourceRequestList(pod, r.resources)
		for i := range len(r.resources) {
			requested[i] += podRequests[i]
		}
	}
	allocatable := make([]int64, len(r.resources))
	allocated := make([]int64, len(r.resources))
	// Calculate resources on all nodes in this placement
	for _, node := range podGroupAssignments.Nodes {
		_, nodeAllocated, nodeAllocatable := r.calculateNodeAllocatableRequest(ctx, node, requested, nil)
		for i := range r.resources {
			allocatable[i] += nodeAllocatable[i]
			allocated[i] += nodeAllocated[i]
			// requested includes both the already existing pods
			// and the pod group pods that are being scheduled for this placement
			requested[i] += nodeAllocated[i]
		}
	}

	score := r.scorer(requested, allocated, allocatable)

	logger := klog.FromContext(ctx)
	if loggerV := logger.V(10); loggerV.Enabled() { // Serializing these maps is costly.
		loggerV.Info("Listed internal info for placement's allocatable resources, requested resources and placement score", "podGroup",
			klog.KRef(podGroupInfo.GetNamespace(), podGroupInfo.GetName()), "placement", podGroupAssignments.Name, "resourceAllocationScorer", r.Name,
			"allocatableResource", allocatable, "requestedResource", requested, "resourceScore", score,
		)
	}

	return score, nil
}
