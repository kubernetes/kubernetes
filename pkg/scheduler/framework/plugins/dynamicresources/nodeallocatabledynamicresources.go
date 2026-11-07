/*
Copyright 2025 The Kubernetes Authors.

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

package dynamicresources

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	resourcehelper "k8s.io/component-helpers/resource"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// calculateAndCheckNodeAllocatableResources calculates the total node-allocatable resources (e.g., CPU, memory)
// requested by a pod, considering both its standard container requests and any additional resources
// derived from its DRA claims. It then checks if this aggregated demand fits within the node's remaining allocatable capacity.
func (pl *DynamicResources) calculateAndCheckNodeAllocatableResources(ctx context.Context, state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocations map[types.UID]*resourceapi.AllocationResult, extendedResourceRequestMappings []v1.ContainerExtendedResourceRequest) ([]v1.ExtraNodeAllocatableResource, *fwk.Status) {
	if len(state.nodeAllocatableSlices) == 0 {
		return nil, nil
	}
	logger := klog.FromContext(ctx)

	nodeAllocatableClaims := []*resourceapi.ResourceClaim{}

	for _, claim := range state.claims.all() {
		alloc := claim.Status.Allocation
		if a, ok := allocations[claim.UID]; ok {
			alloc = a
		}
		if alloc != nil && len(alloc.Devices.Results) > 0 {
			for _, result := range alloc.Devices.Results {
				if result.AdminAccess != nil && *result.AdminAccess {
					continue
				}
				if nodeAllocatableDevice(state.nodeAllocatableSlices, &result) != nil {
					nodeAllocatableClaims = append(nodeAllocatableClaims, claim)
					break
				}
			}
		}
	}

	if len(nodeAllocatableClaims) == 0 {
		return nil, nil // No nodeAllocatable resources to check
	}

	otherEntries := state.extraNodeAllocatableResources.GetExcludingPlugin(nodeInfo.Node().Name, Name)
	totalPodDemand, extraNodeAllocatableResources, status := pl.getPodNodeAllocatableResourceFootprint(logger, pod, allocations, nodeAllocatableClaims, extendedResourceRequestMappings, otherEntries, state.nodeAllocatableSlices)
	if status != nil {
		logger.V(5).Info("calculateAndCheckNodeAllocatableResources: getPodNodeAllocatableResourceFootprint failed", "status", status)
		return nil, status
	}

	if status := pl.nodeFitsResources(nodeInfo, totalPodDemand); status != nil {
		return nil, status
	}
	logger.V(5).Info("Pod fits on node ( including DRA Node Allocatable Resources)", "pod", klog.KObj(pod), "node", nodeInfo.Node().Name)
	return extraNodeAllocatableResources, nil
}

// allocationHasNodeAllocatableMapping reports whether any device of the allocation maps node allocatable resources.
func allocationHasNodeAllocatableMapping(alloc *resourceapi.AllocationResult, nodeAllocatableSlices map[string][]*draapi.ResourceSlice) bool {
	for i := range alloc.Devices.Results {
		result := &alloc.Devices.Results[i]
		if result.AdminAccess != nil && *result.AdminAccess {
			continue
		}
		device := nodeAllocatableDevice(nodeAllocatableSlices, result)
		if device == nil {
			continue
		}
		for _, r := range device.NodeAllocatableResources {
			if r.Mapping != nil {
				return true
			}
		}
	}
	return false
}

// nodeAllocatableSlicesByPool returns the slices that have devices with node allocatable resources, by pool name.
func nodeAllocatableSlicesByPool(slices []*draapi.ResourceSlice) map[string][]*draapi.ResourceSlice {
	pools := make(map[string][]*draapi.ResourceSlice)
	for _, slice := range slices {
		for i := range slice.Spec.Devices {
			if len(slice.Spec.Devices[i].NodeAllocatableResources) > 0 {
				pool := slice.Spec.Pool.Name.String()
				pools[pool] = append(pools[pool], slice)
				break
			}
		}
	}
	return pools
}

// nodeAllocatableDevice returns the allocated device if it has node allocatable resources.
func nodeAllocatableDevice(nodeAllocatableSlices map[string][]*draapi.ResourceSlice, result *resourceapi.DeviceRequestAllocationResult) *draapi.Device {
	for _, slice := range nodeAllocatableSlices[result.Pool] {
		if slice.Spec.Driver.String() != result.Driver {
			continue
		}
		for i := range slice.Spec.Devices {
			device := &slice.Spec.Devices[i]
			if device.Name.String() == result.Device && len(device.NodeAllocatableResources) > 0 {
				return device
			}
		}
	}
	return nil
}

// addDeviceMapping calculates the mapping quantity for a device and adds it to the totals map.
func addDeviceMapping(
	resourceName v1.ResourceName,
	mappingMap *resourceapi.NodeAllocatableMapping,
	result *resourceapi.DeviceRequestAllocationResult,
	key v1.ObjectReference,
	totalResources map[v1.ResourceName]resource.Quantity,
) error {
	if mappingMap.CapacityKey != nil && *mappingMap.CapacityKey != "" {
		capacityKey := *mappingMap.CapacityKey
		if result.ConsumedCapacity == nil {
			return fmt.Errorf("claim %s/%s, device %s: ConsumedCapacity is nil, but Capacity key '%s' is set in NodeAllocatableResources for resource %s", key.Namespace, key.Name, result.Device, capacityKey, resourceName)
		}
		if consumed, exists := draapi.LookupByQualifiedName(result.ConsumedCapacity, capacityKey, result.Driver); exists {
			if consumed.Sign() < 0 {
				return fmt.Errorf("claim %s/%s, device %s: ConsumedCapacity for capacity key '%s' must not be negative, got %s", key.Namespace, key.Name, result.Device, capacityKey, consumed.String())
			}
			// If !exists - the capacityKey is not in ConsumedCapacity, this mapping is not relevant for this allocation
			consumedQuantity := consumed.DeepCopy()
			quantityOne := resource.MustParse("1")
			if mappingMap.CapacityMultiplier != nil && !mappingMap.CapacityMultiplier.Equal(quantityOne) {
				multiplier := mappingMap.CapacityMultiplier.DeepCopy()
				qDec := consumedQuantity.AsDec()
				qDec.Mul(qDec, multiplier.AsDec())
				consumedQuantity = *resource.NewDecimalQuantity(*qDec, consumedQuantity.Format)
			}
			if consumedQuantity.Sign() < 0 {
				return fmt.Errorf("claim %s/%s, device %s: computed node-allocatable footprint for resource %s must not be negative, got %s", key.Namespace, key.Name, result.Device, resourceName, consumedQuantity.String())
			}
			current := totalResources[resourceName]
			current.Add(consumedQuantity)
			totalResources[resourceName] = current
		}
		return nil
	}

	// Note: For the same device, we cannot have both DeviceMultiplier and CapacityKey set (enforced during API validation).
	// Therefore, it is safe to treat these code paths as mutually exclusive here.
	if mappingMap.DeviceMultiplier != nil {
		current := totalResources[resourceName]
		current.Add(mappingMap.DeviceMultiplier.DeepCopy())
		totalResources[resourceName] = current
	}
	return nil
}

// addDeviceOverhead calculates the overhead resources for a device and adds them to the totals map.
func addDeviceOverhead(
	resourceName v1.ResourceName,
	newOverhead *resourceapi.NodeAllocatableOverhead,
	totalResources map[v1.ResourceName]resourceapi.NodeAllocatableOverhead,
) {
	if newOverhead.PerPod == nil && newOverhead.PerContainer == nil {
		return
	}

	current, exists := totalResources[resourceName]
	if !exists {
		current = resourceapi.NodeAllocatableOverhead{}
	}

	if newOverhead.PerPod != nil {
		if current.PerPod == nil {
			q := newOverhead.PerPod.DeepCopy()
			current.PerPod = &q
		} else {
			current.PerPod.Add(*newOverhead.PerPod)
		}
	}
	if newOverhead.PerContainer != nil {
		if current.PerContainer == nil {
			q := newOverhead.PerContainer.DeepCopy()
			current.PerContainer = &q
		} else {
			current.PerContainer.Add(*newOverhead.PerContainer)
		}
	}

	totalResources[resourceName] = current
}

// addResourceList adds the quantities in newList to list.
func addResourceList(list, newList v1.ResourceList) {
	for name, quantity := range newList {
		q := list[name]
		q.Add(quantity)
		list[name] = q
	}
}

// deviceContainers returns the sorted names of the containers that use a device of the claim.
func deviceContainers(pod *v1.Pod, claim v1.ObjectReference, result *resourceapi.DeviceRequestAllocationResult, claimNametoUID map[string]types.UID, extendedResourceRequestMappings []v1.ContainerExtendedResourceRequest) []string {
	containers := []string{}
	if isSpecialClaimName(claim.Name) {
		// Extended resource claims are synthesized by the scheduler and do not appear in
		// container.Resources.Claims, so their requesting containers come from the request mappings.
		for _, m := range extendedResourceRequestMappings {
			if m.RequestName == result.Request {
				containers = append(containers, m.ContainerName)
			}
		}
	} else {
		// Containers name main requests, so a device allocated for subrequest "gpu/large" belongs to "gpu".
		request := resourceclaim.BaseRequestRef(result.Request)
		usesDevice := func(ref v1.ResourceClaim) bool {
			if uid, ok := claimNametoUID[ref.Name]; !ok || uid != claim.UID {
				return false
			}
			// Without an explicit request, the container gets every device of the claim.
			return ref.Request == "" || ref.Request == request
		}
		for _, ctrs := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
			for _, ctr := range ctrs {
				if slices.ContainsFunc(ctr.Resources.Claims, usesDevice) {
					containers = append(containers, ctr.Name)
				}
			}
		}
	}
	slices.Sort(containers)
	return slices.Compact(containers)
}

// buildNodeAllocatableDRAInfo processes the node allocatable resource allocations for a pod.
// It translates the allocated devices and quantities from DRA claims into a list of ExtraNodeAllocatableResource
func (pl *DynamicResources) buildNodeAllocatableDRAInfo(pod *v1.Pod, nodeAllocatableClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult, claimNametoUID map[string]types.UID, extendedResourceRequestMappings []v1.ContainerExtendedResourceRequest, nodeAllocatableSlices map[string][]*draapi.ResourceSlice) ([]v1.ExtraNodeAllocatableResource, error) {
	if len(nodeAllocatableClaimAllocations) == 0 {
		return []v1.ExtraNodeAllocatableResource{}, nil
	}
	// The devices of a claim that the same containers use form a group.
	type deviceGroup struct {
		claim      types.UID
		containers string
	}
	groupContainers := make(map[deviceGroup][]string)
	totalDirectMappedResourcesPerGroup := make(map[deviceGroup]map[v1.ResourceName]resource.Quantity)
	totalOverheadResourcesPerGroup := make(map[deviceGroup]map[v1.ResourceName]resourceapi.NodeAllocatableOverhead)

	for key, alloc := range nodeAllocatableClaimAllocations {
		for _, result := range alloc.Devices.Results {
			if result.AdminAccess != nil && *result.AdminAccess {
				continue
			}
			device := nodeAllocatableDevice(nodeAllocatableSlices, &result)
			if device == nil {
				continue
			}

			containers := deviceContainers(pod, key, &result, claimNametoUID, extendedResourceRequestMappings)
			group := deviceGroup{claim: key.UID, containers: strings.Join(containers, ",")}
			if _, ok := groupContainers[group]; !ok {
				groupContainers[group] = containers
				totalDirectMappedResourcesPerGroup[group] = make(map[v1.ResourceName]resource.Quantity)
				totalOverheadResourcesPerGroup[group] = make(map[v1.ResourceName]resourceapi.NodeAllocatableOverhead)
			}
			for resourceName, resourceMap := range device.NodeAllocatableResources {
				if resourceMap.Mapping != nil {
					if err := addDeviceMapping(resourceName, resourceMap.Mapping, &result, key, totalDirectMappedResourcesPerGroup[group]); err != nil {
						return nil, err
					}
				}
				if resourceMap.Overhead != nil {
					addDeviceOverhead(resourceName, resourceMap.Overhead, totalOverheadResourcesPerGroup[group])
				}
			}
		}
	}

	// The pod contribution of a group is its mapping and perPod overhead, which the pod uses once.
	// The perContainer contribution is its perContainer overhead, which each container of the group uses.
	podContributions := v1.ResourceList{}
	containerRequests := make(map[string]v1.ResourceList)
	containerLimits := make(map[string]v1.ResourceList)
	for group, containers := range groupContainers {
		podContribution := v1.ResourceList{}
		addResourceList(podContribution, totalDirectMappedResourcesPerGroup[group])
		perContainerContribution := v1.ResourceList{}
		for name, overhead := range totalOverheadResourcesPerGroup[group] {
			if overhead.PerPod != nil {
				addResourceList(podContribution, v1.ResourceList{name: *overhead.PerPod})
			}
			if overhead.PerContainer != nil {
				addResourceList(perContainerContribution, v1.ResourceList{name: *overhead.PerContainer})
			}
		}
		addResourceList(podContributions, podContribution)
		for _, name := range containers {
			if containerRequests[name] == nil {
				containerRequests[name] = v1.ResourceList{}
				containerLimits[name] = v1.ResourceList{}
			}
			addResourceList(containerRequests[name], perContainerContribution)
			addResourceList(containerLimits[name], perContainerContribution)
			// The container has access to the pod resources, so we bump up the limits.
			// It is not in the requests, because we cannot count the pod entry once per each container using the devices.
			addResourceList(containerLimits[name], podContribution)
		}
	}

	var containerEntries []v1.ExtraNodeAllocatableResource
	for container := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		limits := containerLimits[container.Name]
		if len(limits) > 0 {
			entry := v1.ExtraNodeAllocatableResource{ContainerName: container.Name, Limits: limits}
			if requests := containerRequests[container.Name]; len(requests) > 0 {
				entry.Requests = requests
			}
			containerEntries = append(containerEntries, entry)
		}
	}
	// The pod entry has the pod contribution of each group once, plus the container's requests
	// with init containers at their peak.
	podRequests := resourcehelper.AggregateContainerRequestsFromSource(pod, containerRequests)
	addResourceList(podRequests, podContributions)
	if len(podRequests) == 0 {
		return []v1.ExtraNodeAllocatableResource{}, nil
	}

	podEntry := v1.ExtraNodeAllocatableResource{Requests: podRequests, Limits: podRequests.DeepCopy()}
	return append([]v1.ExtraNodeAllocatableResource{podEntry}, containerEntries...), nil
}

// validateNodeAllocatableDRAClaimSharing ensures that a node-allocatable DRA claim is not already in use by another pod on this node.
func (pl *DynamicResources) validateNodeAllocatableDRAClaimSharing(pod *v1.Pod, claim *resourceapi.ResourceClaim, state *stateData, podGroupState *podGroupStateData) *fwk.Status {
	if claim == nil {
		return nil
	}

	if !state.claimHasNodeAllocatableMappedDevice[claim.UID] {
		// Overhead-only claims are allowed to be shared.
		return nil
	}

	// If the claim is already reserved for another pod or pod group, fail immediately.
	if len(claim.Status.ReservedFor) > 0 && !resourceclaim.IsReservedForPod(pod, claim, pl.fts.EnableDRAWorkloadResourceClaims) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node allocatable resource claim %s has a mapped device and cannot be shared across pods", claim.Name))
	}

	// Check if another pod in the same PodGroup has reserved/allocated this claim in the current cycle.
	if podGroupState != nil {
		if pendingAllocPodUIDs, ok := podGroupState.pendingAllocations[claim.UID]; ok {
			for uid := range pendingAllocPodUIDs {
				if uid != pod.UID {
					return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node allocatable resource claim %s has a mapped device and cannot be shared across pods", claim.Name))
				}
			}
		}
	}

	return nil
}

// validatePodLevelResourcesCoverDRA checks if
// 1. pod-level requests is not less than the AGGREGATE container level requests including DRA
// 2. pod-level limits is not less than the INDIVIDUAL container level limits including DRA
// This follows the same pattern as in validatePodResourceConsistency() in pkg/apis/core/validation/validation.go
func (pl *DynamicResources) validatePodLevelResourcesCoverDRA(pod *v1.Pod) *fwk.Status {
	if !pl.fts.EnablePodLevelResources || pod.Spec.Resources == nil {
		return nil
	}
	if len(pod.Status.ExtraNodeAllocatableResources) == 0 {
		return nil
	}

	if pod.Spec.Resources.Requests != nil {
		// Calculate Sum(Containers) + DRA + overhead resources. Skip pod level resources for this sum
		opts := resourcehelper.PodResourcesOptions{
			SkipPodLevelResources:            true,
			UseExtraNodeAllocatableResources: true,
		}
		requestWithoutPodLevel := resourcehelper.AggregateContainerRequests(pod, opts)

		// For resources specified at pod level, check if container and DRA aggregates do not exceed pod level budget.
		for resName, podLevelReq := range pod.Spec.Resources.Requests {
			if !resourcehelper.IsSupportedPodLevelResource(resName) {
				continue
			}
			val, ok := requestWithoutPodLevel[resName]
			if !ok {
				continue
			}
			if val.Cmp(podLevelReq) > 0 {
				return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod level request for %s is insufficient to cover the aggregated container and node-allocatable DRA requests", resName))
			}
		}
	}

	if pod.Spec.Resources.Limits != nil {
		opts := resourcehelper.PodResourcesOptions{
			SkipPodLevelResources:            true,
			UseExtraNodeAllocatableResources: true,
		}
		limitsWithoutPodLevel := resourcehelper.AggregateContainerLimits(pod, opts)

		// Pod level hugepage limits must be always equal or greater than the aggregated
		// container level hugepage limits + DRA limits
		for resourceName, ctrLims := range limitsWithoutPodLevel {
			if !v1helper.IsHugePageResourceName(resourceName) {
				continue
			}

			podLevelResLimit, hasLimit := pod.Spec.Resources.Limits[resourceName]
			if !hasLimit {
				continue
			}

			if ctrLims.Cmp(podLevelResLimit) > 0 {
				return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod level limit for %s is insufficient to cover the aggregated container and node-allocatable DRA limits", resourceName))
			}
		}

		// Individual Container limits + DRA overheads must be <= Pod-level limits.
		containerExtraNodeAllocatableLimits := make(map[string]v1.ResourceList, len(pod.Spec.Containers))
		for _, ctr := range pod.Spec.Containers {
			containerExtraNodeAllocatableLimits[ctr.Name] = resourcehelper.GetExtraNodeAllocatableResourcesLimits(pod, ctr.Name)
		}

		for _, ctr := range pod.Spec.Containers {
			for resourceName, ctrLimit := range ctr.Resources.Limits {
				if v1helper.IsHugePageResourceName(resourceName) {
					continue
				}

				// Skip if the pod-level limit of the resource is not set.
				podLevelResLimit, exists := pod.Spec.Resources.Limits[resourceName]
				if !exists {
					continue
				}

				extraNodeAllocatableLimit := containerExtraNodeAllocatableLimits[ctr.Name][resourceName]
				effectiveLimit := ctrLimit.DeepCopy()
				effectiveLimit.Add(extraNodeAllocatableLimit)

				if effectiveLimit.Cmp(podLevelResLimit) > 0 {
					return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod level limit for %s is insufficient to cover the limit and DRA overhead for container %s", resourceName, ctr.Name))
				}
			}
		}
	}

	return nil
}

// getPodNodeAllocatableResourceFootprint determines the total nodeAllocatable resource demand of a pod.
// otherEntries, recorded by other plugins, count towards the demand but are not returned.
func (pl *DynamicResources) getPodNodeAllocatableResourceFootprint(logger klog.Logger, pod *v1.Pod, allocations map[types.UID]*resourceapi.AllocationResult, nodeAllocatableClaims []*resourceapi.ResourceClaim, extendedResourceRequestMappings []v1.ContainerExtendedResourceRequest, otherEntries []v1.ExtraNodeAllocatableResource, nodeAllocatableSlices map[string][]*draapi.ResourceSlice) (*framework.Resource, []v1.ExtraNodeAllocatableResource, *fwk.Status) {
	nodeAllocatableDRAAllocations := make(map[v1.ObjectReference]*resourceapi.AllocationResult)
	// Add pre-allocated claims
	for _, claim := range nodeAllocatableClaims {
		key := v1.ObjectReference{
			Namespace: claim.Namespace,
			Name:      claim.Name,
			UID:       claim.UID,
		}
		if claim.Status.Allocation != nil {
			nodeAllocatableDRAAllocations[key] = claim.Status.Allocation
		}
		if alloc, ok := allocations[claim.UID]; ok {
			nodeAllocatableDRAAllocations[key] = alloc
		}
	}

	claimNametoUID := make(map[string]types.UID)
	if err := pl.foreachPodResourceClaim(pod, func(name string, claim *resourceapi.ResourceClaim) {
		claimNametoUID[name] = claim.UID
	}); err != nil {
		return nil, nil, statusError(logger, fmt.Errorf("processing pod resource claims: %w", err))
	}

	extraNodeAllocatableResources, err := pl.buildNodeAllocatableDRAInfo(pod, nodeAllocatableDRAAllocations, claimNametoUID, extendedResourceRequestMappings, nodeAllocatableSlices)
	if err != nil {
		return nil, nil, statusError(logger, err)
	}

	// Calculate the final totalPodDemand to be used for node fitting
	optsTotal := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:            !pl.fts.EnablePodLevelResources,
		UseExtraNodeAllocatableResources: true,
	}
	// Perform shallow copy - we only use the resource information from container and pod spec to calculatate the resource foot print.
	podCopy := *pod
	podCopy.Status.ExtraNodeAllocatableResources = framework.MergeExtraNodeAllocatableResources(otherEntries, extraNodeAllocatableResources)
	totalPodDemandRes := resourcehelper.PodRequests(&podCopy, optsTotal)

	// The API validation in pkg/apis/core/validation/validation.go only checks pod.Spec.Resources against container
	// requests within the Spec. It cannot account for DRA-derived resources, which are determined after device allocation.
	if status := pl.validatePodLevelResourcesCoverDRA(&podCopy); status != nil {
		return nil, nil, status
	}

	totalPodDemand := framework.NewResource(totalPodDemandRes)
	logger.V(5).Info("Total Pod Demand After DRA", "pod", klog.KObj(pod), "demand", totalPodDemand)

	return totalPodDemand, extraNodeAllocatableResources, nil
}

// insufficientResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type insufficientResource struct {
	resourceName v1.ResourceName
	// We explicitly have a parameter for reason to avoid formatting a message on the fly
	// for common resources, which is expensive for cluster autoscaler simulations.
	reason    string
	requested int64
	used      int64
	capacity  int64
	// unresolvable indicates whether this node could be schedulable for the pod by the preemption,
	// which is determined by comparing the node's size and the pod's request.
	unresolvable bool
}

// nodeFitsResources checks if the node has sufficient capacity to accommodate the pod's nodeAllocatable resource requirements.
// TODO(pravk03): Unify this with the fitsRequest() in pkg/scheduler/framework/plugins/noderesources/fit.go.
func (pl *DynamicResources) nodeFitsResources(nodeInfo fwk.NodeInfo, podRequest *framework.Resource) *fwk.Status {
	insufficientResources := make([]insufficientResource, 0, 4)

	if podRequest.MilliCPU > 0 && podRequest.MilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-nodeInfo.GetRequested().GetMilliCPU()) {
		insufficientResources = append(insufficientResources, insufficientResource{
			resourceName: v1.ResourceCPU,
			reason:       "Insufficient cpu",
			requested:    podRequest.MilliCPU,
			used:         nodeInfo.GetRequested().GetMilliCPU(),
			capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && podRequest.Memory > (nodeInfo.GetAllocatable().GetMemory()-nodeInfo.GetRequested().GetMemory()) {
		insufficientResources = append(insufficientResources, insufficientResource{
			resourceName: v1.ResourceMemory,
			reason:       "Insufficient memory",
			requested:    podRequest.Memory,
			used:         nodeInfo.GetRequested().GetMemory(),
			capacity:     nodeInfo.GetAllocatable().GetMemory(),
			unresolvable: podRequest.Memory > nodeInfo.GetAllocatable().GetMemory(),
		})
	}
	if podRequest.EphemeralStorage > 0 &&
		podRequest.EphemeralStorage > (nodeInfo.GetAllocatable().GetEphemeralStorage()-nodeInfo.GetRequested().GetEphemeralStorage()) {
		insufficientResources = append(insufficientResources, insufficientResource{
			resourceName: v1.ResourceEphemeralStorage,
			reason:       "Insufficient ephemeral-storage",
			requested:    podRequest.EphemeralStorage,
			used:         nodeInfo.GetRequested().GetEphemeralStorage(),
			capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
	}

	for resName, reqQuant := range podRequest.ScalarResources {
		if v1helper.IsHugePageResourceName(resName) {
			nodeCapacity := nodeInfo.GetAllocatable().GetScalarResources()[resName]
			nodeRequested := nodeInfo.GetRequested().GetScalarResources()[resName]
			available := nodeCapacity - nodeRequested

			if reqQuant > available {
				insufficientResources = append(insufficientResources, insufficientResource{
					resourceName: resName,
					reason:       fmt.Sprintf("Insufficient %s", resName),
					requested:    reqQuant,
					used:         nodeRequested,
					capacity:     nodeCapacity,
					unresolvable: reqQuant > nodeCapacity,
				})
			}
		}
	}

	if len(insufficientResources) > 0 {
		failureReasons := make([]string, 0, len(insufficientResources))
		statusCode := fwk.Unschedulable
		for i := range insufficientResources {
			failureReasons = append(failureReasons, insufficientResources[i].reason)
			if insufficientResources[i].unresolvable {
				statusCode = fwk.UnschedulableAndUnresolvable
			}
		}
		return fwk.NewStatus(statusCode, failureReasons...)
	}
	return nil
}

// draExtraNodeAllocatableResourcesStatus validates the calculated ExtraNodeAllocatableResources
// against the assumed pod status, and returns the entries of this plugin to be patched onto
// the pod's status during PreBind.
func (pl *DynamicResources) draExtraNodeAllocatableResourcesStatus(
	ctx context.Context,
	pod *v1.Pod,
	nodeName string,
	state *stateData,
) ([]v1.ExtraNodeAllocatableResource, error) {
	if !pl.fts.EnableDRANodeAllocatableResources || state.extraNodeAllocatableResources == nil {
		return nil, nil
	}
	nodeExtraResources := state.extraNodeAllocatableResources.Get(nodeName)
	if len(nodeExtraResources) == 0 {
		return nil, nil
	}
	logger := klog.FromContext(ctx)

	// The incoming 'pod' is from the scheduler cache and would have ExtraNodeAllocatableResources
	// of all plugins pre-populated in the assume phase without persisting to the API server.
	if !apiequality.Semantic.DeepEqual(pod.Status.ExtraNodeAllocatableResources, nodeExtraResources) {
		logger.V(5).Info("ExtraNodeAllocatableResources difference: assumed pod status does not match calculated status", "pod", klog.KObj(pod))
		return nil, errors.New("assumed pod status does not match calculated status to be patched")
	}

	return state.extraNodeAllocatableResources.GetFromPlugin(nodeName, Name), nil
}

// clearExtraNodeAllocatableResources removes the entries of all plugins from the pod status.
func (pl *DynamicResources) clearExtraNodeAllocatableResources(ctx context.Context, pod *v1.Pod) error {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Clearing ExtraNodeAllocatableResources", "pod", klog.KObj(pod))

	// An explicit empty list distinguishes an intentional clear from an old
	// client omitting a field that it does not know about. PatchPodStatus cannot
	// preserve that distinction because the field has an omitempty JSON tag.
	//
	// The uid is included as a precondition so the patch cannot silently apply
	// to a different pod object if this one got deleted and recreated with the
	// same name in the meantime.
	patch := fmt.Appendf(nil, `{"metadata":{"uid":%q},"status":{"extraNodeAllocatableResources":[]}}`, pod.UID)
	err := retry.OnError(retry.DefaultBackoff, schedutil.Retriable, func() error {
		_, err := pl.clientset.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.MergePatchType, patch, metav1.PatchOptions{}, "status")
		return err
	})
	if err != nil {
		return fmt.Errorf("clear extraNodeAllocatableResources of pod %s: %w", klog.KObj(pod), err)
	}
	logger.V(5).Info("Cleared ExtraNodeAllocatableResources", "pod", klog.KObj(pod))
	return nil
}
