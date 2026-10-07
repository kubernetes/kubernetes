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
	"fmt"
	"sort"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// calculateAndCheckNodeAllocatableResources calculates the total node-allocatable resources (e.g., CPU, memory)
// requested by a pod, considering both its standard container requests and any additional resources
// derived from its DRA claims. It then checks if this aggregated demand fits within the node's remaining allocatable capacity.
func (pl *DynamicResources) calculateAndCheckNodeAllocatableResources(ctx context.Context, state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocations map[types.UID]*resourceapi.AllocationResult) ([]v1.AdditionalNodeAllocatableResource, *fwk.Status) {
	logger := klog.FromContext(ctx)

	nodeAllocatableClaims := []*resourceapi.ResourceClaim{}

	var err error

	nodeSlices, err := filterSlicesForNode(pl.draManager, nodeInfo.Node())
	if err != nil {
		logger.Error(err, "Failed to list resource slices")
		return nil, statusError(logger, err)
	}

	for _, claim := range state.claims.all() {
		alloc := claim.Status.Allocation
		if a, ok := allocations[claim.UID]; ok {
			alloc = a
		}
		if alloc != nil && len(alloc.Devices.Results) > 0 {
			for _, result := range alloc.Devices.Results {
				device, err := getDeviceFromSlices(nodeSlices, &result, nodeInfo.Node())
				if err != nil {
					logger.Error(err, "Failed to get device from manager", "claim", klog.KObj(claim), "device", result.Device, "pool", result.Pool, "driver", result.Driver)
					continue
				}
				if device != nil && len(device.NodeAllocatableResources) > 0 {
					nodeAllocatableClaims = append(nodeAllocatableClaims, claim)
					break
				}
			}
		}
	}

	if len(nodeAllocatableClaims) == 0 {
		return nil, nil // No nodeAllocatable resources to check
	}

	totalPodDemand, additionalResources, status := pl.getPodNodeAllocatableResourceFootprint(logger, state, pod, allocations, nodeAllocatableClaims, nodeSlices, nodeInfo.Node())
	if status != nil {
		logger.V(5).Info("calculateAndCheckNodeAllocatableResources: getPodNodeAllocatableResourceFootprint failed", "status", status)
		return nil, status
	}

	if status := pl.nodeFitsResources(nodeInfo, totalPodDemand); status != nil {
		return nil, status
	}
	logger.V(5).Info("Pod fits on node ( including DRA Node Allocatable Resources)", "pod", klog.KObj(pod), "node", nodeInfo.Node().Name)
	return additionalResources, nil
}

func getDeviceFromManager(draManager fwk.SharedDRAManager, result *resourceapi.DeviceRequestAllocationResult) (*resourceapi.Device, error) {
	slices, err := draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return nil, fmt.Errorf("listing resource slices: %w", err)
	}
	return getDeviceFromSlices(slices, result, nil)
}

func filterSlicesForNode(draManager fwk.SharedDRAManager, node *v1.Node) ([]*resourceapi.ResourceSlice, error) {
	slices, err := draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return nil, fmt.Errorf("listing resource slices: %w", err)
	}

	if node == nil {
		return slices, nil
	}
	var nodeSlices []*resourceapi.ResourceSlice
	for _, slice := range slices {
		if slice.Spec.NodeName != nil && *slice.Spec.NodeName == node.Name {
			nodeSlices = append(nodeSlices, slice)
		} else if slice.Spec.AllNodes != nil && *slice.Spec.AllNodes {
			nodeSlices = append(nodeSlices, slice)
		} else if slice.Spec.NodeSelector != nil {
			selector, err := nodeaffinity.NewNodeSelector(slice.Spec.NodeSelector)
			if err == nil && selector.Match(node) {
				nodeSlices = append(nodeSlices, slice)
			}
		} else if slice.Spec.PerDeviceNodeSelection != nil && *slice.Spec.PerDeviceNodeSelection {
			nodeSlices = append(nodeSlices, slice)
		}
	}
	return nodeSlices, nil
}

func deviceMatchesNode(device *resourceapi.Device, node *v1.Node) bool {
	if node == nil {
		return true
	}
	if device.NodeName != nil && *device.NodeName == node.Name {
		return true
	}
	if device.AllNodes != nil && *device.AllNodes {
		return true
	}
	if device.NodeSelector != nil {
		selector, err := nodeaffinity.NewNodeSelector(device.NodeSelector)
		if err == nil && selector.Match(node) {
			return true
		}
	}
	return false
}

func getDeviceFromSlices(slices []*resourceapi.ResourceSlice, result *resourceapi.DeviceRequestAllocationResult, node *v1.Node) (*resourceapi.Device, error) {
	for _, slice := range slices {
		if slice.Spec.Driver == result.Driver && slice.Spec.Pool.Name == result.Pool {
			for i := range slice.Spec.Devices {
				if slice.Spec.Devices[i].Name == result.Device {
					device := &slice.Spec.Devices[i]
					if slice.Spec.PerDeviceNodeSelection != nil && *slice.Spec.PerDeviceNodeSelection {
						if !deviceMatchesNode(device, node) {
							return nil, fmt.Errorf("device %s does not match node", result.Device)
						}
					}
					return device, nil
				}
			}
		}
	}
	return nil, fmt.Errorf("device %s not found in pool %s for driver %s", result.Device, result.Pool, result.Driver)
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
		if consumed, exists := result.ConsumedCapacity[capacityKey]; exists {
			// If !exists - the capacityKey is not in ConsumedCapacity, this mapping is not relevant for this allocation
			consumedQuantity := consumed.DeepCopy()
			quantityOne := resource.MustParse("1")
			if mappingMap.CapacityMultiplier != nil && !mappingMap.CapacityMultiplier.Equal(quantityOne) {
				multiplier := mappingMap.CapacityMultiplier.DeepCopy()
				qDec := consumedQuantity.AsDec()
				qDec.Mul(qDec, multiplier.AsDec())
				consumedQuantity = *resource.NewDecimalQuantity(*qDec, consumedQuantity.Format)
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
	totalResources map[v1.ResourceName]v1.NodeAllocatableOverheadResources,
) {
	if newOverhead.PerPod == nil && newOverhead.PerContainer == nil {
		return
	}

	current, exists := totalResources[resourceName]
	if !exists {
		current = v1.NodeAllocatableOverheadResources{Name: resourceName}
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

// buildNodeAllocatableDRAInfo processes the node allocatable resource allocations for a pod.
// It translates the allocated devices and quantities from DRA claims into a list of v1.AdditionalNodeAllocatableResource.
func (pl *DynamicResources) buildNodeAllocatableDRAInfo(pod *v1.Pod, nodeAllocatableClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult, claimNametoUID map[string]types.UID, slices []*resourceapi.ResourceSlice, node *v1.Node) ([]v1.AdditionalNodeAllocatableResource, error) {
	if len(nodeAllocatableClaimAllocations) == 0 {
		return []v1.AdditionalNodeAllocatableResource{}, nil
	}

	claimToResource := make(map[types.UID]v1.AdditionalNodeAllocatableResource)

	for key, alloc := range nodeAllocatableClaimAllocations {
		totalDirectMappedResourcesPerClaim := make(map[v1.ResourceName]resource.Quantity)
		totalOverheadResourcesPerClaim := make(map[v1.ResourceName]v1.NodeAllocatableOverheadResources)

		for _, result := range alloc.Devices.Results {
			device, err := getDeviceFromSlices(slices, &result, node)
			if err != nil {
				return nil, fmt.Errorf("claim %s/%s, device %s, driver %s: %w", key.Namespace, key.Name, result.Device, result.Driver, err)
			}
			if device == nil || device.NodeAllocatableResources == nil {
				continue
			}

			for resourceName, resourceMap := range device.NodeAllocatableResources {
				if resourceMap.Mapping != nil {
					if err := addDeviceMapping(resourceName, resourceMap.Mapping, &result, key, totalDirectMappedResourcesPerClaim); err != nil {
						return nil, err
					}
				}
				if resourceMap.Overhead != nil {
					addDeviceOverhead(resourceName, resourceMap.Overhead, totalOverheadResourcesPerClaim)
				}
			}
		}

		if len(totalDirectMappedResourcesPerClaim) > 0 || len(totalOverheadResourcesPerClaim) > 0 {
			res := v1.AdditionalNodeAllocatableResource{
				Source: v1.AdditionalNodeAllocatableReference{
					APIGroup: resourceapi.GroupName,
					Kind:     "ResourceClaim",
					Name:     key.Name,
				},
				Containers: []string{},
				Mapping:    []v1.NodeAllocatableMappedResources{},
				Overhead:   []v1.NodeAllocatableOverheadResources{},
			}

			for name, quantity := range totalDirectMappedResourcesPerClaim {
				q := quantity.DeepCopy()
				res.Mapping = append(res.Mapping, v1.NodeAllocatableMappedResources{
					Name:     name,
					Quantity: &q,
				})
			}
			for _, overhead := range totalOverheadResourcesPerClaim {
				res.Overhead = append(res.Overhead, overhead)
			}

			sort.Slice(res.Mapping, func(i, j int) bool {
				return res.Mapping[i].Name < res.Mapping[j].Name
			})
			sort.Slice(res.Overhead, func(i, j int) bool {
				return res.Overhead[i].Name < res.Overhead[j].Name
			})
			claimToResource[key.UID] = res
		}
	}

	for _, containers := range [][]v1.Container{pod.Spec.InitContainers, pod.Spec.Containers} {
		for _, container := range containers {
			for _, podClaim := range container.Resources.Claims {
				if claimUID, ok := claimNametoUID[podClaim.Name]; ok {
					if res, ok := claimToResource[claimUID]; ok {
						res.Containers = append(res.Containers, container.Name)
						claimToResource[claimUID] = res
					}
				}
			}
		}
	}

	additionalResources := make([]v1.AdditionalNodeAllocatableResource, 0, len(claimToResource))
	for _, res := range claimToResource {
		additionalResources = append(additionalResources, res)
	}

	// Sort the results for consistent output.
	sort.Slice(additionalResources, func(i, j int) bool {
		return additionalResources[i].Source.Name < additionalResources[j].Source.Name
	})

	return additionalResources, nil
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
	if len(pod.Status.AdditionalNodeAllocatableResources) == 0 {
		return nil
	}

	if pod.Spec.Resources.Requests != nil {
		// Calculate Sum(Containers) + DRA + overhead resources. Skip pod level resources for this sum
		opts := resourcehelper.PodResourcesOptions{
			SkipPodLevelResources:                 true,
			UseAdditionalNodeAllocatableResources: true,
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
			SkipPodLevelResources:                 true,
			UseAdditionalNodeAllocatableResources: true,
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
		containerDRAAllocations := make(map[string]v1.ResourceList, len(pod.Spec.Containers))
		for _, ctr := range pod.Spec.Containers {
			containerDRAAllocations[ctr.Name] = resourcehelper.GetContainerDRAAllocations(pod, ctr.Name)
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

				draResAllocation := containerDRAAllocations[ctr.Name][resourceName]
				effectiveLimit := ctrLimit.DeepCopy()
				effectiveLimit.Add(draResAllocation)

				if effectiveLimit.Cmp(podLevelResLimit) > 0 {
					return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod level limit for %s is insufficient to cover the limit and DRA overhead for container %s", resourceName, ctr.Name))
				}
			}
		}
	}

	return nil
}

// getPodNodeAllocatableResourceFootprint determines the total nodeAllocatable resource demand of a pod.
func (pl *DynamicResources) getPodNodeAllocatableResourceFootprint(logger klog.Logger, state *stateData, pod *v1.Pod, allocations map[types.UID]*resourceapi.AllocationResult, nodeAllocatableClaims []*resourceapi.ResourceClaim, slices []*resourceapi.ResourceSlice, node *v1.Node) (*framework.Resource, []v1.AdditionalNodeAllocatableResource, *fwk.Status) {
	nodeAllocatableDRAAllocations := make(map[v1.ObjectReference]*resourceapi.AllocationResult)
	// Add pre-allocated claims
	for _, claim := range nodeAllocatableClaims {
		name := claim.Name
		if isSpecialClaimName(name) {
			// Pod status must reference the name the claim gets in the API.
			name = state.draExtendedResource.preGeneratedClaimName
		}
		key := v1.ObjectReference{
			Namespace: claim.Namespace,
			Name:      name,
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

	additionalResources, err := pl.buildNodeAllocatableDRAInfo(pod, nodeAllocatableDRAAllocations, claimNametoUID, slices, node)
	if err != nil {
		return nil, nil, statusError(logger, err)
	}

	// Calculate the final totalPodDemand to be used for node fitting
	optsTotal := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:                 !pl.fts.EnablePodLevelResources,
		UseAdditionalNodeAllocatableResources: true,
	}
	// Perform shallow copy - we only use the resource information from container and pod spec to calculatate the resource foot print.
	podCopy := *pod
	podCopy.Status.AdditionalNodeAllocatableResources = additionalResources
	totalPodDemandRes := resourcehelper.PodRequests(&podCopy, optsTotal)

	// The API validation in pkg/apis/core/validation/validation.go only checks pod.Spec.Resources against container
	// requests within the Spec. It cannot account for DRA-derived resources, which are determined after device allocation.
	if status := pl.validatePodLevelResourcesCoverDRA(&podCopy); status != nil {
		return nil, nil, status
	}

	totalPodDemand := framework.NewResource(totalPodDemandRes)
	logger.V(5).Info("Total Pod Demand After DRA", "pod", klog.KObj(pod), "demand", totalPodDemand)

	return totalPodDemand, additionalResources, nil
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
