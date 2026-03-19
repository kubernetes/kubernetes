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
	"sort"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// ExtractPodNodeAllocatableResourceClaimStatus returns the pod nodeAllocatable claim status stored in state
func ExtractPodNodeAllocatableResourceClaimStatus(logger klog.Logger, state fwk.CycleState, nodeName string) []v1.NodeAllocatableResourceClaimStatus {
	s, err := state.Read(names.DynamicResources)
	if err != nil {
		// DynamicResources plugin didn't run or no state
		return nil
	}

	draState, ok := s.(*stateData)
	if !ok {
		logger.Error(errors.New("invalid DynamicResources state type"), "Failed to cast CycleState data")
		return nil
	}

	if nodeAlloc, exists := draState.nodeAllocations[nodeName]; exists {
		return nodeAlloc.nodeAllocatableResourceClaimStatuses
	}

	return nil
}

// calculateAndCheckNodeAllocatableResources calculates the total node-allocatable resources (e.g., CPU, memory)
// requested by a pod, considering both its standard container requests and any additional resources
// derived from its DRA claims. It then checks if this aggregated demand fits within the node's remaining allocatable capacity.
func (pl *DynamicResources) calculateAndCheckNodeAllocatableResources(ctx context.Context, state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocations map[types.UID]*resourceapi.AllocationResult) ([]v1.NodeAllocatableResourceClaimStatus, *fwk.Status) {
	logger := klog.FromContext(ctx)

	nodeAllocatableClaims := []*resourceapi.ResourceClaim{}
	nodeAllocatableClaimUIDs := sets.New[types.UID]()

	for _, claim := range state.claims.all() {
		if alloc, ok := allocations[claim.UID]; ok {
			for _, result := range alloc.Devices.Results {
				device, err := getDeviceFromManager(pl.draManager, result.Pool, result.Device)
				if err != nil {
					logger.Error(err, "Failed to get device from manager", "claim", klog.KObj(claim), "device", result.Device, "pool", result.Pool)
					continue
				}
				if device != nil && len(device.NodeAllocatableResourceMappings) > 0 {
					if !nodeAllocatableClaimUIDs.Has(claim.UID) {
						nodeAllocatableClaims = append(nodeAllocatableClaims, claim)
						nodeAllocatableClaimUIDs.Insert(claim.UID)
					}
					break
				}
			}
		}
	}

	if len(nodeAllocatableClaims) == 0 {
		return nil, nil // No nodeAllocatable resources to check
	}

	totalPodDemand, nodeAllocatableClaimStatus, status := pl.getPodNodeAllocatableResourceFootprint(logger, nodeInfo, pod, state, allocations, nodeAllocatableClaims)
	if status != nil {
		logger.V(5).Info("calculateAndCheckNodeAllocatableResources: getPodNodeAllocatableResourceFootprint failed", "status", status)
		return nil, status
	}

	if status := pl.nodeFitsResources(nodeInfo, totalPodDemand); status != nil {
		return nil, status
	}
	logger.V(5).Info("Pod fits on node ( including DRA Node Allocatable Resources)", "pod", klog.KObj(pod), "node", nodeInfo.Node().Name)
	return nodeAllocatableClaimStatus, nil
}

// getDeviceFromManager retrieves a specific Device object from the DRA manager's cache and
// looks for the device matching the given poolName and deviceName.
func getDeviceFromManager(draManager fwk.SharedDRAManager, poolName, deviceName string) (*resourceapi.Device, error) {
	slices, err := draManager.ResourceSlices().ListWithDeviceTaintRules()
	if err != nil {
		return nil, fmt.Errorf("listing resource slices: %w", err)
	}
	for _, slice := range slices {
		if slice.Spec.Pool.Name == poolName {
			for i := range slice.Spec.Devices {
				if slice.Spec.Devices[i].Name == deviceName {
					return &slice.Spec.Devices[i], nil
				}
			}
		}
	}
	return nil, fmt.Errorf("device %s not found in pool %s", deviceName, poolName)
}

// buildNodeAllocatableDRAInfo processes the node allocatable resource allocations for a pod.
// It translates the allocated devices and quantities from DRA claims into a list of v1.NodeAllocatableResourceClaimStatus.
func (pl *DynamicResources) buildNodeAllocatableDRAInfo(pod *v1.Pod, nodeAllocatableClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult, claimNametoUID map[string]types.UID) ([]v1.NodeAllocatableResourceClaimStatus, error) {
	allContainers := make([]v1.Container, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	allContainers = append(allContainers, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	claimToStatus := make(map[types.UID]v1.NodeAllocatableResourceClaimStatus)

	for key, alloc := range nodeAllocatableClaimAllocations {
		currentClaimStatus := v1.NodeAllocatableResourceClaimStatus{
			ResourceClaimName: key.Name,
			Containers:        []string{},
			Resources:         map[v1.ResourceName]resource.Quantity{},
		}

		hasNodeAllocatableClaims := false
		for _, result := range alloc.Devices.Results {
			device, err := getDeviceFromManager(pl.draManager, result.Pool, result.Device)
			if err != nil {
				return nil, fmt.Errorf("claim %s/%s, device %s: %w", key.Namespace, key.Name, result.Device, err)
			}
			if device == nil || device.NodeAllocatableResourceMappings == nil {
				continue
			}

			for resourceName, resourceMap := range device.NodeAllocatableResourceMappings {
				quantity := resource.Quantity{}

				if resourceMap.CapacityKey != nil && *resourceMap.CapacityKey != "" {
					capacityKey := *resourceMap.CapacityKey
					if result.ConsumedCapacity == nil {
						return nil, fmt.Errorf("claim %s/%s, device %s: ConsumedCapacity is nil, but Capacity key '%s' is set in NodeAllocatableResourceMappings for resource %s", key.Namespace, key.Name, result.Device, capacityKey, resourceName)
					}
					if consumed, exists := result.ConsumedCapacity[capacityKey]; exists {
						quantity = consumed.DeepCopy()
						if resourceMap.AllocationMultiplier != nil {
							qDec := quantity.AsDec()
							multiplier := resourceMap.AllocationMultiplier.DeepCopy()
							qDec.Mul(qDec, multiplier.AsDec())
							quantity = *resource.NewDecimalQuantity(*qDec, quantity.Format)
						}
					} else {
						// If the capacityKey is not in ConsumedCapacity, this mapping is not relevant for this allocation
						continue
					}
				} else if resourceMap.AllocationMultiplier != nil {
					quantity = resourceMap.AllocationMultiplier.DeepCopy()
				}

				if currentClaimStatus.Resources == nil {
					currentClaimStatus.Resources = make(map[v1.ResourceName]resource.Quantity)
				}
				curQuantity, ok := currentClaimStatus.Resources[resourceName]
				if !ok {
					currentClaimStatus.Resources[resourceName] = quantity
				} else {
					curQuantity.Add(quantity)
					currentClaimStatus.Resources[resourceName] = curQuantity
				}

				hasNodeAllocatableClaims = true

			}
		}

		if hasNodeAllocatableClaims {
			claimToStatus[key.UID] = currentClaimStatus
		}
	}

	for _, container := range allContainers {
		for _, podClaim := range container.Resources.Claims {
			if claimUID, ok := claimNametoUID[podClaim.Name]; ok {
				if nodeAllocatableClaimStatus, ok := claimToStatus[claimUID]; ok {
					nodeAllocatableClaimStatus.Containers = append(nodeAllocatableClaimStatus.Containers, container.Name)
					claimToStatus[claimUID] = nodeAllocatableClaimStatus
				}
			}
		}
	}

	nodeAllocatableClaimInfoList := make([]v1.NodeAllocatableResourceClaimStatus, 0, len(claimToStatus))
	for _, status := range claimToStatus {
		nodeAllocatableClaimInfoList = append(nodeAllocatableClaimInfoList, status)
	}

	// Sort the results for consistent output.
	sort.Slice(nodeAllocatableClaimInfoList, func(i, j int) bool {
		return nodeAllocatableClaimInfoList[i].ResourceClaimName < nodeAllocatableClaimInfoList[j].ResourceClaimName
	})

	return nodeAllocatableClaimInfoList, nil
}

// validateNodeAllocatableDRAClaimSharing ensures that a node-allocatable DRA claim is not already in use by another pod on this node.
func (pl *DynamicResources) validateNodeAllocatableDRAClaimSharing(pod *v1.Pod, nodeInfo fwk.NodeInfo, claim *resourceapi.ResourceClaim) *fwk.Status {
	if claim == nil {
		return nil
	}
	claimKey := types.NamespacedName{Namespace: pod.Namespace, Name: claim.Name}
	claimStates := nodeInfo.GetNodeAllocatableDRAClaimState()
	if state, ok := claimStates[claimKey]; ok && state != nil {
		if state.ConsumerPods.Len() > 1 {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node allocatable resource claim %s shared by multiple pods", claimKey.Name))
		}
		if state.ConsumerPods.Len() == 1 && !state.ConsumerPods.Has(pod.UID) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node allocatable resource claim %s is already used by another pod", claimKey.Name))
		}
	}
	return nil
}

// validatePodLevelRequestsCoverDRA checks if the pod-level requests, if specified, are sufficient to cover
// the container level and DRA claim requests.
func (pl *DynamicResources) validatePodLevelRequestsCoverDRA(logger klog.Logger, pod *v1.Pod, requestWithPodLevel v1.ResourceList) *fwk.Status {
	if !pl.fts.EnablePodLevelResources || pod.Spec.Resources == nil || pod.Spec.Resources.Requests == nil {
		return nil
	}

	// Calculate Sum(Containers) + DRA + overhead resources.. Skip pod level resources for this sum
	optsSum := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:                    true,
		UseDRANodeAllocatableResourceClaimStatus: true,
	}
	requestWithoutPodLevel := resourcehelper.PodRequests(pod, optsSum)

	// For resources specified at pod level, check if container and DRA aggregates does not exceed pod level budget.
	for resName, podLevelReq := range requestWithPodLevel {
		val, ok := requestWithoutPodLevel[resName]
		if !ok {
			continue
		}
		if val.Cmp(podLevelReq) > 0 {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod level request for %s is insufficient to cover the aggregated container and node-allocatable DRA requests", resName))
		}
	}
	return nil
}

// getPodNodeAllocatableResourceFootprint determines the total nodeAllocatable resource demand of a pod.
func (pl *DynamicResources) getPodNodeAllocatableResourceFootprint(logger klog.Logger, nodeInfo fwk.NodeInfo, pod *v1.Pod, state *stateData, allocations map[types.UID]*resourceapi.AllocationResult, nodeAllocatableClaims []*resourceapi.ResourceClaim) (*framework.Resource, []v1.NodeAllocatableResourceClaimStatus, *fwk.Status) {
	nodeAllocatableDRAAllocations := make(map[v1.ObjectReference]*resourceapi.AllocationResult)
	// Add pre-allocated claims
	for _, claim := range nodeAllocatableClaims {
		key := v1.ObjectReference{
			Namespace: pod.Namespace,
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

	nodeAllocatableStatus, err := pl.buildNodeAllocatableDRAInfo(pod, nodeAllocatableDRAAllocations, claimNametoUID)
	if err != nil {
		return nil, nil, statusError(logger, err)
	}

	for _, status := range nodeAllocatableStatus {
		// TODO(KEP-5517): Evaluate if its ok to have no containers referencing a node allocatable resource claim.
		// This is pending on defining kubelet cgroup enforcement.
		if len(status.Containers) == 0 {
			return nil, nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("claim %s: node-allocatable resource claim not referenced by any container within the pod", status.ResourceClaimName))
		}
	}

	// Calculate the final totalPodDemand to be used for node fitting
	optsTotal := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:                    !pl.fts.EnablePodLevelResources,
		UseDRANodeAllocatableResourceClaimStatus: true,
	}
	podCopy := pod.DeepCopy()
	podCopy.Status.NodeAllocatableResourceClaimStatuses = nodeAllocatableStatus
	totalPodDemandRes := resourcehelper.PodRequests(podCopy, optsTotal)

	// Validate that pod-level requests, if specified, cover the aggregated container + DRA requests.
	// The API validation in pkg/apis/core/validation/validation.go only checks pod.Spec.Resources against container
	// requests within the Spec. It cannot account for DRA-derived resources, which are determined after device allocation.
	if status := pl.validatePodLevelRequestsCoverDRA(logger, podCopy, totalPodDemandRes); status != nil {
		return nil, nil, status
	}

	totalPodDemand := framework.NewResource(totalPodDemandRes)
	logger.V(5).Info("Total Pod Demand After DRA", "pod", klog.KObj(pod), "demand", totalPodDemand)

	return totalPodDemand, nodeAllocatableStatus, nil
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

func (pl *DynamicResources) patchNodeAllocatableResourceClaimStatus(ctx context.Context, pod *v1.Pod, nodeAllocatableClaimStatus []v1.NodeAllocatableResourceClaimStatus) *fwk.Status {

	if len(nodeAllocatableClaimStatus) == 0 {
		return nil
	}
	logger := klog.FromContext(ctx)

	// The incoming 'pod' is from the scheduler cache and would have NodeAllocatableResourceClaimStatus
	// pre-populated in the assume phase without persisting to the API server.
	// schedutil.PatchPodStatus skips patching if the old and new status are identical.
	// To ensure the status is persisted to the API server we clear it in the baseStatus, forcing a patch.
	baseStatus := pod.Status.DeepCopy()
	if !apiequality.Semantic.DeepEqual(baseStatus.NodeAllocatableResourceClaimStatuses, nodeAllocatableClaimStatus) {
		logger.V(5).Info("NodeAllocatableResourceClaimStatuses difference: assumed pod status does not match calculated status", "pod", klog.KObj(pod))
		return statusError(logger, errors.New("assumed pod status does not match calculated status to be patched"))
	}
	baseStatus.NodeAllocatableResourceClaimStatuses = nil

	targetStatus := pod.Status.DeepCopy()

	targetStatus.NodeAllocatableResourceClaimStatuses = nodeAllocatableClaimStatus
	if err := schedutil.PatchPodStatus(ctx, pl.clientset, pod.Name, pod.Namespace, baseStatus, targetStatus); err != nil {
		return statusError(logger, fmt.Errorf("updating pod %s/%s NodeAllocatableResourceClaimStatuses: %w", pod.Namespace, pod.Name, err))
	}
	logger.V(5).Info("Patched pod status with NodeAllocatableResourceClaimStatuses", "pod", klog.KObj(pod), "status", targetStatus.NodeAllocatableResourceClaimStatuses)

	return nil
}

func (pl *DynamicResources) clearNodeAllocatableResourceClaimStatus(ctx context.Context, pod *v1.Pod) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Clearing NodeAllocatableResourceClaimStatuses on Unreserve", "pod", klog.KObj(pod))

	targetStatus := pod.Status.DeepCopy()
	targetStatus.NodeAllocatableResourceClaimStatuses = nil

	if err := schedutil.PatchPodStatus(ctx, pl.clientset, pod.Name, pod.Namespace, &pod.Status, targetStatus); err != nil {
		logger.Error(err, "Failed to clear NodeAllocatableResourceClaimStatuses on Unreserve", "pod", klog.KObj(pod))
	} else {
		logger.V(5).Info("Cleared NodeAllocatableResourceClaimStatuses", "pod", klog.KObj(pod))
	}
}
