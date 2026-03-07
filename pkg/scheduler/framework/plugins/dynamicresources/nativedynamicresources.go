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
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// deviceClassManagesNativeResource checks if the given device class is configured to manage native resources.
func (pl *DynamicResources) deviceClassManagesNativeResource(logger klog.Logger, deviceClassName string) (bool, *fwk.Status) {
	if !pl.fts.EnableDRANativeResources {
		return false, nil
	}
	if deviceClassName == "" {
		return false, nil
	}

	deviceClass, err := pl.draManager.DeviceClasses().Get(deviceClassName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, statusUnschedulable(logger, fmt.Sprintf("device class %s does not exist", deviceClassName))
		}

		return false, statusError(logger, err)
	}
	if deviceClass != nil && deviceClass.Spec.ManagesNativeResources != nil && *deviceClass.Spec.ManagesNativeResources {
		return true, nil
	}
	return false, nil
}

// ExtractPodNativeResourceClaimStatus returns the pod native claim status stored in state
func ExtractPodNativeResourceClaimStatus(logger klog.Logger, state fwk.CycleState, nodeName string) []v1.PodNativeResourceClaimStatus {
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
		return nodeAlloc.nativeResourceClaimStatus
	}

	return nil
}

// calculateAndCheckNativeResources calculates the total native resources required by the pod based on it DRA claims
// and standard resources. It aggregates resource demands from all containers
// and verifies against the node's allocatable resources.
func (pl *DynamicResources) calculateAndCheckNativeResources(ctx context.Context, state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocations map[types.UID]*resourceapi.AllocationResult) ([]v1.PodNativeResourceClaimStatus, *fwk.Status) {
	logger := klog.FromContext(ctx)

	totalPodDemand, nativeClaimStatus, status := pl.getPodNativeResourceFootprint(logger, nodeInfo, pod, state, allocations)
	if status != nil {
		return nil, status
	}

	if status := pl.nodeFitsNativeResources(nodeInfo, totalPodDemand); status != nil {
		return nil, status
	}
	logger.V(5).Info("Pod fits on node ( including DRA Native Resources)", "pod", klog.KObj(pod), "node", nodeInfo.Node().Name)
	return nativeClaimStatus, nil
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

// buildNativeDRAInfo processes the native resource allocations for a pod.
// It translates the allocated devices and quantities from DRA claims into a list of v1.PodNativeResourceClaimStatus.
func (pl *DynamicResources) buildNativeDRAInfo(pod *v1.Pod, nativeClaimAllocations map[v1.ObjectReference]*resourceapi.AllocationResult, claimNametoUID map[string]types.UID) ([]v1.PodNativeResourceClaimStatus, error) {
	allContainers := make([]v1.Container, 0, len(pod.Spec.InitContainers)+len(pod.Spec.Containers))
	allContainers = append(allContainers, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	nativeClaimInfo := make(map[types.UID]v1.PodNativeResourceClaimStatus)

	for key, alloc := range nativeClaimAllocations {
		currentClaimStatus := v1.PodNativeResourceClaimStatus{
			ClaimInfo:  key,
			Containers: []string{},
			Resources:  []v1.NativeResourceAllocation{},
		}

		hasNativeClaims := false
		for _, result := range alloc.Devices.Results {
			device, err := getDeviceFromManager(pl.draManager, result.Pool, result.Device)
			if err != nil {
				return nil, fmt.Errorf("claim %s/%s, device %s: %w", key.Namespace, key.Name, result.Device, err)
			}
			if device == nil || device.NativeResourceMappings == nil {
				continue
			}

			for resourceName, natResMap := range device.NativeResourceMappings {
				quantity := resource.Quantity{}

				if natResMap.CapacityKey != nil && *natResMap.CapacityKey != "" {
					capacityKey := *natResMap.CapacityKey
					if result.ConsumedCapacity == nil {
						return nil, fmt.Errorf("claim %s/%s, device %s: ConsumedCapacity is nil, but Capacity key '%s' is set in NativeResourceMappings for resource %s", key.Namespace, key.Name, result.Device, capacityKey, resourceName)
					}
					if consumed, exists := result.ConsumedCapacity[capacityKey]; exists {
						quantity = consumed.DeepCopy()
						if natResMap.PerAllocatedUnitQuantity != nil {
							qDec := quantity.AsDec()
							pDec := (*natResMap.PerAllocatedUnitQuantity).AsDec()
							qDec.Mul(qDec, pDec)
							quantity = *resource.NewDecimalQuantity(*qDec, quantity.Format)
						}
					} else {
						// If the capacityKey is not in ConsumedCapacity, this mapping is not relevant for this allocation
						continue
					}
				} else if natResMap.PerAllocatedUnitQuantity != nil {
					quantity = *natResMap.PerAllocatedUnitQuantity
				}

				merged := false
				for i := range currentClaimStatus.Resources {
					existing := &currentClaimStatus.Resources[i]
					if existing.ResourceName == resourceName {
						existing.Quantity.Add(quantity)
						merged = true
						break
					}
				}

				if !merged {
					currentClaimStatus.Resources = append(currentClaimStatus.Resources, v1.NativeResourceAllocation{
						ResourceName: resourceName,
						Quantity:     quantity,
					})
				}

				hasNativeClaims = true

			}
		}

		if hasNativeClaims {
			nativeClaimInfo[key.UID] = currentClaimStatus
		}
	}

	for _, container := range allContainers {
		for _, podClaim := range container.Resources.Claims {
			if claimUID, ok := claimNametoUID[podClaim.Name]; ok {
				if nativeClaimStatus, ok := nativeClaimInfo[claimUID]; ok {
					nativeClaimStatus.Containers = append(nativeClaimStatus.Containers, container.Name)
					nativeClaimInfo[claimUID] = nativeClaimStatus
				}
			}
		}
	}

	nativeClaimInfoList := make([]v1.PodNativeResourceClaimStatus, 0, len(nativeClaimInfo))
	for _, status := range nativeClaimInfo {
		nativeClaimInfoList = append(nativeClaimInfoList, status)
	}

	// Sort the results for consistent output.
	sort.Slice(nativeClaimInfoList, func(i, j int) bool {
		return nativeClaimInfoList[i].ClaimInfo.UID < nativeClaimInfoList[j].ClaimInfo.UID
	})
	for i := range nativeClaimInfoList {
		sort.Strings(nativeClaimInfoList[i].Containers)
		sort.Slice(nativeClaimInfoList[i].Resources, func(a, b int) bool {
			return nativeClaimInfoList[i].Resources[a].ResourceName < nativeClaimInfoList[i].Resources[b].ResourceName
		})
	}

	return nativeClaimInfoList, nil
}

func (pl *DynamicResources) validateNativeDRAClaims(pod *v1.Pod, nodeInfo fwk.NodeInfo, nativeResourceClaimStatus []v1.PodNativeResourceClaimStatus) error {
	if len(nativeResourceClaimStatus) == 0 {
		return nil
	}
	for _, claim := range nativeResourceClaimStatus {
		claimStates := nodeInfo.GetNativeResourceDRAClaimStates()
		if state, ok := claimStates[claim.ClaimInfo.UID]; ok && state != nil {
			if state.ConsumerPods.Len() > 1 {
				return fmt.Errorf("native resource claim %s shared by multiple pods", claim.ClaimInfo.Name)
			}
			if state.ConsumerPods.Len() == 1 && !state.ConsumerPods.Has(pod.UID) {
				return fmt.Errorf("native resource claim %s is already used by another pod", claim.ClaimInfo.Name)
			}
		}
	}
	return nil
}

// getPodNativeResourceFootprint determines the total native resource demand of a pod.
func (pl *DynamicResources) getPodNativeResourceFootprint(logger klog.Logger, nodeInfo fwk.NodeInfo, pod *v1.Pod, state *stateData, allocations map[types.UID]*resourceapi.AllocationResult) (*framework.Resource, []v1.PodNativeResourceClaimStatus, *fwk.Status) {
	nativeDRAAllocations := make(map[v1.ObjectReference]*resourceapi.AllocationResult)
	// Add pre-allocated claims
	for _, claim := range state.claims.nativeResourceClaims() {
		key := v1.ObjectReference{
			Namespace: pod.Namespace,
			Name:      claim.Name,
			UID:       claim.UID,
		}
		if claim.Status.Allocation != nil {
			nativeDRAAllocations[key] = claim.Status.Allocation
		}
		if alloc, ok := allocations[claim.UID]; ok {
			nativeDRAAllocations[key] = alloc
		}
	}

	claimNametoUID := make(map[string]types.UID)
	if err := pl.foreachPodResourceClaim(pod, func(name string, claim *resourceapi.ResourceClaim) {
		claimNametoUID[name] = claim.UID
	}); err != nil {
		return nil, nil, statusError(logger, fmt.Errorf("processing pod resource claims: %w", err))
	}

	podNativeDRAStatus, err := pl.buildNativeDRAInfo(pod, nativeDRAAllocations, claimNametoUID)
	if err != nil {
		return nil, nil, statusError(logger, err)
	}

	if err := pl.validateNativeDRAClaims(pod, nodeInfo, podNativeDRAStatus); err != nil {
		return nil, nil, fwk.NewStatus(fwk.Unschedulable, err.Error())
	}

	// Calculate Effective Container Requests for PodRequests helper
	opts := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:           !pl.fts.EnablePodLevelResources,
		UseStatusResources:              false,
		UseDRANativeResourceClaimStatus: true,
	}
	podCopy := pod.DeepCopy()
	podCopy.Status.NativeResourceClaimStatus = podNativeDRAStatus
	totalPodDemandRes := resourcehelper.PodRequests(podCopy, opts)

	totalPodDemand := framework.NewResource(totalPodDemandRes)
	logger.V(5).Info("Total Pod Demand After DRA", "pod", klog.KObj(pod), "demand", totalPodDemand)

	return totalPodDemand, podNativeDRAStatus, nil
}

// insufficientNativeResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type insufficientNativeResource struct {
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

// nodeFitsNativeResources checks if the node has sufficient capacity to accommodate the pod's native resource requirements.
func (pl *DynamicResources) nodeFitsNativeResources(nodeInfo fwk.NodeInfo, podRequest *framework.Resource) *fwk.Status {
	insufficientResources := make([]insufficientNativeResource, 0, 4)

	if podRequest.MilliCPU > 0 && podRequest.MilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-nodeInfo.GetRequested().GetMilliCPU()) {
		insufficientResources = append(insufficientResources, insufficientNativeResource{
			ResourceName: v1.ResourceCPU,
			Reason:       "Insufficient cpu",
			Requested:    podRequest.MilliCPU,
			Used:         nodeInfo.GetRequested().GetMilliCPU(),
			Capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			Unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && podRequest.Memory > (nodeInfo.GetAllocatable().GetMemory()-nodeInfo.GetRequested().GetMemory()) {
		insufficientResources = append(insufficientResources, insufficientNativeResource{
			ResourceName: v1.ResourceMemory,
			Reason:       "Insufficient memory",
			Requested:    podRequest.Memory,
			Used:         nodeInfo.GetRequested().GetMemory(),
			Capacity:     nodeInfo.GetAllocatable().GetMemory(),
			Unresolvable: podRequest.Memory > nodeInfo.GetAllocatable().GetMemory(),
		})
	}
	if podRequest.EphemeralStorage > 0 &&
		podRequest.EphemeralStorage > (nodeInfo.GetAllocatable().GetEphemeralStorage()-nodeInfo.GetRequested().GetEphemeralStorage()) {
		insufficientResources = append(insufficientResources, insufficientNativeResource{
			ResourceName: v1.ResourceEphemeralStorage,
			Reason:       "Insufficient ephemeral-storage",
			Requested:    podRequest.EphemeralStorage,
			Used:         nodeInfo.GetRequested().GetEphemeralStorage(),
			Capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			Unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
	}

	for resName, reqQuant := range podRequest.ScalarResources {
		if v1helper.IsHugePageResourceName(resName) {
			nodeCapacity := nodeInfo.GetAllocatable().GetScalarResources()[resName]
			nodeRequested := nodeInfo.GetRequested().GetScalarResources()[resName]
			available := nodeCapacity - nodeRequested

			if reqQuant > available {
				insufficientResources = append(insufficientResources, insufficientNativeResource{
					ResourceName: resName,
					Reason:       fmt.Sprintf("Insufficient %s", resName),
					Requested:    reqQuant,
					Used:         nodeRequested,
					Capacity:     nodeCapacity,
					Unresolvable: reqQuant > nodeCapacity,
				})
			}
		}
	}

	if len(insufficientResources) > 0 {
		var reasons []string
		for _, r := range insufficientResources {
			reasons = append(reasons, fmt.Sprintf("%s (requested: %d, used: %d, capacity: %d)", r.ResourceName, r.Requested, r.Used, r.Capacity))
		}
		msg := fmt.Sprintf("Insufficient resources: %s", strings.Join(reasons, "; "))
		return fwk.NewStatus(fwk.Unschedulable, msg)
	}
	return nil
}

func (pl *DynamicResources) patchNativeResourceClaimStatus(ctx context.Context, state *stateData, pod *v1.Pod, nodeName string) *fwk.Status {
	logger := klog.FromContext(ctx)

	if !pl.fts.EnableDRANativeResources || !state.hasNativeResourceClaims {
		return nil
	}

	podStatusCopy := pod.Status.DeepCopy()
	nativeClaimStatus := state.nodeAllocations[nodeName].nativeResourceClaimStatus
	if apiequality.Semantic.DeepEqual(podStatusCopy.NativeResourceClaimStatus, nativeClaimStatus) {
		logger.V(6).Info("NativeResourceClaimStatus is already up-to-date", "pod", klog.KObj(pod))
		return nil
	}

	podStatusCopy.NativeResourceClaimStatus = nativeClaimStatus

	if err := schedutil.PatchPodStatus(ctx, pl.clientset, pod.Name, pod.Namespace, &pod.Status, podStatusCopy); err != nil {
		return statusError(logger, fmt.Errorf("updating pod %s/%s NativeResourceClaimStatus: %w", pod.Namespace, pod.Name, err))
	}
	logger.V(5).Info("Patched pod status with NativeResourceClaimStatus", "pod", klog.KObj(pod), "status", podStatusCopy.NativeResourceClaimStatus)

	return nil
}
