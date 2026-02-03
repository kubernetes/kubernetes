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
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

func (pl *DynamicResources) deviceClassManagesNativeResource(logger klog.Logger, deviceClassName, requestName string) (bool, *fwk.Status) {
	if !pl.fts.EnableDRANativeResources {
		return false, nil
	}
	if deviceClassName == "" {
		return false, nil
	}

	deviceClass, err := pl.draManager.DeviceClasses().Get(deviceClassName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return false, statusUnschedulable(logger, fmt.Sprintf("request %s: device class %s does not exist", requestName, deviceClassName))
		}

		return false, statusError(logger, err)
	}
	return deviceClass.Spec.ManagesNativeResources, nil
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

func (pl *DynamicResources) checkNativeResources(ctx context.Context, state *stateData, pod *v1.Pod, nodeInfo fwk.NodeInfo, allocations map[string]*resourceapi.AllocationResult) ([]v1.PodNativeResourceClaimStatus, *fwk.Status) {
	logger := klog.FromContext(ctx)

	if !pl.fts.EnableDRANativeResources || !state.hasNativeResources {
		return nil, nil
	}

	totalPodDemand, nativeClaimStatus, status := pl.getPodNativeResourceFootprint(logger, nodeInfo, pod, state, allocations)
	if status != nil {
		return nil, status
	}

	if status := pl.nodeFitsNativeResource(nodeInfo, logger, state, totalPodDemand); status != nil {
		return nil, status
	}
	logger.V(5).Info("Pod fits on node (DRA Native Resources)", "pod", klog.KObj(pod), "node", nodeInfo.Node().Name)
	return nativeClaimStatus, nil
}

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
	return nil, nil // Not found
}

func (pl *DynamicResources) buildNativeDRAInfo(pod *v1.Pod, claimByName map[string]*resourceapi.ResourceClaim, finalAllocs map[types.UID]*resourceapi.AllocationResult, state *stateData) ([]v1.PodNativeResourceClaimStatus, error) {
	allContainers := append(pod.Spec.InitContainers, pod.Spec.Containers...)
	nativeClaimInfo := make(map[types.UID]v1.PodNativeResourceClaimStatus)

	for _, container := range allContainers {
		for _, podClaim := range container.Resources.Claims {
			actualClaim := claimByName[podClaim.Name]
			if actualClaim == nil {
				continue
			}

			alloc := finalAllocs[actualClaim.UID]
			if alloc == nil {
				continue
			}

			// Initialize currentClaimStatus entry if missing
			currentClaimStatus, exists := nativeClaimInfo[actualClaim.UID]
			if !exists {
				currentClaimStatus = v1.PodNativeResourceClaimStatus{
					ClaimInfo: v1.ObjectReference{
						Namespace: pod.Namespace,
						Name:      actualClaim.Name,
						UID:       actualClaim.UID,
					},
					Containers: []string{},
					Resources:  []v1.NativeResourceAllocation{},
				}
			}
			currentClaimStatus.Containers = append(currentClaimStatus.Containers, container.Name)
			if exists {
				nativeClaimInfo[actualClaim.UID] = currentClaimStatus
				// This claim is already processed
				continue
			}

			hasNativeClaims := false

			for _, result := range alloc.Devices.Results {
				device, err := getDeviceFromManager(pl.draManager, result.Pool, result.Device)
				if err != nil {
					return nil, err
				}
				if device == nil || device.NativeResourceMappings == nil {
					continue
				}

				for resourceName, natResMap := range device.NativeResourceMappings {
					quantity := resource.Quantity{}
					perInstance := natResMap.QuantityFrom.PerInstanceQuantity
					capacityKey := natResMap.QuantityFrom.Capacity

					if capacityKey != "" {
						found := false
						if result.ConsumedCapacity != nil {
							if consumed, exists := result.ConsumedCapacity[capacityKey]; exists {
								quantity.Add(consumed)
								found = true
							}
						}

						if !found && device.Capacity != nil {
							if devQuant, exists := device.Capacity[capacityKey]; exists {
								quantity.Add(devQuant.Value)
							}
						}

						if perInstance != nil {
							qDec := quantity.AsDec()
							pDec := perInstance.AsDec()
							qDec.Mul(qDec, pDec)
							quantity = *resource.NewDecimalQuantity(*qDec, quantity.Format)
						}
					} else if perInstance != nil {
						quantity = *perInstance
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
				nativeClaimInfo[actualClaim.UID] = currentClaimStatus
			}
		}
	}

	nativeClaimInfoList := make([]v1.PodNativeResourceClaimStatus, 0, len(nativeClaimInfo))
	for _, status := range nativeClaimInfo {
		nativeClaimInfoList = append(nativeClaimInfoList, status)
	}
	return nativeClaimInfoList, nil
}

func (pl *DynamicResources) validateNativeDRAClaims(pod *v1.Pod, nodeInfo fwk.NodeInfo, nativeResourceClaimStatus []v1.PodNativeResourceClaimStatus) error {
	if len(nativeResourceClaimStatus) == 0 {
		return nil
	}
	if pod.Spec.Resources != nil {
		return fmt.Errorf("cannot use pod level resources with native resource claims")
	}
	for _, claim := range nativeResourceClaimStatus {
		claimStates := nodeInfo.GetNativeResourceDRAClaimStates()
		state, ok := claimStates[claim.ClaimInfo.UID]
		if ok && state != nil && state.ConsumerPods.Len() > 0 {
			if state.ConsumerPods.Has(pod.UID) {
				return fmt.Errorf("cannot share native resource claims across pods")
			}
		}
	}
	return nil
}

func (pl *DynamicResources) getPodNativeResourceFootprint(logger klog.Logger, nodeInfo fwk.NodeInfo, pod *v1.Pod, state *stateData, allocations map[string]*resourceapi.AllocationResult) (*framework.Resource, []v1.PodNativeResourceClaimStatus, *fwk.Status) {

	claimByName := make(map[string]*resourceapi.ResourceClaim)
	for _, claim := range state.claims.allUserClaims() {
		claimByName[claim.Name] = claim
	}

	finalAllocs := make(map[types.UID]*resourceapi.AllocationResult)
	// Add pre-allocated claims
	for _, claim := range state.claims.allUserClaims() {
		if claim.Status.Allocation != nil {
			finalAllocs[claim.UID] = claim.Status.Allocation
		}
	}
	// Add newly allocated claims
	for claimName, alloc := range allocations {
		if claim, ok := claimByName[claimName]; ok {
			if claim.Status.Allocation == nil {
				finalAllocs[claim.UID] = alloc
			}
		}
	}

	podNativeDRAStatus, err := pl.buildNativeDRAInfo(pod, claimByName, finalAllocs, state)
	if err != nil {
		return nil, nil, statusError(logger, err)
	}

	if err := pl.validateNativeDRAClaims(pod, nodeInfo, podNativeDRAStatus); err != nil {
		return nil, nil, statusError(logger, err)
	}

	// Calculate Effective Container Requests for PodRequests helper
	opts := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources:           true,
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

// InsufficientNativeResource describes what kind of resource limit is hit and caused the pod to not fit the node.
type InsufficientNativeResource struct {
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

func (pl *DynamicResources) nodeFitsNativeResource(nodeInfo fwk.NodeInfo, logger klog.Logger, state *stateData, podRequest *framework.Resource) *fwk.Status {
	insufficientResources := make([]InsufficientNativeResource, 0, 4)

	if podRequest.MilliCPU > 0 && podRequest.MilliCPU > (nodeInfo.GetAllocatable().GetMilliCPU()-nodeInfo.GetRequested().GetMilliCPU()) {
		insufficientResources = append(insufficientResources, InsufficientNativeResource{
			ResourceName: v1.ResourceCPU,
			Reason:       "Insufficient cpu",
			Requested:    podRequest.MilliCPU,
			Used:         nodeInfo.GetRequested().GetMilliCPU(),
			Capacity:     nodeInfo.GetAllocatable().GetMilliCPU(),
			Unresolvable: podRequest.MilliCPU > nodeInfo.GetAllocatable().GetMilliCPU(),
		})
	}
	if podRequest.Memory > 0 && podRequest.Memory > (nodeInfo.GetAllocatable().GetMemory()-nodeInfo.GetRequested().GetMemory()) {
		insufficientResources = append(insufficientResources, InsufficientNativeResource{
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
		insufficientResources = append(insufficientResources, InsufficientNativeResource{
			ResourceName: v1.ResourceEphemeralStorage,
			Reason:       "Insufficient ephemeral-storage",
			Requested:    podRequest.EphemeralStorage,
			Used:         nodeInfo.GetRequested().GetEphemeralStorage(),
			Capacity:     nodeInfo.GetAllocatable().GetEphemeralStorage(),
			Unresolvable: podRequest.GetEphemeralStorage() > nodeInfo.GetAllocatable().GetEphemeralStorage(),
		})
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

	if !pl.fts.EnableDRANativeResources || !state.hasNativeResources {
		return nil
	}

	podStatusCopy := pod.Status.DeepCopy()
	// Sort existing status for comparison.
	if podStatusCopy.NativeResourceClaimStatus != nil {
		sort.Slice(podStatusCopy.NativeResourceClaimStatus, func(i, j int) bool {
			return podStatusCopy.NativeResourceClaimStatus[i].ClaimInfo.UID < podStatusCopy.NativeResourceClaimStatus[j].ClaimInfo.UID
		})
		for i := range podStatusCopy.NativeResourceClaimStatus {
			sort.Strings(podStatusCopy.NativeResourceClaimStatus[i].Containers)
			sort.Slice(podStatusCopy.NativeResourceClaimStatus[i].Resources, func(a, b int) bool {
				return podStatusCopy.NativeResourceClaimStatus[i].Resources[a].ResourceName < podStatusCopy.NativeResourceClaimStatus[i].Resources[b].ResourceName
			})
		}
	}

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
