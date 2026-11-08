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

package allocation

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// NewPodResizesAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod resize can be allocated by the kubelet.
func NewPodResizesAdmitHandler(containerManager cm.ContainerManager, resourceManager cm.ResourceManager, containerRuntime kubecontainer.Runtime, allocationManager Manager) lifecycle.PodAdmitHandler {
	return &podResizesAdmitHandler{
		containerManager:  containerManager,
		resourceManager:   resourceManager,
		containerRuntime:  containerRuntime,
		allocationManager: allocationManager,
	}
}

type podResizesAdmitHandler struct {
	containerManager  cm.ContainerManager
	resourceManager   cm.ResourceManager
	containerRuntime  kubecontainer.Runtime
	allocationManager Manager
}

// Admit counts infeasible resizes in metrics, so it must be registered before any other admit
// handler, which ensures it never runs in dry run.
func (h *podResizesAdmitHandler) Admit(ctx context.Context, attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if attrs.Operation != lifecycle.UpdateOperation {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	logger := klog.FromContext(ctx)
	pod := attrs.Pod
	allocatedPod := attrs.Update.AllocatedPod
	// This also covers containers without resources, which aren't resizes.
	if disallowed, msg := h.disallowContainerChangesForPodLevelExclusiveResources(pod, allocatedPod); disallowed {
		logger.V(3).Info(msg, "pod", format.Pod(pod))
		if attrs.Update.Resize {
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_level_resources_exclusive_allocation").Inc()
		}
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
	}
	if !attrs.Update.Resize {
		return lifecycle.PodAdmitResult{Admit: true}
	}
	if resizable, msg, reason := IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		// If there is a pending resize but the resize is not allowed, always use the allocated resources.
		metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
	}
	if resizeNotAllowed, msg := disallowResizeForSwappableContainers(h.containerRuntime, pod, allocatedPod); resizeNotAllowed {
		// If this resize involve swap recalculation, set as infeasible, as IPPR with swap is not supported for beta.
		metrics.PodInfeasibleResizes.WithLabelValues("swap_limitation").Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
	}
	if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, allocatedPod.Spec.Resources) {
		if resizable, msg, reason := IsInPlacePodLevelResourcesVerticalScalingAllowed(pod); !resizable {
			// If there is a pending pod-level resources resize but the resize is not allowed, always use the allocated resources.
			metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
			return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
		}
	}

	if disallowed, msg, reason := disallowResizeForMemoryBackedVolumes(pod, allocatedPod, h.containerManager.GetNodeConfig().CgroupVersion); disallowed {
		logger.V(3).Info(msg, "pod", format.Pod(pod))
		metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
	}

	if v1qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) {
			cpuMode := h.resourceManager.AllocationMode(v1.ResourceCPU)
			if cpuMode == cm.ResourceAllocationModeExclusive && h.guaranteedPodResourceResizeRequired(pod, allocatedPod, v1.ResourceCPU) {
				msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods when CPU allocation mode is %s", cpuMode)
				logger.V(3).Info(msg, "pod", format.Pod(pod))
				metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_cpu_manager_static_policy").Inc()
				return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
			}
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveMemory) {
			memMode := h.resourceManager.AllocationMode(v1.ResourceMemory)
			if memMode == cm.ResourceAllocationModeExclusive && h.guaranteedPodResourceResizeRequired(pod, allocatedPod, v1.ResourceMemory) {
				msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods when memory allocation mode is %s", memMode)
				logger.V(3).Info(msg, "pod", format.Pod(pod))
				metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_memory_manager_static_policy").Inc()
				return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg, RejectionType: lifecycle.RejectionInfeasible}
			}
		}
	}

	return lifecycle.PodAdmitResult{Admit: true}
}

func disallowResizeForMemoryBackedVolumes(desiredPod, allocatedPod *v1.Pod, cgroupVersion int) (bool, string, string) {
	if desiredPod == nil || allocatedPod == nil {
		return false, "", ""
	}

	if IsMemoryBackedVolumeResizeRequested(allocatedPod, desiredPod) {
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
			return true, "Memory-backed emptyDir volume resize is disabled", "memory_volume_resize_gate_off"
		}
		if cgroupVersion == 1 {
			return true, "Memory-backed emptyDir volume resize is not supported on cgroups v1", "cgroups_v1_unsupported"
		}
	}

	return false, "", ""
}

func podSwapLimitChanged(allocatedPod, desiredPod *v1.Pod) bool {
	allocatedSwap, allocatedHasSwap := cm.PodSwapLimit(allocatedPod)
	desiredSwap, desiredHasSwap := cm.PodSwapLimit(desiredPod)
	return allocatedHasSwap != desiredHasSwap || (allocatedHasSwap && allocatedSwap != desiredSwap)
}

func containerSwapLimitChanged(allocatedContainer, desiredContainer *v1.Container) bool {
	allocatedSwap, allocatedHasSwap := allocatedContainer.Resources.Limits[v1.ResourceSwap]
	desiredSwap, desiredHasSwap := desiredContainer.Resources.Limits[v1.ResourceSwap]
	return allocatedHasSwap != desiredHasSwap || (allocatedHasSwap && !allocatedSwap.Equal(desiredSwap))
}

func disallowResizeForSwappableContainers(runtime kubecontainer.Runtime, desiredPod, allocatedPod *v1.Pod) (bool, string) {
	if desiredPod == nil || allocatedPod == nil {
		return false, ""
	}
	workloadControlledSwap := utilfeature.DefaultFeatureGate.Enabled(features.WorkloadControlledSwap)
	restartableMemoryResizePolicy := func(resizePolicies []v1.ContainerResizePolicy) bool {
		for _, policy := range resizePolicies {
			if policy.ResourceName == v1.ResourceMemory {
				return policy.RestartPolicy == v1.RestartContainer
			}
		}
		return false
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources) && podSwapLimitChanged(allocatedPod, desiredPod) {
		if !workloadControlledSwap || runtime.GetContainerSwapBehavior(desiredPod, nil) == kubetypes.NoSwap {
			return true, "In-place resize of pod swap limit is not supported when swap is disabled on the node."
		}
	}
	const dynamicContainerSwapMsg = "Adding or removing containers with swap is not supported."
	for pair, containerType := range podutil.ContainerPairs(desiredPod, allocatedPod, podutil.InitContainers|podutil.Containers) {
		desiredContainer, allocatedContainer := pair.New, pair.Old
		switch {
		case allocatedContainer == nil:
			if runtime.GetContainerSwapBehavior(desiredPod, desiredContainer) != kubetypes.NoSwap {
				return true, dynamicContainerSwapMsg
			}
			continue
		case desiredContainer == nil:
			if runtime.GetContainerSwapBehavior(allocatedPod, allocatedContainer) != kubetypes.NoSwap {
				return true, dynamicContainerSwapMsg
			}
			continue
		case !IsResizableContainer(desiredContainer, containerType):
			continue
		}
		if containerSwapLimitChanged(allocatedContainer, desiredContainer) {
			if !workloadControlledSwap || (runtime.GetContainerSwapBehavior(desiredPod, desiredContainer) == kubetypes.NoSwap && runtime.GetContainerSwapBehavior(allocatedPod, allocatedContainer) == kubetypes.NoSwap) {
				return true, fmt.Sprintf("In-place resize of swap limit for container %q is not supported when swap is disabled on the node.", desiredContainer.Name)
			}
		}
		memRequestChanged := !desiredContainer.Resources.Requests.Memory().Equal(*allocatedContainer.Resources.Requests.Memory())
		memLimitChanged := workloadControlledSwap && !desiredContainer.Resources.Limits.Memory().Equal(*allocatedContainer.Resources.Limits.Memory())
		if (!memRequestChanged && !memLimitChanged) || restartableMemoryResizePolicy(allocatedContainer.ResizePolicy) {
			continue
		}
		desiredSwapBehavior := runtime.GetContainerSwapBehavior(desiredPod, desiredContainer)
		allocatedSwapBehavior := runtime.GetContainerSwapBehavior(allocatedPod, allocatedContainer)

		if !workloadControlledSwap {
			// Pre-KEP-5359: LimitedSwap derives memory.swap.max from requests.memory,
			// and the resize path could not see or validate that recalculation.
			if desiredSwapBehavior != kubetypes.NoSwap || allocatedSwapBehavior != kubetypes.NoSwap {
				return true, "In-place resize of containers with swap is not supported."
			}
			continue
		}

		// KEP-5359: the swap limit is a first-class resize resource. Numeric
		// increases and decreases of the effective swap limit (whether via explicit
		// limits.swap or proportional LimitedSwap recalculation) are admitted here
		// and validated at actuation against live memory.swap.current in
		// validateMemoryResizeAction, matching limits.memory resize behavior.
		//
		// However, when a resize on a LimitedSwap node makes requests.memory ==
		// limits.memory, GetContainerSwapBehavior transitions from LimitedSwap to
		// NoSwap (disabling LimitedSwap mode altogether). Reject that mode
		// transition as Infeasible unless explicit limits.swap is configured.
		if cm.HasExplicitSwapLimit(desiredPod, desiredContainer) || cm.HasExplicitSwapLimit(allocatedPod, allocatedContainer) {
			continue
		}
		if allocatedSwapBehavior == kubetypes.LimitedSwap && desiredSwapBehavior == kubetypes.NoSwap {
			return true, fmt.Sprintf("In-place resize of container %q would disable its LimitedSwap allocation (memory request equals limit); set resources.limits.swap explicitly or use a RestartContainer resize policy for memory.", desiredContainer.Name)
		}
	}
	return false, ""
}

// disallowContainerChangesForPodLevelExclusiveResources returns whether containers are added to or
// removed from a Guaranteed pod whose pod-level resources the CPU or memory manager allocates
// exclusively. The managers only partition such an allocation among the containers when the pod is
// admitted, so a container added later would run outside of it.
// TODO(KEP-5972): Allow this once the static policies can repartition a pod-level allocation.
func (h *podResizesAdmitHandler) disallowContainerChangesForPodLevelExclusiveResources(pod, allocatedPod *v1.Pod) (bool, string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResourceManagers) || !resourcehelper.IsPodLevelResourcesSet(pod) || v1qos.GetPodQOS(pod) != v1.PodQOSGuaranteed {
		return false, ""
	}
	changed := false
	for pair := range podutil.ContainerPairs(pod, allocatedPod, podutil.InitContainers|podutil.Containers) {
		if pair.New == nil || pair.Old == nil {
			changed = true
			break
		}
	}
	if !changed {
		return false, ""
	}
	for _, resourceName := range []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory} {
		if mode := h.resourceManager.AllocationMode(resourceName); mode == cm.ResourceAllocationModeExclusive {
			return true, fmt.Sprintf("Adding or removing containers is infeasible for Guaranteed Pods with pod-level resources when %s allocation mode is %s", resourceName, mode)
		}
	}
	return false, ""
}

func (h *podResizesAdmitHandler) guaranteedPodResourceResizeRequired(pod, allocatedPod *v1.Pod, resourceName v1.ResourceName) bool {
	// Adding or removing a container that requests the resource changes the allocation like a
	// resize does.
	for pair, containerType := range podutil.ContainerPairs(pod, allocatedPod, podutil.InitContainers|podutil.Containers) {
		container := pair.New
		if container == nil {
			container = pair.Old
		}
		if !IsResizableContainer(container, containerType) {
			continue
		}
		// For Guaranteed pods, requests must equal limits, so checking requests is sufficient.
		var desiredRequests, allocatedRequests v1.ResourceList
		if pair.New != nil {
			desiredRequests = pair.New.Resources.Requests
		}
		if pair.Old != nil {
			allocatedRequests = pair.Old.Resources.Requests
		}
		if desired, allocated := desiredRequests[resourceName], allocatedRequests[resourceName]; !desired.Equal(allocated) {
			return true
		}
	}
	return false
}

// FindVolume returns the volume with the given name from the slice, checking idxHint first for 1:1 common-case alignment.
func FindVolume(volumes []v1.Volume, name string, idxHint int) *v1.Volume {
	if len(volumes) > idxHint && volumes[idxHint].Name == name {
		return &volumes[idxHint]
	}
	for i := range volumes {
		if volumes[i].Name == name {
			return &volumes[i]
		}
	}
	return nil
}

// IsMemoryBackedVolumeResizeRequested returns true if any of the pod's memory-backed emptyDir volumes has its size limit changed.
func IsMemoryBackedVolumeResizeRequested(desiredPod, allocatedPod *v1.Pod) bool {
	for i, vol := range desiredPod.Spec.Volumes {
		if !VolHasMemoryBackedEmptyDir(&vol) {
			continue
		}
		allocVol := FindVolume(allocatedPod.Spec.Volumes, vol.Name, i)
		if allocVol == nil || !VolHasMemoryBackedEmptyDir(allocVol) {
			continue
		}
		hasLimit, hasAllocLimit := VolHasMemoryBackedEmptyDirSizeLimit(&vol), VolHasMemoryBackedEmptyDirSizeLimit(allocVol)
		if hasLimit != hasAllocLimit || (hasLimit && !vol.EmptyDir.SizeLimit.Equal(*allocVol.EmptyDir.SizeLimit)) {
			return true
		}
	}
	return false
}
