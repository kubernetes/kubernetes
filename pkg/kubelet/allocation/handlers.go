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
	"fmt"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// NewPodResizesAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod resize can be allocated by the kubelet.
func NewPodResizesAdmitHandler(containerManager cm.ContainerManager, containerRuntime kubecontainer.Runtime, allocationManager Manager, logger klog.Logger) lifecycle.PodAdmitHandler {
	return &podResizesAdmitHandler{
		containerManager:  containerManager,
		containerRuntime:  containerRuntime,
		allocationManager: allocationManager,
		logger:            logger,
	}
}

type podResizesAdmitHandler struct {
	containerManager  cm.ContainerManager
	containerRuntime  kubecontainer.Runtime
	allocationManager Manager
	logger            klog.Logger
}

func (h *podResizesAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if attrs.Operation != lifecycle.ResizeOperation {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	pod := attrs.Pod
	allocatedPod, _ := h.allocationManager.UpdatePodFromAllocation(pod)
	if resizable, msg, reason := IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		// If there is a pending resize but the resize is not allowed, always use the allocated resources.
		metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
	}
	if resizeNotAllowed, msg := disallowResizeForSwappableContainers(h.containerRuntime, pod, allocatedPod); resizeNotAllowed {
		// If this resize involve swap recalculation, set as infeasible, as IPPR with swap is not supported for beta.
		metrics.PodInfeasibleResizes.WithLabelValues("swap_limitation").Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
	}
	if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, allocatedPod.Spec.Resources) {
		if resizable, msg, reason := IsInPlacePodLevelResourcesVerticalScalingAllowed(pod); !resizable {
			// If there is a pending pod-level resources resize but the resize is not allowed, always use the allocated resources.
			metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
			return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
		}
	}

	if v1qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) &&
			h.containerManager.GetNodeConfig().CPUManagerPolicy == string(cpumanager.PolicyStatic) &&
			h.guaranteedPodResourceResizeRequired(pod, v1.ResourceCPU) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside CPU Manager policy \"%s\"", string(cpumanager.PolicyStatic))
			h.logger.V(3).Info(msg, "pod", format.Pod(pod))
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_cpu_manager_static_policy").Inc()
			return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveMemory) &&
			h.containerManager.GetNodeConfig().MemoryManagerPolicy == string(memorymanager.PolicyTypeStatic) &&
			h.guaranteedPodResourceResizeRequired(pod, v1.ResourceMemory) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside Memory Manager policy \"%s\"", string(memorymanager.PolicyTypeStatic))
			h.logger.V(3).Info(msg, "pod", format.Pod(pod))
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_memory_manager_static_policy").Inc()
			return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
		}
	}

	allocatable := h.containerManager.GetNodeAllocatableAbsolute()
	cpuAvailable := allocatable.Cpu().MilliValue()
	memAvailable := allocatable.Memory().Value()
	cpuRequests := resource.GetResourceRequest(pod, v1.ResourceCPU)
	memRequests := resource.GetResourceRequest(pod, v1.ResourceMemory)

	if cpuRequests > cpuAvailable || memRequests > memAvailable {
		var msg string
		if memRequests > memAvailable {
			msg = fmt.Sprintf("memory, requested: %d, capacity: %d", memRequests, memAvailable)
		} else {
			msg = fmt.Sprintf("cpu, requested: %d, capacity: %d", cpuRequests, cpuAvailable)
		}
		msg = "Node didn't have enough capacity: " + msg
		h.logger.V(3).Info(msg, "pod", klog.KObj(pod))
		metrics.PodInfeasibleResizes.WithLabelValues("insufficient_node_allocatable").Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
	}

	return lifecycle.PodAdmitResult{Admit: true}
}

func disallowResizeForSwappableContainers(runtime kubecontainer.Runtime, desiredPod, allocatedPod *v1.Pod) (bool, string) {
	if desiredPod == nil || allocatedPod == nil {
		return false, ""
	}
	restartableMemoryResizePolicy := func(resizePolicies []v1.ContainerResizePolicy) bool {
		for _, policy := range resizePolicies {
			if policy.ResourceName == v1.ResourceMemory {
				return policy.RestartPolicy == v1.RestartContainer
			}
		}
		return false
	}
	allocatedContainers := make(map[string]v1.Container)
	for _, container := range append(allocatedPod.Spec.Containers, allocatedPod.Spec.InitContainers...) {
		allocatedContainers[container.Name] = container
	}
	for _, desiredContainer := range append(desiredPod.Spec.Containers, desiredPod.Spec.InitContainers...) {
		allocatedContainer, ok := allocatedContainers[desiredContainer.Name]
		if !ok {
			continue
		}
		origMemRequest := desiredContainer.Resources.Requests[v1.ResourceMemory]
		newMemRequest := allocatedContainer.Resources.Requests[v1.ResourceMemory]
		if !origMemRequest.Equal(newMemRequest) && !restartableMemoryResizePolicy(allocatedContainer.ResizePolicy) {
			aSwapBehavior := runtime.GetContainerSwapBehavior(desiredPod, &desiredContainer)
			bSwapBehavior := runtime.GetContainerSwapBehavior(allocatedPod, &allocatedContainer)
			if aSwapBehavior != kubetypes.NoSwap || bSwapBehavior != kubetypes.NoSwap {
				return true, "In-place resize of containers with swap is not supported."
			}
		}
	}
	return false, ""
}

func (h *podResizesAdmitHandler) guaranteedPodResourceResizeRequired(pod *v1.Pod, resourceName v1.ResourceName) bool {
	for container, containerType := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if !IsResizableContainer(container, containerType) {
			continue
		}
		requestedResources := container.Resources
		allocatedresources, _ := h.allocationManager.GetContainerResourceAllocation(pod.UID, container.Name)
		// For Guaranteed pods, requests must equal limits, so checking requests is sufficient.
		if !requestedResources.Requests[resourceName].Equal(allocatedresources.Requests[resourceName]) {
			return true
		}
	}
	return false
}
