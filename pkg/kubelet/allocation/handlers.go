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

package allocation

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// NewPodResizesAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod resize can be allocated by the kubelet.
func NewPodResizesAdmitHandler(containerManager cm.ContainerManager, allocationManager Manager) lifecycle.PodAdmitHandler {
	return &podResizesAdmitHandler{
		containerManager:  containerManager,
		allocationManager: allocationManager,
	}
}

type podResizesAdmitHandler struct {
	containerManager  cm.ContainerManager
	allocationManager Manager
}

func (h *podResizesAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if attrs.Operation != lifecycle.ResizeOperation {
		return lifecycle.PodAdmitResult{Admit: true}
	}

	pod := attrs.Pod
	if v1qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) &&
			h.containerManager.GetNodeConfig().CPUManagerPolicy == string(cpumanager.PolicyStatic) &&
			h.guaranteedPodResourceResizeRequired(pod, v1.ResourceCPU) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside CPU Manager policy \"%s\"", string(cpumanager.PolicyStatic))
			klog.V(3).InfoS(msg, "pod", format.Pod(pod))
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_cpu_manager_static_policy").Inc()
			return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.MemoryManager) &&
			!utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveMemory) &&
			h.containerManager.GetNodeConfig().MemoryManagerPolicy == string(memorymanager.PolicyTypeStatic) &&
			h.guaranteedPodResourceResizeRequired(pod, v1.ResourceMemory) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside Memory Manager policy \"%s\"", string(memorymanager.PolicyTypeStatic))
			klog.V(3).InfoS(msg, "pod", format.Pod(pod))
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
		klog.V(3).InfoS(msg, "pod", klog.KObj(pod))
		metrics.PodInfeasibleResizes.WithLabelValues("insufficient_node_allocatable").Inc()
		return lifecycle.PodAdmitResult{Admit: false, Reason: v1.PodReasonInfeasible, Message: msg}
	}

	return lifecycle.PodAdmitResult{Admit: true}
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
