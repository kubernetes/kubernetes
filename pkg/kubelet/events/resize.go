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

package events

import (
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

type containerAllocation struct {
	Name      string                  `json:"name"`
	Resources v1.ResourceRequirements `json:"resources,omitempty"`
}

type podResourceSummary struct {
	// TODO: resources v1.ResourceRequirements, add pod-level resources here once resizing pod-level resources is supported
	InitContainers []containerAllocation `json:"initContainers,omitempty"`
	Containers     []containerAllocation `json:"containers,omitempty"`
}

// PodResizeCompletedMsg generates the pod resize completed event message.
func PodResizeCompletedMsg(pod *v1.Pod, observedGeneration int64) string {
	type podResourceSummaryWithGeneration struct {
		ObservedGeneration int64 `json:"observedGeneration"`
		*podResourceSummary
	}
	resources, _ := makeResourceSummary(pod)
	podResizeSource := podResourceSummaryWithGeneration{
		ObservedGeneration: observedGeneration,
		podResourceSummary: resources,
	}
	podResizeMsgDetailsJSON, err := json.Marshal(podResizeSource)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(pod))
		return "Pod resize completed"
	}
	podResizeCompletedMsg := fmt.Sprintf("Pod resize completed: %s", string(podResizeMsgDetailsJSON))
	return podResizeCompletedMsg
}

// PodResizeInProgressMsg generates the pod resize in progress event message.
func PodResizeInProgressMsg(allocatedPod *v1.Pod, observedGeneration int64) string {
	type podResizeDiff struct {
		ObservedGeneration int64               `json:"observedGeneration"`
		ActualResources    *podResourceSummary `json:"actual,omitempty"`
		AllocatedResources *podResourceSummary `json:"allocated,omitempty"`
	}

	allocated, actual := makeResourceSummary(allocatedPod)

	diff := &podResizeDiff{
		ObservedGeneration: observedGeneration,
		AllocatedResources: allocated,
		ActualResources:    actual,
	}

	podResizeMsgDetailsJSON, err := json.Marshal(diff)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(allocatedPod))
		return "Pod resize in progress"
	}
	podResizeInProgressMsg := fmt.Sprintf("Pod resize in progress: %s", string(podResizeMsgDetailsJSON))
	return podResizeInProgressMsg
}

// PodResizeInProgressErrorMsg generates the pod resize in progress error event message.
func PodResizeInProgressErrorMsg(allocatedPod *v1.Pod, observedGeneration int64, errorMsg string) string {
	type podResizeDiff struct {
		ObservedGeneration int64               `json:"observedGeneration"`
		ActualResources    *podResourceSummary `json:"actual,omitempty"`
		AllocatedResources *podResourceSummary `json:"allocated,omitempty"`
		Error              string              `json:"error"`
	}

	allocated, actual := makeResourceSummary(allocatedPod)
	diff := &podResizeDiff{
		ObservedGeneration: observedGeneration,
		AllocatedResources: allocated,
		ActualResources:    actual,
		Error:              errorMsg,
	}

	podResizeMsgDetailsJSON, err := json.Marshal(diff)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(allocatedPod))
		return "Pod resize in progress reported an error: " + errorMsg
	}
	podResizeInProgressMsg := fmt.Sprintf("Pod resize in progress reported an error: %s", string(podResizeMsgDetailsJSON))
	return podResizeInProgressMsg
}

// PodResizePendingMsg generates the pod resize pending event message.
func PodResizePendingMsg(pod, allocatedPod *v1.Pod, reason string, observedGeneration int64) string {
	type podResizeDiff struct {
		ObservedGeneration int64               `json:"observedGeneration"`
		AllocatedResources *podResourceSummary `json:"allocated,omitempty"`
		DesiredResources   *podResourceSummary `json:"desired,omitempty"`
	}

	allocated, _ := makeResourceSummary(allocatedPod)
	desired, _ := makeResourceSummary(pod)

	diff := &podResizeDiff{
		ObservedGeneration: observedGeneration,
		AllocatedResources: allocated,
		DesiredResources:   desired,
	}

	podResizeMsgDetailsJSON, err := json.Marshal(diff)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(pod))
		return fmt.Sprintf("Pod resize %s", reason)
	}
	podResizePendingMsg := fmt.Sprintf("Pod resize %s: %s", reason, string(podResizeMsgDetailsJSON))
	return podResizePendingMsg
}

// Returns the desired resources from the spec, and the actual resources from the container statuses.
func makeResourceSummary(pod *v1.Pod) (*podResourceSummary, *podResourceSummary) {
	desiredResources := &podResourceSummary{}
	actualResources := &podResourceSummary{}

	for container, containerType := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		allocation := containerAllocation{
			Name:      container.Name,
			Resources: container.Resources,
		}
		switch containerType {
		case podutil.InitContainers:
			desiredResources.InitContainers = append(desiredResources.InitContainers, allocation)
		case podutil.Containers:
			desiredResources.Containers = append(desiredResources.Containers, allocation)
		}
	}

	for _, container := range pod.Status.InitContainerStatuses {
		allocation := containerAllocation{
			Name: container.Name,
		}
		if container.Resources != nil {
			allocation.Resources = *container.Resources
		}
		actualResources.InitContainers = append(actualResources.InitContainers, allocation)
	}
	for _, container := range pod.Status.ContainerStatuses {
		allocation := containerAllocation{
			Name: container.Name,
		}
		if container.Resources != nil {
			allocation.Resources = *container.Resources
		}
		actualResources.Containers = append(actualResources.Containers, allocation)
	}

	return desiredResources, actualResources
}
