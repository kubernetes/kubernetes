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
	Generation     int64                 `json:"generation"`
	Error          string                `json:"error,omitempty"`
}

// PodResizeCompletedMsg generates the pod resize completed event message.
func PodResizeCompletedMsg(allocatedPod *v1.Pod, generation int64) string {
	return podResizeMessage(allocatedPod, generation, "", "Pod resize completed")
}

// PodResizeStartedMsg generates the pod resize in progress event message.
func PodResizeStartedMsg(allocatedPod *v1.Pod, generation int64) string {
	return podResizeMessage(allocatedPod, generation, "", "Pod resize started")
}

// PodResizeErrorMsg generates the pod resize in progress error event message.
func PodResizeErrorMsg(allocatedPod *v1.Pod, generation int64, errorMsg string) string {
	return podResizeMessage(allocatedPod, generation, errorMsg, "Pod resize error")
}

// PodResizePendingMsg generates the pod resize pending event message.
func PodResizePendingMsg(pod *v1.Pod, reason, message string, generation int64) string {
	return podResizeMessage(pod, generation, message, fmt.Sprintf("Pod resize %s", reason))
}

func podResizeMessage(pod *v1.Pod, generation int64, errorMsg, messagePrefix string) string {
	resources, err := makeResourceSummaryFromSpec(pod, generation, errorMsg)
	if err != nil {
		return messagePrefix
	}
	return fmt.Sprintf("%s: %s", messagePrefix, resources)

}

// Returns the desired resources from the podspec.
func makeResourceSummaryFromSpec(pod *v1.Pod, generation int64, errorMessage string) (string, error) {
	specResources := &podResourceSummary{Generation: generation, Error: errorMessage}
	for container, containerType := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		allocation := containerAllocation{
			Name:      container.Name,
			Resources: container.Resources,
		}
		switch containerType {
		case podutil.InitContainers:
			specResources.InitContainers = append(specResources.InitContainers, allocation)
		case podutil.Containers:
			specResources.Containers = append(specResources.Containers, allocation)
		}
	}

	message, err := json.Marshal(specResources)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(pod))
		return "", err
	}
	return string(message), nil
}
