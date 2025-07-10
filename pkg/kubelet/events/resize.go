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

// PodResizeCompletedMsg gernerates the pod resize completed event message.
func PodResizeCompletedMsg(allocatedPod *v1.Pod) string {
	type containerAllocation struct {
		Name      string                  `json:"name"`
		Resources v1.ResourceRequirements `json:"resources,omitempty"`
	}
	type podResourceSummary struct {
		//TODO: resources v1.ResourceRequirements, add pod-level resources here once resizing pod-level resources is supported
		InitContainers []containerAllocation `json:"initContainers,omitempty"`
		Containers     []containerAllocation `json:"containers,omitempty"`
	}

	podResizeSource := &podResourceSummary{}

	for container, containerType := range podutil.ContainerIter(&allocatedPod.Spec, podutil.InitContainers|podutil.Containers) {
		allocation := containerAllocation{
			Name:      container.Name,
			Resources: container.Resources,
		}
		switch containerType {
		case podutil.InitContainers:
			podResizeSource.InitContainers = append(podResizeSource.InitContainers, allocation)
		case podutil.Containers:
			podResizeSource.Containers = append(podResizeSource.Containers, allocation)
		}
	}

	podResizeMsgDetailsJSON, err := json.Marshal(podResizeSource)
	if err != nil {
		klog.ErrorS(err, "Failed to serialize resource summary", "pod", format.Pod(allocatedPod))
		return "Pod resize completed"
	}
	podResizeCompletedMsg := fmt.Sprintf("Pod resize completed: %s", string(podResizeMsgDetailsJSON))
	return podResizeCompletedMsg
}
