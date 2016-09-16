/*
Copyright 2014 The Kubernetes Authors.

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

package status

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
)

// GeneratePodReadyCondition returns ready condition if all containers in a pod are ready, else it
// returns an unready condition.
func GeneratePodReadyCondition(spec *api.PodSpec, containerStatuses []api.ContainerStatus, podPhase api.PodPhase) api.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil {
		return api.PodCondition{
			Type:   api.PodReady,
			Status: api.ConditionFalse,
			Reason: "UnknownContainerStatuses",
		}
	}
	unknownContainers := []string{}
	unreadyContainers := []string{}
	for _, container := range spec.Containers {
		if containerStatus, ok := api.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				unreadyContainers = append(unreadyContainers, container.Name)
			}
		} else {
			unknownContainers = append(unknownContainers, container.Name)
		}
	}

	// If all containers are known and succeeded, just return PodCompleted.
	if podPhase == api.PodSucceeded && len(unknownContainers) == 0 {
		return api.PodCondition{
			Type:   api.PodReady,
			Status: api.ConditionFalse,
			Reason: "PodCompleted",
		}
	}

	unreadyMessages := []string{}
	if len(unknownContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unknown status: %s", unknownContainers))
	}
	if len(unreadyContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unready status: %s", unreadyContainers))
	}
	unreadyMessage := strings.Join(unreadyMessages, ", ")
	if unreadyMessage != "" {
		return api.PodCondition{
			Type:    api.PodReady,
			Status:  api.ConditionFalse,
			Reason:  "ContainersNotReady",
			Message: unreadyMessage,
		}
	}

	return api.PodCondition{
		Type:   api.PodReady,
		Status: api.ConditionTrue,
	}
}

// GeneratePodInitializedCondition returns initialized condition if all init containers in a pod are ready, else it
// returns an uninitialized condition.
func GeneratePodInitializedCondition(spec *api.PodSpec, containerStatuses []api.ContainerStatus, podPhase api.PodPhase) api.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil && len(spec.InitContainers) > 0 {
		return api.PodCondition{
			Type:   api.PodInitialized,
			Status: api.ConditionFalse,
			Reason: "UnknownContainerStatuses",
		}
	}
	unknownContainers := []string{}
	unreadyContainers := []string{}
	for _, container := range spec.InitContainers {
		if containerStatus, ok := api.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				unreadyContainers = append(unreadyContainers, container.Name)
			}
		} else {
			unknownContainers = append(unknownContainers, container.Name)
		}
	}

	// If all init containers are known and succeeded, just return PodCompleted.
	if podPhase == api.PodSucceeded && len(unknownContainers) == 0 {
		return api.PodCondition{
			Type:   api.PodInitialized,
			Status: api.ConditionTrue,
			Reason: "PodCompleted",
		}
	}

	unreadyMessages := []string{}
	if len(unknownContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unknown status: %s", unknownContainers))
	}
	if len(unreadyContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with incomplete status: %s", unreadyContainers))
	}
	unreadyMessage := strings.Join(unreadyMessages, ", ")
	if unreadyMessage != "" {
		return api.PodCondition{
			Type:    api.PodInitialized,
			Status:  api.ConditionFalse,
			Reason:  "ContainersNotInitialized",
			Message: unreadyMessage,
		}
	}

	return api.PodCondition{
		Type:   api.PodInitialized,
		Status: api.ConditionTrue,
	}
}
