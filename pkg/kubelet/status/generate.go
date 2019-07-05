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

	"k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

const (
	UnknownContainerStatuses = "UnknownContainerStatuses"
	PodCompleted             = "PodCompleted"
	ContainersNotReady       = "ContainersNotReady"
	ContainersNotInitialized = "ContainersNotInitialized"
	ReadinessGatesNotReady   = "ReadinessGatesNotReady"
)

// GenerateContainersReadyCondition returns the status of "ContainersReady" condition.
// The status of "ContainersReady" condition is true when all containers are ready.
func GenerateContainersReadyCondition(spec *v1.PodSpec, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil {
		return v1.PodCondition{
			Type:   v1.ContainersReady,
			Status: v1.ConditionFalse,
			Reason: UnknownContainerStatuses,
		}
	}
	unknownContainers := []string{}
	unreadyContainers := []string{}
	for _, container := range spec.Containers {
		if containerStatus, ok := podutil.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				unreadyContainers = append(unreadyContainers, container.Name)
			}
		} else {
			unknownContainers = append(unknownContainers, container.Name)
		}
	}

	// If all containers are known and succeeded, just return PodCompleted.
	if podPhase == v1.PodSucceeded && len(unknownContainers) == 0 {
		return v1.PodCondition{
			Type:   v1.ContainersReady,
			Status: v1.ConditionFalse,
			Reason: PodCompleted,
		}
	}

	// Generate message for containers in unknown condition.
	unreadyMessages := []string{}
	if len(unknownContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unknown status: %s", unknownContainers))
	}
	if len(unreadyContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unready status: %s", unreadyContainers))
	}
	unreadyMessage := strings.Join(unreadyMessages, ", ")
	if unreadyMessage != "" {
		return v1.PodCondition{
			Type:    v1.ContainersReady,
			Status:  v1.ConditionFalse,
			Reason:  ContainersNotReady,
			Message: unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:   v1.ContainersReady,
		Status: v1.ConditionTrue,
	}
}

// GeneratePodReadyCondition returns "Ready" condition of a pod.
// The status of "Ready" condition is "True", if all containers in a pod are ready
// AND all matching conditions specified in the ReadinessGates have status equal to "True".
func GeneratePodReadyCondition(spec *v1.PodSpec, conditions []v1.PodCondition, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	containersReady := GenerateContainersReadyCondition(spec, containerStatuses, podPhase)
	// If the status of ContainersReady is not True, return the same status, reason and message as ContainersReady.
	if containersReady.Status != v1.ConditionTrue {
		return v1.PodCondition{
			Type:    v1.PodReady,
			Status:  containersReady.Status,
			Reason:  containersReady.Reason,
			Message: containersReady.Message,
		}
	}

	// Evaluate corresponding conditions specified in readiness gate
	// Generate message if any readiness gate is not satisfied.
	unreadyMessages := []string{}
	for _, rg := range spec.ReadinessGates {
		_, c := podutil.GetPodConditionFromList(conditions, rg.ConditionType)
		if c == nil {
			unreadyMessages = append(unreadyMessages, fmt.Sprintf("corresponding condition of pod readiness gate %q does not exist.", string(rg.ConditionType)))
		} else if c.Status != v1.ConditionTrue {
			unreadyMessages = append(unreadyMessages, fmt.Sprintf("the status of pod readiness gate %q is not \"True\", but %v", string(rg.ConditionType), c.Status))
		}
	}

	// Set "Ready" condition to "False" if any readiness gate is not ready.
	if len(unreadyMessages) != 0 {
		unreadyMessage := strings.Join(unreadyMessages, ", ")
		return v1.PodCondition{
			Type:    v1.PodReady,
			Status:  v1.ConditionFalse,
			Reason:  ReadinessGatesNotReady,
			Message: unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:   v1.PodReady,
		Status: v1.ConditionTrue,
	}
}

// GeneratePodInitializedCondition returns initialized condition if all init containers in a pod are ready, else it
// returns an uninitialized condition.
func GeneratePodInitializedCondition(spec *v1.PodSpec, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil && len(spec.InitContainers) > 0 {
		return v1.PodCondition{
			Type:   v1.PodInitialized,
			Status: v1.ConditionFalse,
			Reason: UnknownContainerStatuses,
		}
	}
	unknownContainers := []string{}
	unreadyContainers := []string{}
	for _, container := range spec.InitContainers {
		if containerStatus, ok := podutil.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				unreadyContainers = append(unreadyContainers, container.Name)
			}
		} else {
			unknownContainers = append(unknownContainers, container.Name)
		}
	}

	// If all init containers are known and succeeded, just return PodCompleted.
	if podPhase == v1.PodSucceeded && len(unknownContainers) == 0 {
		return v1.PodCondition{
			Type:   v1.PodInitialized,
			Status: v1.ConditionTrue,
			Reason: PodCompleted,
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
		return v1.PodCondition{
			Type:    v1.PodInitialized,
			Status:  v1.ConditionFalse,
			Reason:  ContainersNotInitialized,
			Message: unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:   v1.PodInitialized,
		Status: v1.ConditionTrue,
	}
}
