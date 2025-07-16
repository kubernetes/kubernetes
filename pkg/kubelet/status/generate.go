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

	v1 "k8s.io/api/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
)

const (
	// UnknownContainerStatuses says that all container statuses are unknown.
	UnknownContainerStatuses = "UnknownContainerStatuses"
	// PodCompleted says that all related containers have succeeded.
	PodCompleted = "PodCompleted"
	// PodFailed says that the pod has failed and as such the containers have failed.
	PodFailed = "PodFailed"
	// ContainersNotReady says that one or more containers are not ready.
	ContainersNotReady = "ContainersNotReady"
	// ContainersNotInitialized says that one or more init containers have not succeeded.
	ContainersNotInitialized = "ContainersNotInitialized"
	// ReadinessGatesNotReady says that one or more pod readiness gates are not ready.
	ReadinessGatesNotReady = "ReadinessGatesNotReady"
)

// GenerateContainersReadyCondition returns the status of "ContainersReady" condition.
// The status of "ContainersReady" condition is true when all containers are ready.
func GenerateContainersReadyCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil {
		return v1.PodCondition{
			Type:               v1.ContainersReady,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.ContainersReady),
			Status:             v1.ConditionFalse,
			Reason:             UnknownContainerStatuses,
		}
	}
	unknownContainers := []string{}
	unreadyContainers := []string{}

	for _, container := range pod.Spec.InitContainers {
		if !podutil.IsRestartableInitContainer(&container) {
			continue
		}

		if containerStatus, ok := podutil.GetContainerStatus(containerStatuses, container.Name); ok {
			if !containerStatus.Ready {
				unreadyContainers = append(unreadyContainers, container.Name)
			}
		} else {
			unknownContainers = append(unknownContainers, container.Name)
		}
	}

	for _, container := range pod.Spec.Containers {
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
		return generateContainersReadyConditionForTerminalPhase(pod, oldPodStatus, podPhase)
	}

	// If the pod phase is failed, explicitly set the ready condition to false for containers since they may be in progress of terminating.
	if podPhase == v1.PodFailed {
		return generateContainersReadyConditionForTerminalPhase(pod, oldPodStatus, podPhase)
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
			Type:               v1.ContainersReady,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.ContainersReady),
			Status:             v1.ConditionFalse,
			Reason:             ContainersNotReady,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.ContainersReady,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.ContainersReady),
		Status:             v1.ConditionTrue,
	}
}

// GeneratePodReadyCondition returns "Ready" condition of a pod.
// The status of "Ready" condition is "True", if all containers in a pod are ready
// AND all matching conditions specified in the ReadinessGates have status equal to "True".
func GeneratePodReadyCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, conditions []v1.PodCondition, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	containersReady := GenerateContainersReadyCondition(pod, oldPodStatus, containerStatuses, podPhase)
	// If the status of ContainersReady is not True, return the same status, reason and message as ContainersReady.
	if containersReady.Status != v1.ConditionTrue {
		return v1.PodCondition{
			Type:               v1.PodReady,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReady),
			Status:             containersReady.Status,
			Reason:             containersReady.Reason,
			Message:            containersReady.Message,
		}
	}

	// Evaluate corresponding conditions specified in readiness gate
	// Generate message if any readiness gate is not satisfied.
	unreadyMessages := []string{}
	for _, rg := range pod.Spec.ReadinessGates {
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
			Type:               v1.PodReady,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReady),
			Status:             v1.ConditionFalse,
			Reason:             ReadinessGatesNotReady,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.PodReady,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReady),
		Status:             v1.ConditionTrue,
	}
}

func isInitContainerInitialized(initContainer *v1.Container, containerStatus *v1.ContainerStatus) bool {
	if podutil.IsRestartableInitContainer(initContainer) {
		if containerStatus.Started == nil || !*containerStatus.Started {
			return false
		}
	} else { // regular init container
		if !containerStatus.Ready {
			return false
		}
	}
	return true
}

// GeneratePodInitializedCondition returns initialized condition if all init containers in a pod are ready, else it
// returns an uninitialized condition.
func GeneratePodInitializedCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil && len(pod.Spec.InitContainers) > 0 {
		return v1.PodCondition{
			Type:               v1.PodInitialized,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodInitialized),
			Status:             v1.ConditionFalse,
			Reason:             UnknownContainerStatuses,
		}
	}

	unknownContainers := []string{}
	incompleteContainers := []string{}
	for _, container := range pod.Spec.InitContainers {
		containerStatus, ok := podutil.GetContainerStatus(containerStatuses, container.Name)
		if !ok {
			unknownContainers = append(unknownContainers, container.Name)
			continue
		}
		if !isInitContainerInitialized(&container, &containerStatus) {
			incompleteContainers = append(incompleteContainers, container.Name)
		}
	}

	// If all init containers are known and succeeded, just return PodCompleted.
	if podPhase == v1.PodSucceeded && len(unknownContainers) == 0 {
		return v1.PodCondition{
			Type:               v1.PodInitialized,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodInitialized),
			Status:             v1.ConditionTrue,
			Reason:             PodCompleted,
		}
	}

	// If there is any regular container that has started, then the pod has
	// been initialized before.
	// This is needed to handle the case where the pod has been initialized but
	// the restartable init containers are restarting.
	if kubecontainer.HasAnyRegularContainerStarted(&pod.Spec, containerStatuses) {
		return v1.PodCondition{
			Type:               v1.PodInitialized,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodInitialized),
			Status:             v1.ConditionTrue,
		}
	}

	unreadyMessages := make([]string, 0, len(unknownContainers)+len(incompleteContainers))
	if len(unknownContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with unknown status: %s", unknownContainers))
	}
	if len(incompleteContainers) > 0 {
		unreadyMessages = append(unreadyMessages, fmt.Sprintf("containers with incomplete status: %s", incompleteContainers))
	}
	unreadyMessage := strings.Join(unreadyMessages, ", ")
	if unreadyMessage != "" {
		return v1.PodCondition{
			Type:               v1.PodInitialized,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodInitialized),
			Status:             v1.ConditionFalse,
			Reason:             ContainersNotInitialized,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.PodInitialized,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodInitialized),
		Status:             v1.ConditionTrue,
	}
}

func GeneratePodReadyToStartContainersCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, podStatus *kubecontainer.PodStatus) v1.PodCondition {
	newSandboxNeeded, _, _ := runtimeutil.PodSandboxChanged(pod, podStatus)
	// if a new sandbox does not need to be created for a pod, it indicates that
	// a sandbox for the pod with networking configured already exists.
	// Otherwise, the kubelet needs to invoke the container runtime to create a
	// fresh sandbox and configure networking for the sandbox.
	if !newSandboxNeeded {
		return v1.PodCondition{
			Type:               v1.PodReadyToStartContainers,
			ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReadyToStartContainers),
			Status:             v1.ConditionTrue,
		}
	}
	return v1.PodCondition{
		Type:               v1.PodReadyToStartContainers,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReadyToStartContainers),
		Status:             v1.ConditionFalse,
	}
}

func generateContainersReadyConditionForTerminalPhase(pod *v1.Pod, oldPodStatus *v1.PodStatus, podPhase v1.PodPhase) v1.PodCondition {
	condition := v1.PodCondition{
		Type:               v1.ContainersReady,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.ContainersReady),
		Status:             v1.ConditionFalse,
	}

	if podPhase == v1.PodFailed {
		condition.Reason = PodFailed
	} else if podPhase == v1.PodSucceeded {
		condition.Reason = PodCompleted
	}

	return condition
}

func generatePodReadyConditionForTerminalPhase(pod *v1.Pod, oldPodStatus *v1.PodStatus, podPhase v1.PodPhase) v1.PodCondition {
	condition := v1.PodCondition{
		Type:               v1.PodReady,
		ObservedGeneration: podutil.GetPodObservedGenerationIfEnabledOnCondition(oldPodStatus, pod.Generation, v1.PodReady),
		Status:             v1.ConditionFalse,
	}

	if podPhase == v1.PodFailed {
		condition.Reason = PodFailed
	} else if podPhase == v1.PodSucceeded {
		condition.Reason = PodCompleted
	}

	return condition
}
