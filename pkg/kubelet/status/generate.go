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
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/securitycontext"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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
	// RestartAllContainersStarted says that a container exited and triggered RestartAllContainer action.
	RestartAllContainersStarted = "RestartAllContainersStarted"
	// ImplicitlyInsecureUserID says a container is running as UID 0 without runAsUser set.
	ImplicitlyInsecureUserID = "ImplicitlyInsecureUserID"
	// ImplicitlyInsecureGroupID says a container is running as GID 0 without runAsGroup set.
	ImplicitlyInsecureGroupID = "ImplicitlyInsecureGroupID"
	// ImplicitlyInsecureUserAndGroupID says a container is implicitly running as both UID 0 and GID 0.
	ImplicitlyInsecureUserAndGroupID = "ImplicitlyInsecureUserAndGroupID"
)

// GenerateContainersReadyCondition returns the status of "ContainersReady" condition.
// The status of "ContainersReady" condition is true when all containers are ready.
func GenerateContainersReadyCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus, podPhase v1.PodPhase) v1.PodCondition {
	// Find if all containers are ready or not.
	if containerStatuses == nil {
		return v1.PodCondition{
			Type:               v1.ContainersReady,
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.ContainersReady),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.ContainersReady),
			Status:             v1.ConditionFalse,
			Reason:             ContainersNotReady,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.ContainersReady,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.ContainersReady),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReady),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReady),
			Status:             v1.ConditionFalse,
			Reason:             ReadinessGatesNotReady,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.PodReady,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReady),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
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
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
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
		// During pod in-place restart, init container status can change from completed to waiting.
		// However, it is assumed that once a pod is initialized, it cannot be uninitialized. If
		// the pod is already initialized, the condition is kept.
		if utilfeature.DefaultFeatureGate.Enabled(features.RestartAllContainersOnContainerExits) {
			for _, cond := range oldPodStatus.Conditions {
				if cond.Type == v1.PodInitialized && cond.Status == v1.ConditionTrue {
					return v1.PodCondition{
						Type:               v1.PodInitialized,
						ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
						Status:             v1.ConditionTrue,
					}
				}
			}
		}
		return v1.PodCondition{
			Type:               v1.PodInitialized,
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
			Status:             v1.ConditionFalse,
			Reason:             ContainersNotInitialized,
			Message:            unreadyMessage,
		}
	}

	return v1.PodCondition{
		Type:               v1.PodInitialized,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodInitialized),
		Status:             v1.ConditionTrue,
	}
}

func GeneratePodReadyToStartContainersCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, podStatus *kubecontainer.PodStatus) v1.PodCondition {
	newSandboxNeeded, _, _, reason := runtimeutil.PodSandboxChanged(pod, podStatus)
	// if a new sandbox does not need to be created for a pod, it indicates that
	// a sandbox for the pod with networking configured already exists.
	// Otherwise, the kubelet needs to invoke the container runtime to create a
	// fresh sandbox and configure networking for the sandbox.
	if !newSandboxNeeded {
		return v1.PodCondition{
			Type:               v1.PodReadyToStartContainers,
			ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReadyToStartContainers),
			Status:             v1.ConditionTrue,
		}
	}
	return v1.PodCondition{
		Type:               v1.PodReadyToStartContainers,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReadyToStartContainers),
		Status:             v1.ConditionFalse,
		Reason:             kubetypes.PodSandboxNotReadyReason,
		Message:            reason,
	}
}

func generateContainersReadyConditionForTerminalPhase(pod *v1.Pod, oldPodStatus *v1.PodStatus, podPhase v1.PodPhase) v1.PodCondition {
	condition := v1.PodCondition{
		Type:               v1.ContainersReady,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.ContainersReady),
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
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, v1.PodReady),
		Status:             v1.ConditionFalse,
	}

	if podPhase == v1.PodFailed {
		condition.Reason = PodFailed
	} else if podPhase == v1.PodSucceeded {
		condition.Reason = PodCompleted
	}

	return condition
}

func GenerateAllContainersRestartingCondition(pod *v1.Pod, podStatus *kubecontainer.PodStatus, oldPodStatus *v1.PodStatus, podPhase v1.PodPhase) v1.PodCondition {
	if podPhase == v1.PodSucceeded {
		return v1.PodCondition{
			Type:   v1.AllContainersRestarting,
			Status: v1.ConditionFalse,
			Reason: PodCompleted,
		}
	}
	if podPhase == v1.PodFailed {
		return v1.PodCondition{
			Type:   v1.AllContainersRestarting,
			Status: v1.ConditionFalse,
			Reason: PodFailed,
		}
	}

	if !kubecontainer.ShouldAllContainersRestart(pod, podStatus, oldPodStatus) {
		return v1.PodCondition{
			Type:   v1.AllContainersRestarting,
			Status: v1.ConditionFalse,
		}
	}
	if kubecontainer.AllContainersRestartCleanedUp(pod, podStatus) {
		return v1.PodCondition{
			Type:   v1.AllContainersRestarting,
			Status: v1.ConditionFalse,
		}
	}
	return v1.PodCondition{
		Type:    v1.AllContainersRestarting,
		Status:  v1.ConditionTrue,
		Reason:  RestartAllContainersStarted,
		Message: "container exited with restart policy rule",
	}
}

// findContainersMissingUserInfo returns names of containers the runtime hasn't reported user info for yet.
func findContainersMissingUserInfo(pod *v1.Pod, containerStatuses []v1.ContainerStatus) []string {
	var names []string
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(container *v1.Container, _ podutil.ContainerType) bool {
		status, ok := podutil.GetContainerStatus(containerStatuses, container.Name)
		if !ok || status.User == nil || status.User.Linux == nil {
			names = append(names, container.Name)
		}
		return true
	})
	return names
}

// findRootContainers returns the names of containers running with an insecure ID, per isIDInsecure and explicit.
func findRootContainers(pod *v1.Pod, containerStatuses []v1.ContainerStatus,
	isIDInsecure func(status v1.ContainerStatus) bool,
	effectiveRequestedID func(pod *v1.Pod, container *v1.Container) (*int64, bool),
	explicit bool) []string {
	var names []string
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(container *v1.Container, _ podutil.ContainerType) bool {
		status, ok := podutil.GetContainerStatus(containerStatuses, container.Name)
		if !ok || status.User == nil || status.User.Linux == nil || !isIDInsecure(status) {
			return true
		}
		requestedID, requested := effectiveRequestedID(pod, container)
		if explicit {
			if requested && *requestedID == 0 {
				names = append(names, container.Name)
			}
		} else if !requested {
			names = append(names, container.Name)
		}
		return true
	})
	return names
}

// GenerateInsecureUserIDCondition returns the "InsecureUserID" condition.
func GenerateInsecureUserIDCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus) v1.PodCondition {
	conditionType := v1.InsecureUserID
	cond := v1.PodCondition{
		Type:               conditionType,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, conditionType),
	}

	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		cond.Status = v1.ConditionFalse
		return cond
	}

	insecureContainerNames := findRootContainers(pod, containerStatuses,
		func(status v1.ContainerStatus) bool { return status.User.Linux.UID == 0 },
		securitycontext.DetermineEffectiveRunAsUser, false)

	if len(insecureContainerNames) > 0 {
		cond.Status = v1.ConditionTrue
		cond.Reason = ImplicitlyInsecureUserID
		cond.Message = fmt.Sprintf("container(s) %s running as UID 0 without runAsUser set", insecureContainerNames)
		return cond
	}

	if unknownContainerNames := findContainersMissingUserInfo(pod, containerStatuses); len(unknownContainerNames) > 0 {
		cond.Status = v1.ConditionUnknown
		cond.Message = fmt.Sprintf("container(s) %s: UID not yet reported", unknownContainerNames)
		return cond
	}

	cond.Status = v1.ConditionFalse
	return cond
}

// findInsecureSupplementalGroupsContainers returns names of containers whose resolved
// supplemental groups include GID 0, split by explicit vs. implicit.
func findInsecureSupplementalGroupsContainers(pod *v1.Pod, containerStatuses []v1.ContainerStatus, explicit bool) []string {
	podRequestsGID0AsSupplementalGroup := false
	if sc := pod.Spec.SecurityContext; sc != nil {
		if sc.FSGroup != nil && *sc.FSGroup == 0 {
			podRequestsGID0AsSupplementalGroup = true
		}
		if slices.Contains(sc.SupplementalGroups, int64(0)) {
			podRequestsGID0AsSupplementalGroup = true
		}
	}

	var containerNames []string
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(container *v1.Container, _ podutil.ContainerType) bool {
		status, ok := podutil.GetContainerStatus(containerStatuses, container.Name)
		if !ok || status.User == nil || status.User.Linux == nil {
			return true
		}
		// Note: CRI runtimes always mirror a container's own primary GID into its reported
		// SupplementalGroups (containerd: https://github.com/containerd/containerd/blob/a8fc3a017297f9ac4a28b115f9b706a90f497851/pkg/oci/spec_opts.go#L134-L141,
		// cri-o: https://github.com/cri-o/cri-o/blob/efbce04159ead73850f34c289f333126ae9b7b88/server/container_create.go#L366-L367).
		// A primary GID of 0 is already covered by the primary-GID checks, so skip it here.
		if status.User.Linux.GID == 0 {
			return true
		}
		hasGID0AsSupplementalGroup := slices.Contains(status.User.Linux.SupplementalGroups, int64(0))
		if hasGID0AsSupplementalGroup && explicit == podRequestsGID0AsSupplementalGroup {
			containerNames = append(containerNames, container.Name)
		}
		return true
	})
	return containerNames
}

// GenerateInsecureGroupIDCondition returns the status of the "InsecureGroupID" condition.
func GenerateInsecureGroupIDCondition(pod *v1.Pod, oldPodStatus *v1.PodStatus, containerStatuses []v1.ContainerStatus) v1.PodCondition {
	conditionType := v1.InsecureGroupID
	cond := v1.PodCondition{
		Type:               conditionType,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(oldPodStatus, pod.Generation, conditionType),
	}

	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		cond.Status = v1.ConditionFalse
		return cond
	}

	containersWithInsecurePrimaryGID := findRootContainers(pod, containerStatuses,
		func(status v1.ContainerStatus) bool { return status.User.Linux.GID == 0 },
		securitycontext.DetermineEffectiveRunAsGroup, false)
	containersWithInsecureSupplementalGroup := findInsecureSupplementalGroupsContainers(pod, containerStatuses, false)

	if len(containersWithInsecurePrimaryGID) > 0 || len(containersWithInsecureSupplementalGroup) > 0 {
		cond.Status = v1.ConditionTrue
		cond.Reason = ImplicitlyInsecureGroupID
		var messages []string
		if len(containersWithInsecurePrimaryGID) > 0 {
			messages = append(messages, fmt.Sprintf("container(s) %s running as GID 0 without runAsGroup set", containersWithInsecurePrimaryGID))
		}
		if len(containersWithInsecureSupplementalGroup) > 0 {
			messages = append(messages, fmt.Sprintf("container(s) %s running with GID 0 merged into supplementalGroups from the image (supplementalGroupsPolicy: Merge)", containersWithInsecureSupplementalGroup))
		}
		cond.Message = strings.Join(messages, "; ")
		return cond
	}

	if unknownContainerNames := findContainersMissingUserInfo(pod, containerStatuses); len(unknownContainerNames) > 0 {
		cond.Status = v1.ConditionUnknown
		cond.Message = fmt.Sprintf("container(s) %s: GID not yet reported", unknownContainerNames)
		return cond
	}

	cond.Status = v1.ConditionFalse
	return cond
}

// IsPodExplicitlyInsecureUserID reports whether a container explicitly requests UID 0.
func IsPodExplicitlyInsecureUserID(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool {
	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		return false
	}
	return len(findRootContainers(pod, containerStatuses,
		func(status v1.ContainerStatus) bool { return status.User.Linux.UID == 0 },
		securitycontext.DetermineEffectiveRunAsUser, true)) > 0
}

// IsPodExplicitlyInsecureGroupID reports whether a container explicitly requests GID 0.
func IsPodExplicitlyInsecureGroupID(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool {
	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		return false
	}
	return len(findRootContainers(pod, containerStatuses,
		func(status v1.ContainerStatus) bool { return status.User.Linux.GID == 0 },
		securitycontext.DetermineEffectiveRunAsGroup, true)) > 0
}

// IsPodImplicitlyInsecurePrimaryGroupID reports whether a container's primary GID is
// implicitly 0, ignoring supplemental groups.
func IsPodImplicitlyInsecurePrimaryGroupID(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool {
	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		return false
	}
	return len(findRootContainers(pod, containerStatuses,
		func(status v1.ContainerStatus) bool { return status.User.Linux.GID == 0 },
		securitycontext.DetermineEffectiveRunAsGroup, false)) > 0
}

// IsPodImplicitlyInsecureSupplementalGroups reports whether a container implicitly has
// GID 0 as a supplemental group.
func IsPodImplicitlyInsecureSupplementalGroups(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool {
	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		return false
	}
	return len(findInsecureSupplementalGroupsContainers(pod, containerStatuses, false)) > 0
}

// IsPodExplicitlyInsecureSupplementalGroups reports whether a pod explicitly requests
// GID 0 as a supplemental group.
func IsPodExplicitlyInsecureSupplementalGroups(pod *v1.Pod, containerStatuses []v1.ContainerStatus) bool {
	if pod.Spec.HostUsers != nil && !*pod.Spec.HostUsers {
		return false
	}
	return len(findInsecureSupplementalGroupsContainers(pod, containerStatuses, true)) > 0
}
