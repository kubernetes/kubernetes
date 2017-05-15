/*
Copyright 2016 The Kubernetes Authors.

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

package kuberuntime

import (
	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
)

// statusGroup groups the sandbox statuses, init container statuses and app container
// (i.e., regular user-defined container in the pod spec) statues.
type statusGroup struct {
	sandboxStatuses       []*runtimeapi.PodSandboxStatus
	initContainerStatuses []*kubecontainer.ContainerStatus
	appContainerStatuses  []*kubecontainer.ContainerStatus
}

// newStatusGroup returns a statusGroup object that contains the sandbox statuses,
// init container statuses and the app container statuses.
func newStatusGroup(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *statusGroup {
	var initContainerStatuses []*kubecontainer.ContainerStatus
	var appContainerStatuses []*kubecontainer.ContainerStatus

	initContainers := make(map[string]struct{})
	appContainers := make(map[string]struct{})

	for _, c := range pod.Spec.InitContainers {
		initContainers[c.Name] = struct{}{}
	}
	for _, c := range pod.Spec.Containers {
		appContainers[c.Name] = struct{}{}
	}

	for _, s := range podStatus.ContainerStatuses {
		if _, ok := initContainers[s.Name]; ok {
			initContainerStatuses = append(initContainerStatuses, s)
			continue
		}
		if _, ok := appContainers[s.Name]; ok {
			appContainerStatuses = append(appContainerStatuses, s)
		}
	}

	return &statusGroup{
		sandboxStatuses:       podStatus.SandboxStatuses,
		initContainerStatuses: initContainerStatuses,
		appContainerStatuses:  appContainerStatuses,
	}
}

// containerKillInfo describes a running container that needs to be killed, and the reason for that.
type containerKillInfo struct {
	// The status of the container, it contains the
	// container ID, name, restart count, finished timestamp.
	// The status will be passed to containerStartInfo if the container
	// needs to be restart.
	status *kubecontainer.ContainerStatus
	// The reason why the container will be killed, e.g. due to failures.
	reason string
	// The boolean indicates whether the container will be restarted.
	restart bool
}

// containerStartInfo describes a container that needs to be started, and the reason for that.
type containerStartInfo struct {
	// The pointer to the API container that needs to be started.
	container *v1.Container
	// The reason why the container will be started, e.g. due to restart.
	// Used for logging.
	reason string
	// The last status of the container if it exits.
	// This is used to determine the "attempt count" and whether we need to do backoff.
	status *kubecontainer.ContainerStatus
}

// sandboxKillInfo describes a sandbox that needs to be killed, and the reason for that.
type sandboxKillInfo struct {
	// The ID of the sandbox, runtime uses it to kill the sandbox.
	id string
	// The name of the sandbox, used for logging.
	name string
	// The reason why the sandbox will be killed. e.g. due to failures.
	reason string
	// The boolean indicates whether the sandbox will be restarted.
	restart bool
}

// sandboxStartInfo describes a sandbox that needs to be started, and the reason for that.
type sandboxStartInfo struct {
	// The pointer to the API pod that needs to be started.
	sandbox *v1.Pod
	// The number of the attempts that we have tried so far to start
	// the sandbox.
	attempt int
	// The reason why the sandbox will be started. e.g. due to restart.
	reason string
}

// syncAction describes the actions that needs to be taken for one sync pod iteration.
// Note that containersToKill/containersToStart includes both regular (app) and init containers.
type syncAction struct {
	containersToKill  []*containerKillInfo
	containersToStart []*containerStartInfo

	sandboxesToKill []*sandboxKillInfo
	sandboxToStart  *sandboxStartInfo
}

// addSandboxToKill add one more sandbox info to the sandboxesToKill list.
func (a *syncAction) addSandboxToKill(status *runtimeapi.PodSandboxStatus, reason string, restart bool) {
	var name string

	// Metadata must be non nil.
	if status.Metadata != nil {
		name = status.Metadata.Name
	}
	a.sandboxesToKill = append(a.sandboxesToKill, &sandboxKillInfo{
		id:      status.Id,
		name:    name,
		reason:  reason,
		restart: restart,
	})
}

// addContainerToKill add one more container info the the containersToKill list.
func (a *syncAction) addContainerToKill(s *kubecontainer.ContainerStatus, reason string, restart bool) {
	a.containersToKill = append(a.containersToKill, &containerKillInfo{
		status:  s,
		reason:  reason,
		restart: restart,
	})
}

// addSandboxToStart add the info of the sandbox that's going to be started.
func (a *syncAction) addSandboxToStart(pod *v1.Pod, attempt int, reason string) {
	a.sandboxToStart = &sandboxStartInfo{
		attempt: attempt,
		reason:  reason,
	}
}

// addContainerToStart add one more container info the the containersToStart list.
func (a *syncAction) addContainerToStart(container *v1.Container, reason string, status *kubecontainer.ContainerStatus) {
	a.containersToStart = append(a.containersToStart, &containerStartInfo{
		container: container,
		reason:    reason,
		status:    status,
	})
}

// needsToKillContainers returns true if there are any containers need to be killed.
func (a *syncAction) needsToKillContainers() bool {
	return len(a.containersToKill) > 0
}

// needsToKillSandboxes returns true if there are any sandboxes need to be killed.
func (a *syncAction) needsToKillSandboxes() bool {
	return len(a.sandboxesToKill) > 0
}

// needsToStartSandbox returns true if a sandbox needs to be started.
func (a *syncAction) needsToStartSandbox() bool {
	return a.sandboxToStart != nil
}

// needsToStartContainers returns true if there are any containers that need
// to be started.
func (a *syncAction) needsToStartContainers() bool {
	return len(a.containersToStart) > 0
}

func (a *syncAction) computeSyncActionKillingPhase(statuses *statusGroup, manager proberesults.Manager) {
}
func (a *syncAction) computeSyncActionStartingPhase(statuses *statusGroup)  {}
func (a *syncAction) computeSyncActionSandboxCleanup(statuses *statusGroup) {}

// computeSyncAction generates the sync action object that contains the actions
// we need to take in the current sync pod iteration.
// The whole computation is splitted into 3 phases:
// - (1) killing phase computation, which constructs a list of sandboxs and containers that needs to be killed.
// - (2) starting phase computation, which constructs the sandbox and a list containers that needs to be started.
// - (3) sandbox cleanup phase computation, which checks if we need to clean up the sandbox.
func computeSyncAction(pod *v1.Pod, statuses *statusGroup, livenessManager proberesults.Manager) *syncAction {
	var a syncAction

	// (1) compute the sandboxes, and containers that we need to kill.
	a.computeSyncActionKillingPhase(statuses, livenessManager)

	// (2) compute the sandbox, and containers that we need to start.
	a.computeSyncActionStartingPhase(statuses)

	// (3) do an additional check to make sure we kill the sandbox when
	//     all containers are exited (or will be killed), and none of them will
	//     be restarted.
	a.computeSyncActionSandboxCleanup(statuses)

	return &a
}
