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
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

// statusGroup groups the sandbox statuses, init container statuses and app container statues.
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

// findFirstContainerStatusByName returns the first container status
// that matches the given name.
func findFirstContainerStatusByName(name string, s []*kubecontainer.ContainerStatus) *kubecontainer.ContainerStatus {
	for _, status := range s {
		if status.Name == name {
			return status
		}
	}
	return nil
}

// containerKillInfo describes a running container that needs to be killed, and the reason for that.
type containerKillInfo struct {
	// The ID of the container, runtime uses it to kill the container.
	id kubecontainer.ContainerID
	// The name of the container, used for logging.
	name string
	// The reason why the container will be killed, e.g. due to failures.
	reason string
	// The boolean indicates whether the container will be restarted.
	restart bool

	// The status of the container, could used by the starting phase
	// to know the attempt count and finish timestamp for back-off.
	// TODO(yifan): This has some duplicated info including id and name.
	// Figure out if we want to merge two.
	status *kubecontainer.ContainerStatus
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

// sandboxContext includes the information that are necessary in the container creation phase.
type sandboxContext struct {
	// The pod references the API pod object for this sync action.
	pod *v1.Pod
	// The status of the sandbox.
	sandboxStatus *runtimeapi.PodSandboxStatus
	// The pull secrets.
	pullSecrets []v1.Secret
}

// syncAction describes the actions that needs to be taken for one sync pod iteration.
type syncAction struct {
	containersToKill  []*containerKillInfo
	containersToStart []*containerStartInfo // Take cares of the init containers.

	sandboxesToKill []*sandboxKillInfo
	sandboxToStart  *sandboxStartInfo

	// The context about the sandbox that are necessary in
	// sandbox termination, creation, and container creation phase.
	sandboxContext *sandboxContext
}

// newSyncAction returns a new *syncAction object.
func newSyncAction(pod *v1.Pod, pullSecrets []v1.Secret) *syncAction {
	return &syncAction{
		sandboxContext: &sandboxContext{
			pod:         pod,
			pullSecrets: pullSecrets,
		},
	}
}

// findSandboxKillInfoByID returns the sandboxKillInfo that
// has the given sandbox ID.
func (a *syncAction) findSandboxKillInfoByID(id string) *sandboxKillInfo {
	for _, info := range a.sandboxesToKill {
		if info.id == id {
			return info
		}
	}
	return nil
}

// findContainerKillInfoByID returns the containerKillInfo that
// has the given container ID.
func (a *syncAction) findContainerKillInfoByID(id kubecontainer.ContainerID) *containerKillInfo {
	for _, info := range a.containersToKill {
		if info.id == id {
			return info
		}
	}
	return nil
}

// needsToKillContainers returns true if there are any containers need to be killed.
func (a *syncAction) needsToKillContainers() bool {
	return len(a.containersToKill) > 0
}

// killContainers kills all the containers in the containersToKill list
// and returns the sync result.
// If any error happens, it will return the error as well.
//
// TODO(yifan): Replace the kubeGenericRuntimeManager with interface for testing.
func (a *syncAction) killContainers(m *kubeGenericRuntimeManager) ([]*kubecontainer.SyncResult, error) {
	var syncResult []*kubecontainer.SyncResult
	var wg sync.WaitGroup
	syncResultCh := make(chan *kubecontainer.SyncResult, len(a.containersToKill))
	wg.Add(len(a.containersToKill))
	for _, info := range a.containersToKill {
		go func(info *containerKillInfo) {
			defer utilruntime.HandleCrash()
			defer wg.Done()

			// For killings that happen inside sync pod, there's no gracePeriodOverride.
			result := kubecontainer.NewSyncResult(kubecontainer.KillContainer, info.name)
			err := m.killContainer(a.sandboxContext.pod, info.id, info.name, info.reason, nil)
			if err == nil {
				info.status.FinishedAt = time.Now() // Update the 'FinishedAt' field so we can do correct backoff on killed containers.
			} else {
				result.Fail(kubecontainer.ErrKillContainer, err.Error())
			}
			syncResultCh <- result
		}(info)
	}
	wg.Wait()
	close(syncResultCh)

	var errlist []error
	for result := range syncResultCh {
		syncResult = append(syncResult, result)
		if result.Error != nil {
			errlist = append(errlist, result.Error)
		}
	}
	return syncResult, errors.NewAggregate(errlist)
}

// needsToKillSandboxes returns true if there are any sandboxes need to be killed.
func (a *syncAction) needsToKillSandboxes() bool {
	return len(a.sandboxesToKill) > 0
}

// killSandboxes kills all the sandboxes in the sandboxesToKill list
// and returns the sync result.
// If any error happens, it will return the error as well.
//
// TODO(yifan): Replace the kubeGenericRuntimeManager with interface for testing.
func (a *syncAction) killSandboxes(m *kubeGenericRuntimeManager) ([]*kubecontainer.SyncResult, error) {
	var syncResult []*kubecontainer.SyncResult
	var wg sync.WaitGroup

	syncResultCh := make(chan *kubecontainer.SyncResult, 2*len(a.sandboxesToKill))
	wg.Add(len(a.sandboxesToKill))

	for _, info := range a.sandboxesToKill {
		go func(info *sandboxKillInfo) {
			defer utilruntime.HandleCrash()
			defer wg.Done()

			result := kubecontainer.NewSyncResult(kubecontainer.KillPodSandbox, info.name)
			if err := m.killPodSandbox(info.id, info.name, info.reason); err != nil {
				result.Fail(kubecontainer.ErrKillPodSandbox, err.Error())
			}
			syncResultCh <- result
		}(info)
	}
	wg.Wait()
	close(syncResultCh)

	var errlist []error
	for result := range syncResultCh {
		syncResult = append(syncResult, result)
		if result.Error != nil {
			errlist = append(errlist, result.Error)
		}
	}
	return syncResult, errors.NewAggregate(errlist)
}

// needsToStartSandbox returns true if a sandbox needs to be started.
func (a *syncAction) needsToStartSandbox() bool {
	return a.sandboxToStart != nil
}

// startSandbox starts the sandbox. If any error happens, it returns the error.
// startSandbox will also override the sandbox ID and pod IP in the pod status
// after setting up the network plugin, so they can be used to create/start
// containers.
//
// TODO(yifan): Replace the kubeGenericRuntimeManager with interface for testing.
func (a *syncAction) startSandbox(m *kubeGenericRuntimeManager) ([]*kubecontainer.SyncResult, error) {
	var syncResult []*kubecontainer.SyncResult

	pod := a.sandboxContext.pod
	createResult := kubecontainer.NewSyncResult(kubecontainer.CreatePodSandbox, format.Pod(pod))
	syncResult = append(syncResult, createResult)

	// 1. Run the sandbox.
	sandboxID, err := m.runPodSandbox(pod, a.sandboxToStart.attempt, a.sandboxToStart.reason)
	if err != nil {
		createResult.Fail(kubecontainer.ErrCreatePodSandbox, fmt.Sprintf("runPodsandbox for pod %q failed: %v", format.Pod(pod), err))
		return syncResult, err
	}

	// 2. Update the sandbox status.
	a.sandboxContext.sandboxStatus, err = m.getSandboxStatus(sandboxID)
	if err != nil {
		createResult.Fail(kubecontainer.ErrCreatePodSandbox, err.Error())
		// TODO(yifan): Maybe also kill the sandbox here.
		return syncResult, err
	}

	return syncResult, nil
}

// needsToStartContainers returns true if there are any containers that need
// to be started.
func (a *syncAction) needsToStartContainers() bool {
	return len(a.containersToStart) > 0
}

// startContainers starts all the containers that need to be started, and returns
// any errors during the process.
// Note that this function does not distinguish between init containers and app
// containers, they are managed by computeSyncAction function.
// We might need to change this if we later want to start containers concurrently,
// because init containers will always start sequentially.
//
// TODO(yifan): Replace the kubeGenericRuntimeManager with interface for testing.
func (a *syncAction) startContainers(m *kubeGenericRuntimeManager, backoff *flowcontrol.Backoff) ([]*kubecontainer.SyncResult, error) {
	var syncResult []*kubecontainer.SyncResult

	pod := a.sandboxContext.pod
	pullSecrets := a.sandboxContext.pullSecrets

	// 1. Get the sandbox config if it is empty, this happens
	// when the sandbox is not newly created.
	sandboxID, sandboxIP, sandboxConfig, err := m.getSandboxInfoAndConfig(pod, a.sandboxContext.sandboxStatus)
	if err != nil {
		result := kubecontainer.NewSyncResult(kubecontainer.ConfigPodSandbox, a.sandboxContext.pod.Name)
		syncResult = append(syncResult, result)
		result.Fail(kubecontainer.ErrConfigPodSandbox, err.Error())
		return syncResult, err
	}

	// 2. Start each container.
	for _, info := range a.containersToStart {
		result := kubecontainer.NewSyncResult(kubecontainer.StartContainer, info.container.Name)
		syncResult = append(syncResult, result)

		container := info.container
		reason := info.reason
		status := info.status

		err, msg := m.runContainer(sandboxID, sandboxIP, sandboxConfig, pod, container, status, reason, pullSecrets, backoff)
		if err != nil {
			result.Fail(err, msg)
			continue
			// Return or continue?
		}
	}

	var errlist []error
	for _, result := range syncResult {
		if result.Error != nil {
			errlist = append(errlist, result.Error)
		}
	}
	return syncResult, errors.NewAggregate(errlist)
}

// addSandboxToKill add one more sandbox info to the sandboxesToKill list.
func (a *syncAction) addSandboxToKill(status *runtimeapi.PodSandboxStatus, reason string, restart bool) {
	a.sandboxesToKill = append(a.sandboxesToKill, &sandboxKillInfo{
		id:      status.GetId(),
		name:    status.Metadata.GetName(),
		reason:  reason,
		restart: restart,
	})
}

// addContainerToKill add one more container info the the containersToKill list.
func (a *syncAction) addContainerToKill(s *kubecontainer.ContainerStatus, reason string, restart bool) {
	a.containersToKill = append(a.containersToKill, &containerKillInfo{
		id:      s.ID,
		name:    s.Name,
		reason:  reason,
		restart: restart,
		status:  s,
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

// computeSyncActionKillingPhase computes the containersToKill list and sandboxesToKill list.
// TODO(yifan): Make kubeGenericRuntimeManager an interface for testing.
func (a *syncAction) computeSyncActionKillingPhase(statuses *statusGroup, manager proberesults.Manager) {
	pod := a.sandboxContext.pod
	restartPolicy := pod.Spec.RestartPolicy

	sandboxStatuses := statuses.sandboxStatuses
	initContainerStatuses := statuses.initContainerStatuses
	appContainerStatuses := statuses.appContainerStatuses

	// Phase 1, construct the sandbox killing list.
	//
	// A sandbox needs to be killed, if:
	// - (1.1) It's duplicated.
	// - (1.2) Init container spec is changed, this will also restart the sandbox.
	// - (1.3) At least one init container failed, and the restart policy is "Never".

	// (1.1)
	for i := 1; i < len(sandboxStatuses); i++ {
		s := sandboxStatuses[i]
		if s.GetState() == runtimeapi.PodSandboxState_SANDBOX_READY {
			a.addSandboxToKill(s, "duplicated sandbox", false)
		}
	}

	// (1.2)
	killSandbox := false
	var sandboxStatus *runtimeapi.PodSandboxStatus
	if len(sandboxStatuses) > 0 {
		sandboxStatus = sandboxStatuses[0]

		if sandboxStatus.GetState() == runtimeapi.PodSandboxState_SANDBOX_READY {
			// Now we have at most one sandbox, check if the spec has changed.
			// Currently the sandbox spec will not change, so we only check if any init container specs
			// have changed.
			if initContainersChanged(pod, sandboxStatus) {
				killSandbox = true
				reason := "init containers have been changed"
				a.addSandboxToKill(sandboxStatus, reason, true)
			}
		}
	}

	// (1.3)
	for _, s := range initContainerStatuses {
		if isContainerFailed(s) && restartPolicy == v1.RestartPolicyNever {
			killSandbox = true
			a.addSandboxToKill(sandboxStatus, "one of the init containers has failed", false)
			break
		}
	}

	// Phase 2, construct the container killing list.
	//
	// A container needs to be killed, if:
	// - (2.1) Its parent sandbox box is not found, not ready or will be killed.
	// - (2.2) It's not found in the pod spec.
	// - (2.3) It's duplicated.
	// - (2.4) Its spec has been changed.
	// - (2.5) It's unhealthy.

	// Populate the sandbox status.
	a.sandboxContext.sandboxStatus = sandboxStatus

	// (2.1)
	var reason string
	switch {
	case sandboxStatus == nil:
		reason = "sandbox is not found"
	case sandboxStatus.GetState() != runtimeapi.PodSandboxState_SANDBOX_READY:
		reason = "sandbox is not ready"
	case killSandbox:
		reason = "sandbox will be killed"
	}
	if reason != "" {
		for _, s := range append(initContainerStatuses, appContainerStatuses...) {
			if isContainerActive(s) {
				a.addContainerToKill(s, reason, true)
			}
		}
		return
	}

	// (2.2) ~ (2.5)
	type record struct {
		container *v1.Container
		found     bool
	}
	// (a) Check init containers.
	initContainersInSpec := make(map[string]*record)
	for i := range pod.Spec.InitContainers {
		c := &pod.Spec.InitContainers[i]
		initContainersInSpec[c.Name] = &record{c, false}
	}
	for _, s := range initContainerStatuses {
		if !isContainerActive(s) {
			continue
		}

		r, ok := initContainersInSpec[s.Name]
		if !ok {
			a.addContainerToKill(s, "container is not found in the pod spec", false)
			continue
		}
		if r.found {
			a.addContainerToKill(s, "duplicated container", false)
			continue
		}
		r.found = true
	}

	// (b) Check app containers.
	appContainersInSpec := make(map[string]*record)
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		appContainersInSpec[c.Name] = &record{c, false}
	}
	for _, s := range appContainerStatuses {
		if !isContainerActive(s) {
			continue
		}
		r, ok := appContainersInSpec[s.Name]
		if !ok {
			a.addContainerToKill(s, "container is not found in the pod spec", false)
			continue
		}
		if r.found {
			a.addContainerToKill(s, "duplicated container", false)
			continue
		}
		r.found = true

		if containerChanged(r.container, s) {
			a.addContainerToKill(s, "container has been changed", true)
			continue
		}

		if containerUnhealthy(manager, s) {
			restart := restartPolicy != v1.RestartPolicyNever
			a.addContainerToKill(s, "container is unhealthy", restart)
			continue
		}
	}

	return
}

// computeSyncActionStartingPhase computes the sandboxToStart and containersToStart list.
func (a *syncAction) computeSyncActionStartingPhase(statuses *statusGroup) {
	pod := a.sandboxContext.pod
	restartPolicy := pod.Spec.RestartPolicy

	initContainerStatuses := statuses.initContainerStatuses
	appContainerStatuses := statuses.appContainerStatuses

	// Phase 1, check if the sandbox needs to be started.
	//
	// A sandbox will be started, if:
	// - (1.1) It doesn't exist.
	// - (1.2) It's not ready.
	// - (1.3) The sandbox will be killed and should be restarted.

	// (1.1) ~ (1.2)
	status := a.sandboxContext.sandboxStatus
	if status == nil {
		a.addSandboxToStart(pod, 0, "new sandbox")
	} else {
		if status.GetState() != runtimeapi.PodSandboxState_SANDBOX_READY {
			a.addSandboxToStart(pod, int(status.GetMetadata().GetAttempt()+1), "restart unready sandbox")
		}
	}

	// (1.3)
	for _, sandbox := range a.sandboxesToKill {
		if sandbox.restart {
			a.addSandboxToStart(pod, int(status.GetMetadata().GetAttempt()+1), "restart killed sandbox")
			break // Should only have at most one sandbox to be restarted.
		}
	}

	// Phase 2, check if any init containers need to start.
	//
	// An init container will be started, if:
	// - (2.1) The sandbox will be started. In this case, start the first init container if it exists.
	// - (2.2) The previous init container succeeded, start the next one.
	// - (2.3) The previous init container failed, and it requires a restart.

	// (2.1)
	if a.sandboxToStart != nil && len(pod.Spec.InitContainers) > 0 {
		c := &pod.Spec.InitContainers[0]
		s := findFirstContainerStatusByName(c.Name, initContainerStatuses)
		a.addContainerToStart(c, "start init container", s)
		return
	}

	// (2.2) ~ (2.3)
	// we start the iteration backwards since terminated
	// init containers could be pruned.
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		c := &pod.Spec.InitContainers[i]

		s := findFirstContainerStatusByName(c.Name, initContainerStatuses)
		if s == nil {
			continue
		}

		// When the init container is found, it can be in 3 states:
		// - Active ('created' or 'running').
		// - Failed ('unknown' or 'exited' but exit code is non-zero).
		// - Succeeded ('exited' and exit code is zero).
		if isContainerActive(s) {
			// The init container is still active, nothing to do in this sync iteration.
			return
		}
		if isContainerFailed(s) {
			if restartPolicy == v1.RestartPolicyAlways {
				a.addContainerToStart(c, "restart init container", s)
			}
			// Restart policy "Never" is taken care by computeSyncActionKillingPhase(),
			// so we will start nothing here.
			return
		}

		// Now the previous init container succeeded, let's start the next one.
		if i == len(pod.Spec.InitContainers)-1 { // All init containers finished, enter phase 3.
			break
		}

		c = &pod.Spec.InitContainers[i+1]
		a.addContainerToStart(c, "start init container", nil)
		return
	}

	// Phase 3, check if any app containers need to be restarted.
	// This only happens when all init containers succeeds or there is no init containers.
	//
	// - (3.1) Start any containers that are not currently active, or exited but requires restart.
	// - (3.2) Start any to-be-killed containers that require a restart.

	// (3.1)
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]

		s := findFirstContainerStatusByName(c.Name, appContainerStatuses)
		if s == nil {
			a.addContainerToStart(c, "start container", nil)
			continue
		}

		if isContainerActive(s) {
			continue
		}

		// The container is not 'Active' ('created' or 'running'), it can be in 'unknown' or 'exited' state.
		// The container is considered as 'Failed' if it's in 'unknown' state or the exit code is non-zero.

		switch restartPolicy {
		case v1.RestartPolicyNever:
			continue
		case v1.RestartPolicyAlways:
			a.addContainerToStart(c, "restart non-active container since restart policy is 'Always'", s)
		case v1.RestartPolicyOnFailure:
			if isContainerFailed(s) {
				a.addContainerToStart(c, "restart failed container since restart policy is 'OnFailure'", s)
			}
		}
	}

	// (3.2)
	for _, c := range a.containersToKill {
		if c.restart {
			// Get the container spec.
			// Since we know the container will restart, so the container spec must exist
			container := findContainerSpecByName(c.name, pod)
			a.addContainerToStart(container, "restart the killed container", c.status)
		}
	}
}

// computeSyncActionSandboxCleanup makes sure the sandbox will be terminated if:
// - (1) No active containers are left, and no more containers will be stared.
// - (2) All active containers will be killed, and no more containers will be started.
func (a *syncAction) computeSyncActionSandboxCleanup(statuses *statusGroup) {
	sandboxStatuses := statuses.sandboxStatuses
	initContainerStatuses := statuses.initContainerStatuses
	appContainerStatuses := statuses.appContainerStatuses

	// No-op if the sandbox doesn't exist.
	if len(sandboxStatuses) == 0 {
		return
	}

	sandboxStatus := sandboxStatuses[0]

	// No-op if we already decide to kill the sandbox.
	if a.findSandboxKillInfoByID(sandboxStatus.GetId()) != nil {
		return
	}

	// No-op if we still need to start more containers.
	if len(a.containersToStart) > 0 {
		return
	}

	// (1).
	//
	// If no active containers left, kill the sandbox.
	var activeContainers []*kubecontainer.ContainerStatus
	for _, s := range append(initContainerStatuses, appContainerStatuses...) {
		if isContainerActive(s) {
			activeContainers = append(activeContainers, s)
		}
	}

	if len(activeContainers) == 0 {
		a.addSandboxToKill(sandboxStatus, "all containers terminated", false)
	}

	// (2).
	//
	// If all active containers will be killed, kill the sandbox as well.
	// This happens when all contaieners are unhealthy, and the pod's restart policy is "Never".
	for _, s := range activeContainers {
		if a.findContainerKillInfoByID(s.ID) == nil {
			return
		}
	}

	a.addSandboxToKill(sandboxStatus, "all containers will be killed", false)
	return
}

// computeSyncAction generates the sync action object that contains the actions
// we need to take in the current sync pod iteration.
// The whole computation is splitted into 3 phases:
// - (1) killing phase computation, which constructs a list of sandboxs and containers that needs to be killed.
// - (2) starting phase computation, which constructs the sandbox and a list containers that needs to be started.
// - (3) sandbox cleanup phase computation, which checks if we need to clean up the sandbox.
func (a *syncAction) computeSyncAction(pod *v1.Pod, statuses *statusGroup, livenessManager proberesults.Manager) {
	// (1) compute the sandboxes, and containers that we need to kill.
	a.computeSyncActionKillingPhase(statuses, livenessManager)

	// (2) compute the sandbox, and containers that we need to start.
	a.computeSyncActionStartingPhase(statuses)

	// (3) do additional check to make sure we kill the sandbox when
	//     all containers are exited (or will be killed), and none of them will
	//     be restarted.
	a.computeSyncActionSandboxCleanup(statuses)

	return
}
