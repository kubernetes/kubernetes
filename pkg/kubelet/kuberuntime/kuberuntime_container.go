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
	"io"
	"io/ioutil"
	"math/rand"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	kubetypes "k8s.io/kubernetes/pkg/types"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/selinux"
	"k8s.io/kubernetes/pkg/util/sets"
)

// startContainer starts a container and returns a message indicates why it is failed on error.
// It starts the container through the following steps:
// * pull the image
// * create the container
// * start the container
// * run the post start lifecycle hooks (if applicable)
func (m *kubeGenericRuntimeManager) startContainer(podSandboxID string, podSandboxConfig *runtimeapi.PodSandboxConfig, container *v1.Container, pod *v1.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []v1.Secret, podIP string) (string, error) {
	// Step 1: pull the image.
	err, msg := m.imagePuller.EnsureImageExists(pod, container, pullSecrets)
	if err != nil {
		return msg, err
	}

	// Step 2: create the container.
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Can't make a ref to pod %q, container %v: %v", format.Pod(pod), container.Name, err)
	}
	glog.V(4).Infof("Generating ref for container %s: %#v", container.Name, ref)

	// For a new container, the RestartCount should be 0
	restartCount := 0
	containerStatus := podStatus.FindContainerStatusByName(container.Name)
	if containerStatus != nil {
		restartCount = containerStatus.RestartCount + 1
	}

	containerConfig, err := m.generateContainerConfig(container, pod, restartCount, podIP)
	if err != nil {
		m.recorder.Eventf(ref, v1.EventTypeWarning, events.FailedToCreateContainer, "Failed to create container with error: %v", err)
		return "Generate Container Config Failed", err
	}
	containerID, err := m.runtimeService.CreateContainer(podSandboxID, containerConfig, podSandboxConfig)
	if err != nil {
		m.recorder.Eventf(ref, v1.EventTypeWarning, events.FailedToCreateContainer, "Failed to create container with error: %v", err)
		return "Create Container Failed", err
	}
	m.recorder.Eventf(ref, v1.EventTypeNormal, events.CreatedContainer, "Created container with id %v", containerID)
	if ref != nil {
		m.containerRefManager.SetRef(kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   containerID,
		}, ref)
	}

	// Step 3: start the container.
	err = m.runtimeService.StartContainer(containerID)
	if err != nil {
		m.recorder.Eventf(ref, v1.EventTypeWarning, events.FailedToStartContainer,
			"Failed to start container with id %v with error: %v", containerID, err)
		return "Start Container Failed", err
	}
	m.recorder.Eventf(ref, v1.EventTypeNormal, events.StartedContainer, "Started container with id %v", containerID)

	// Symlink container logs to the legacy container log location for cluster logging
	// support.
	// TODO(random-liu): Remove this after cluster logging supports CRI container log path.
	containerMeta := containerConfig.GetMetadata()
	sandboxMeta := podSandboxConfig.GetMetadata()
	legacySymlink := legacyLogSymlink(containerID, containerMeta.GetName(), sandboxMeta.GetName(),
		sandboxMeta.GetNamespace())
	containerLog := filepath.Join(podSandboxConfig.GetLogDirectory(), containerConfig.GetLogPath())
	if err := m.osInterface.Symlink(containerLog, legacySymlink); err != nil {
		glog.Errorf("Failed to create legacy symbolic link %q to container %q log %q: %v",
			legacySymlink, containerID, containerLog, err)
	}

	// Step 4: execute the post start hook.
	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		kubeContainerID := kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   containerID,
		}
		msg, handlerErr := m.runner.Run(kubeContainerID, pod, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			err := fmt.Errorf("PostStart handler: %v", handlerErr)
			m.generateContainerEvent(kubeContainerID, v1.EventTypeWarning, events.FailedPostStartHook, msg)
			m.killContainer(pod, kubeContainerID, container.Name, "FailedPostStartHook", nil)
			return "PostStart Hook Failed", err
		}
	}

	return "", nil
}

// generateContainerConfig generates container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateContainerConfig(container *v1.Container, pod *v1.Pod, restartCount int, podIP string) (*runtimeapi.ContainerConfig, error) {
	opts, err := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP)
	if err != nil {
		return nil, err
	}

	uid, username, err := m.getImageUser(container.Image)
	if err != nil {
		return nil, err
	}
	if uid != nil {
		// Verify RunAsNonRoot. Non-root verification only supports numeric user.
		if err := verifyRunAsNonRoot(pod, container, *uid); err != nil {
			return nil, err
		}
	} else {
		glog.Warningf("Non-root verification doesn't support non-numeric user (%s)", *username)
	}

	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)
	containerLogsPath := buildContainerLogsPath(container.Name, restartCount)
	restartCountUint32 := uint32(restartCount)
	config := &runtimeapi.ContainerConfig{
		Metadata: &runtimeapi.ContainerMetadata{
			Name:    &container.Name,
			Attempt: &restartCountUint32,
		},
		Image:       &runtimeapi.ImageSpec{Image: &container.Image},
		Command:     command,
		Args:        args,
		WorkingDir:  &container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount),
		Devices:     makeDevices(opts),
		Mounts:      m.makeMounts(opts, container),
		LogPath:     &containerLogsPath,
		Stdin:       &container.Stdin,
		StdinOnce:   &container.StdinOnce,
		Tty:         &container.TTY,
		Linux:       m.generateLinuxContainerConfig(container, pod, uid, username),
	}

	// set environment variables
	envs := make([]*runtimeapi.KeyValue, len(opts.Envs))
	for idx := range opts.Envs {
		e := opts.Envs[idx]
		envs[idx] = &runtimeapi.KeyValue{
			Key:   &e.Name,
			Value: &e.Value,
		}
	}
	config.Envs = envs

	return config, nil
}

// generateLinuxContainerConfig generates linux container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateLinuxContainerConfig(container *v1.Container, pod *v1.Pod, uid *int64, username *string) *runtimeapi.LinuxContainerConfig {
	lc := &runtimeapi.LinuxContainerConfig{
		Resources:       &runtimeapi.LinuxContainerResources{},
		SecurityContext: m.determineEffectiveSecurityContext(pod, container, uid, username),
	}

	// set linux container resources
	var cpuShares int64
	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	memoryLimit := container.Resources.Limits.Memory().Value()
	oomScoreAdj := int64(qos.GetContainerOOMScoreAdjust(pod, container,
		int64(m.machineInfo.MemoryCapacity)))
	// If request is not specified, but limit is, we want request to default to limit.
	// API server does this for new containers, but we repeat this logic in Kubelet
	// for containers running on existing Kubernetes clusters.
	if cpuRequest.IsZero() && !cpuLimit.IsZero() {
		cpuShares = milliCPUToShares(cpuLimit.MilliValue())
	} else {
		// if cpuRequest.Amount is nil, then milliCPUToShares will return the minimal number
		// of CPU shares.
		cpuShares = milliCPUToShares(cpuRequest.MilliValue())
	}
	lc.Resources.CpuShares = &cpuShares
	if memoryLimit != 0 {
		lc.Resources.MemoryLimitInBytes = &memoryLimit
	}
	// Set OOM score of the container based on qos policy. Processes in lower-priority pods should
	// be killed first if the system runs out of memory.
	lc.Resources.OomScoreAdj = &oomScoreAdj

	if m.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned
		// to allow full usage of cpu resource.
		cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())
		lc.Resources.CpuQuota = &cpuQuota
		lc.Resources.CpuPeriod = &cpuPeriod
	}

	return lc
}

// makeDevices generates container devices for kubelet runtime v1.
func makeDevices(opts *kubecontainer.RunContainerOptions) []*runtimeapi.Device {
	devices := make([]*runtimeapi.Device, len(opts.Devices))

	for idx := range opts.Devices {
		device := opts.Devices[idx]
		devices[idx] = &runtimeapi.Device{
			HostPath:      &device.PathOnHost,
			ContainerPath: &device.PathInContainer,
			Permissions:   &device.Permissions,
		}
	}

	return devices
}

// makeMounts generates container volume mounts for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) makeMounts(opts *kubecontainer.RunContainerOptions, container *v1.Container) []*runtimeapi.Mount {
	volumeMounts := []*runtimeapi.Mount{}

	for idx := range opts.Mounts {
		v := opts.Mounts[idx]
		selinuxRelabel := v.SELinuxRelabel && selinux.SELinuxEnabled()
		mount := &runtimeapi.Mount{
			HostPath:       &v.HostPath,
			ContainerPath:  &v.ContainerPath,
			Readonly:       &v.ReadOnly,
			SelinuxRelabel: &selinuxRelabel,
		}

		volumeMounts = append(volumeMounts, mount)
	}

	// The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
	if opts.PodContainerDir != "" && len(container.TerminationMessagePath) != 0 {
		// Because the PodContainerDir contains pod uid and container name which is unique enough,
		// here we just add a random id to make the path unique for different instances
		// of the same container.
		cid := makeUID()
		containerLogPath := filepath.Join(opts.PodContainerDir, cid)
		fs, err := m.osInterface.Create(containerLogPath)
		if err != nil {
			glog.Errorf("Error on creating termination-log file %q: %v", containerLogPath, err)
		} else {
			fs.Close()
			selinuxRelabel := selinux.SELinuxEnabled()
			volumeMounts = append(volumeMounts, &runtimeapi.Mount{
				HostPath:       &containerLogPath,
				ContainerPath:  &container.TerminationMessagePath,
				SelinuxRelabel: &selinuxRelabel,
			})
		}
	}

	return volumeMounts
}

// getKubeletContainers lists containers managed by kubelet.
// The boolean parameter specifies whether returns all containers including
// those already exited and dead containers (used for garbage collection).
func (m *kubeGenericRuntimeManager) getKubeletContainers(allContainers bool) ([]*runtimeapi.Container, error) {
	filter := &runtimeapi.ContainerFilter{
		LabelSelector: map[string]string{kubernetesManagedLabel: "true"},
	}
	if !allContainers {
		runningState := runtimeapi.ContainerState_CONTAINER_RUNNING
		filter.State = &runningState
	}

	containers, err := m.getContainersHelper(filter)
	if err != nil {
		glog.Errorf("getKubeletContainers failed: %v", err)
		return nil, err
	}

	return containers, nil
}

// getContainers lists containers by filter.
func (m *kubeGenericRuntimeManager) getContainersHelper(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	resp, err := m.runtimeService.ListContainers(filter)
	if err != nil {
		return nil, err
	}

	return resp, err
}

// makeUID returns a randomly generated string.
func makeUID() string {
	return fmt.Sprintf("%08x", rand.Uint32())
}

// getTerminationMessage gets termination message of the container.
func getTerminationMessage(status *runtimeapi.ContainerStatus, kubeStatus *kubecontainer.ContainerStatus, terminationMessagePath string) string {
	message := ""

	if !kubeStatus.FinishedAt.IsZero() || kubeStatus.ExitCode != 0 {
		if terminationMessagePath == "" {
			return ""
		}

		for _, mount := range status.Mounts {
			if mount.GetContainerPath() == terminationMessagePath {
				path := mount.GetHostPath()
				if data, err := ioutil.ReadFile(path); err != nil {
					message = fmt.Sprintf("Error on reading termination-log %s: %v", path, err)
				} else {
					message = string(data)
				}
				break
			}
		}
	}

	return message
}

// getPodContainerStatuses gets all containers' statuses for the pod.
func (m *kubeGenericRuntimeManager) getPodContainerStatuses(uid kubetypes.UID, name, namespace string) ([]*kubecontainer.ContainerStatus, error) {
	// Select all containers of the given pod.
	containers, err := m.runtimeService.ListContainers(&runtimeapi.ContainerFilter{
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: string(uid)},
	})
	if err != nil {
		glog.Errorf("ListContainers error: %v", err)
		return nil, err
	}

	statuses := make([]*kubecontainer.ContainerStatus, len(containers))
	// TODO: optimization: set maximum number of containers per container name to examine.
	for i, c := range containers {
		status, err := m.runtimeService.ContainerStatus(c.GetId())
		if err != nil {
			glog.Errorf("ContainerStatus for %s error: %v", c.GetId(), err)
			return nil, err
		}

		annotatedInfo := getContainerInfoFromAnnotations(c.Annotations)
		labeledInfo := getContainerInfoFromLabels(c.Labels)
		cStatus := &kubecontainer.ContainerStatus{
			ID: kubecontainer.ContainerID{
				Type: m.runtimeName,
				ID:   c.GetId(),
			},
			Name:         labeledInfo.ContainerName,
			Image:        status.Image.GetImage(),
			ImageID:      status.GetImageRef(),
			Hash:         annotatedInfo.Hash,
			RestartCount: annotatedInfo.RestartCount,
			State:        toKubeContainerState(c.GetState()),
			CreatedAt:    time.Unix(0, status.GetCreatedAt()),
		}

		if c.GetState() == runtimeapi.ContainerState_CONTAINER_RUNNING {
			cStatus.StartedAt = time.Unix(0, status.GetStartedAt())
		} else {
			cStatus.Reason = status.GetReason()
			cStatus.Message = status.GetMessage()
			cStatus.ExitCode = int(status.GetExitCode())
			cStatus.FinishedAt = time.Unix(0, status.GetFinishedAt())
		}

		tMessage := getTerminationMessage(status, cStatus, annotatedInfo.TerminationMessagePath)
		// Use the termination message written by the application is not empty
		if len(tMessage) != 0 {
			cStatus.Message = tMessage
		}
		statuses[i] = cStatus
	}

	sort.Sort(containerStatusByCreated(statuses))
	return statuses, nil
}

// generateContainerEvent generates an event for the container.
func (m *kubeGenericRuntimeManager) generateContainerEvent(containerID kubecontainer.ContainerID, eventType, reason, message string) {
	ref, ok := m.containerRefManager.GetRef(containerID)
	if !ok {
		glog.Warningf("No ref for container %q", containerID)
		return
	}
	m.recorder.Event(ref, eventType, reason, message)
}

// executePreStopHook runs the pre-stop lifecycle hooks if applicable and returns the duration it takes.
func (m *kubeGenericRuntimeManager) executePreStopHook(pod *v1.Pod, containerID kubecontainer.ContainerID, containerSpec *v1.Container, gracePeriod int64) int64 {
	glog.V(3).Infof("Running preStop hook for container %q", containerID.String())

	start := metav1.Now()
	done := make(chan struct{})
	go func() {
		defer close(done)
		defer utilruntime.HandleCrash()
		if msg, err := m.runner.Run(containerID, pod, containerSpec, containerSpec.Lifecycle.PreStop); err != nil {
			glog.Errorf("preStop hook for container %q failed: %v", containerSpec.Name, err)
			m.generateContainerEvent(containerID, v1.EventTypeWarning, events.FailedPreStopHook, msg)
		}
	}()

	select {
	case <-time.After(time.Duration(gracePeriod) * time.Second):
		glog.V(2).Infof("preStop hook for container %q did not complete in %d seconds", containerID, gracePeriod)
	case <-done:
		glog.V(3).Infof("preStop hook for container %q completed", containerID)
	}

	return int64(metav1.Now().Sub(start.Time).Seconds())
}

// restoreSpecsFromContainerLabels restores all information needed for killing a container. In some
// case we may not have pod and container spec when killing a container, e.g. pod is deleted during
// kubelet restart.
// To solve this problem, we've already written necessary information into container labels. Here we
// just need to retrieve them from container labels and restore the specs.
// TODO(random-liu): Add a node e2e test to test this behaviour.
// TODO(random-liu): Change the lifecycle handler to just accept information needed, so that we can
// just pass the needed function not create the fake object.
func (m *kubeGenericRuntimeManager) restoreSpecsFromContainerLabels(containerID kubecontainer.ContainerID) (*v1.Pod, *v1.Container, error) {
	var pod *v1.Pod
	var container *v1.Container
	s, err := m.runtimeService.ContainerStatus(containerID.ID)
	if err != nil {
		return nil, nil, err
	}

	l := getContainerInfoFromLabels(s.Labels)
	a := getContainerInfoFromAnnotations(s.Annotations)
	// Notice that the followings are not full spec. The container killing code should not use
	// un-restored fields.
	pod = &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			UID:                        l.PodUID,
			Name:                       l.PodName,
			Namespace:                  l.PodNamespace,
			DeletionGracePeriodSeconds: a.PodDeletionGracePeriod,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: a.PodTerminationGracePeriod,
		},
	}
	container = &v1.Container{
		Name:  l.ContainerName,
		Ports: a.ContainerPorts,
		TerminationMessagePath: a.TerminationMessagePath,
	}
	if a.PreStopHandler != nil {
		container.Lifecycle = &v1.Lifecycle{
			PreStop: a.PreStopHandler,
		}
	}
	return pod, container, nil
}

// killContainer kills a container through the following steps:
// * Run the pre-stop lifecycle hooks (if applicable).
// * Stop the container.
func (m *kubeGenericRuntimeManager) killContainer(pod *v1.Pod, containerID kubecontainer.ContainerID, containerName string, reason string, gracePeriodOverride *int64) error {
	var containerSpec *v1.Container
	if pod != nil {
		containerSpec = kubecontainer.GetContainerSpec(pod, containerName)
	} else {
		// Restore necessary information if one of the specs is nil.
		restoredPod, restoredContainer, err := m.restoreSpecsFromContainerLabels(containerID)
		if err != nil {
			return err
		}
		pod, containerSpec = restoredPod, restoredContainer
	}
	// From this point , pod and container must be non-nil.
	gracePeriod := int64(minimumGracePeriodInSeconds)
	switch {
	case pod.DeletionGracePeriodSeconds != nil:
		gracePeriod = *pod.DeletionGracePeriodSeconds
	case pod.Spec.TerminationGracePeriodSeconds != nil:
		gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
	}

	glog.V(2).Infof("Killing container %q with %d second grace period", containerID.String(), gracePeriod)

	// Run the pre-stop lifecycle hooks if applicable.
	if containerSpec.Lifecycle != nil && containerSpec.Lifecycle.PreStop != nil {
		gracePeriod = gracePeriod - m.executePreStopHook(pod, containerID, containerSpec, gracePeriod)
	}
	// always give containers a minimal shutdown window to avoid unnecessary SIGKILLs
	if gracePeriod < minimumGracePeriodInSeconds {
		gracePeriod = minimumGracePeriodInSeconds
	}
	if gracePeriodOverride != nil {
		gracePeriod = *gracePeriodOverride
		glog.V(3).Infof("Killing container %q, but using %d second grace period override", containerID, gracePeriod)
	}

	err := m.runtimeService.StopContainer(containerID.ID, gracePeriod)
	if err != nil {
		glog.Errorf("Container %q termination failed with gracePeriod %d: %v", containerID.String(), gracePeriod, err)
	} else {
		glog.V(3).Infof("Container %q exited normally", containerID.String())
	}

	message := fmt.Sprintf("Killing container with id %s", containerID.String())
	if reason != "" {
		message = fmt.Sprint(message, ":", reason)
	}
	m.generateContainerEvent(containerID, v1.EventTypeNormal, events.KillingContainer, message)
	m.containerRefManager.ClearRef(containerID)

	return err
}

// killContainersWithSyncResult kills all pod's containers with sync results.
func (m *kubeGenericRuntimeManager) killContainersWithSyncResult(pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) (syncResults []*kubecontainer.SyncResult) {
	containerResults := make(chan *kubecontainer.SyncResult, len(runningPod.Containers))
	wg := sync.WaitGroup{}

	wg.Add(len(runningPod.Containers))
	for _, container := range runningPod.Containers {
		go func(container *kubecontainer.Container) {
			defer utilruntime.HandleCrash()
			defer wg.Done()

			killContainerResult := kubecontainer.NewSyncResult(kubecontainer.KillContainer, container.Name)
			if err := m.killContainer(pod, container.ID, container.Name, "Need to kill Pod", gracePeriodOverride); err != nil {
				killContainerResult.Fail(kubecontainer.ErrKillContainer, err.Error())
			}
			containerResults <- killContainerResult
		}(container)
	}
	wg.Wait()
	close(containerResults)

	for containerResult := range containerResults {
		syncResults = append(syncResults, containerResult)
	}
	return
}

// pruneInitContainers ensures that before we begin creating init containers, we have reduced the number
// of outstanding init containers still present. This reduces load on the container garbage collector
// by only preserving the most recent terminated init container.
func (m *kubeGenericRuntimeManager) pruneInitContainersBeforeStart(pod *v1.Pod, podStatus *kubecontainer.PodStatus, initContainersToKeep map[kubecontainer.ContainerID]int) {
	// only the last execution of each init container should be preserved, and only preserve it if it is in the
	// list of init containers to keep.
	initContainerNames := sets.NewString()
	for _, container := range pod.Spec.InitContainers {
		initContainerNames.Insert(container.Name)
	}
	for name := range initContainerNames {
		count := 0
		for _, status := range podStatus.ContainerStatuses {
			if status.Name != name || !initContainerNames.Has(status.Name) || status.State != kubecontainer.ContainerStateExited {
				continue
			}
			count++
			// keep the first init container for this name
			if count == 1 {
				continue
			}
			// if there is a reason to preserve the older container, do so
			if _, ok := initContainersToKeep[status.ID]; ok {
				continue
			}

			// prune all other init containers that match this container name
			glog.V(4).Infof("Removing init container %q instance %q %d", status.Name, status.ID.ID, count)
			if err := m.runtimeService.RemoveContainer(status.ID.ID); err != nil {
				utilruntime.HandleError(fmt.Errorf("failed to remove pod init container %q: %v; Skipping pod %q", status.Name, err, format.Pod(pod)))
				continue
			}

			// remove any references to this container
			if _, ok := m.containerRefManager.GetRef(status.ID); ok {
				m.containerRefManager.ClearRef(status.ID)
			} else {
				glog.Warningf("No ref for container %q", status.ID)
			}
		}
	}
}

// findNextInitContainerToRun returns the status of the last failed container, the
// next init container to start, or done if there are no further init containers.
// Status is only returned if an init container is failed, in which case next will
// point to the current container.
func findNextInitContainerToRun(pod *v1.Pod, podStatus *kubecontainer.PodStatus) (status *kubecontainer.ContainerStatus, next *v1.Container, done bool) {
	if len(pod.Spec.InitContainers) == 0 {
		return nil, nil, true
	}

	// If there are failed containers, return the status of the last failed one.
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		container := &pod.Spec.InitContainers[i]
		status := podStatus.FindContainerStatusByName(container.Name)
		if status != nil && isContainerFailed(status) {
			return status, container, false
		}
	}

	// There are no failed containers now.
	for i := len(pod.Spec.InitContainers) - 1; i >= 0; i-- {
		container := &pod.Spec.InitContainers[i]
		status := podStatus.FindContainerStatusByName(container.Name)
		if status == nil {
			continue
		}

		// container is still running, return not done.
		if status.State == kubecontainer.ContainerStateRunning {
			return nil, nil, false
		}

		if status.State == kubecontainer.ContainerStateExited {
			// all init containers successful
			if i == (len(pod.Spec.InitContainers) - 1) {
				return nil, nil, true
			}

			// all containers up to i successful, go to i+1
			return nil, &pod.Spec.InitContainers[i+1], false
		}
	}

	return nil, &pod.Spec.InitContainers[0], false
}

// GetContainerLogs returns logs of a specific container.
func (m *kubeGenericRuntimeManager) GetContainerLogs(pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) (err error) {
	status, err := m.runtimeService.ContainerStatus(containerID.ID)
	if err != nil {
		return fmt.Errorf("failed to get container status %q: %v", containerID, err)
	}
	labeledInfo := getContainerInfoFromLabels(status.Labels)
	annotatedInfo := getContainerInfoFromAnnotations(status.Annotations)
	path := buildFullContainerLogsPath(pod.UID, labeledInfo.ContainerName, annotatedInfo.RestartCount)
	return ReadLogs(path, logOptions, stdout, stderr)
}

// GetExec gets the endpoint the runtime will serve the exec request from.
func (m *kubeGenericRuntimeManager) GetExec(id kubecontainer.ContainerID, cmd []string, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	req := &runtimeapi.ExecRequest{
		ContainerId: &id.ID,
		Cmd:         cmd,
		Tty:         &tty,
		Stdin:       &stdin,
	}
	resp, err := m.runtimeService.Exec(req)
	if err != nil {
		return nil, err
	}

	return url.Parse(resp.GetUrl())
}

// GetAttach gets the endpoint the runtime will serve the attach request from.
func (m *kubeGenericRuntimeManager) GetAttach(id kubecontainer.ContainerID, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	req := &runtimeapi.AttachRequest{
		ContainerId: &id.ID,
		Stdin:       &stdin,
		Tty:         &tty,
	}
	resp, err := m.runtimeService.Attach(req)
	if err != nil {
		return nil, err
	}
	return url.Parse(resp.GetUrl())
}

// RunInContainer synchronously executes the command in the container, and returns the output.
func (m *kubeGenericRuntimeManager) RunInContainer(id kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	stdout, stderr, err := m.runtimeService.ExecSync(id.ID, cmd, 0)
	// NOTE(timstclair): This does not correctly interleave stdout & stderr, but should be sufficient
	// for logging purposes. A combined output option will need to be added to the ExecSyncRequest
	// if more precise output ordering is ever required.
	return append(stdout, stderr...), err
}

// removeContainer removes the container and the container logs.
// Notice that we remove the container logs first, so that container will not be removed if
// container logs are failed to be removed, and kubelet will retry this later. This guarantees
// that container logs to be removed with the container.
// Notice that we assume that the container should only be removed in non-running state, and
// it will not write container logs anymore in that state.
func (m *kubeGenericRuntimeManager) removeContainer(containerID string) error {
	glog.V(4).Infof("Removing container %q", containerID)
	// Remove the container log.
	// TODO: Separate log and container lifecycle management.
	if err := m.removeContainerLog(containerID); err != nil {
		return err
	}
	// Remove the container.
	return m.runtimeService.RemoveContainer(containerID)
}

// removeContainerLog removes the container log.
func (m *kubeGenericRuntimeManager) removeContainerLog(containerID string) error {
	// Remove the container log.
	status, err := m.runtimeService.ContainerStatus(containerID)
	if err != nil {
		return fmt.Errorf("failed to get container status %q: %v", containerID, err)
	}
	labeledInfo := getContainerInfoFromLabels(status.Labels)
	annotatedInfo := getContainerInfoFromAnnotations(status.Annotations)
	path := buildFullContainerLogsPath(labeledInfo.PodUID, labeledInfo.ContainerName, annotatedInfo.RestartCount)
	if err := m.osInterface.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove container %q log %q: %v", containerID, path, err)
	}

	// Remove the legacy container log symlink.
	// TODO(random-liu): Remove this after cluster logging supports CRI container log path.
	legacySymlink := legacyLogSymlink(containerID, labeledInfo.ContainerName, labeledInfo.PodName,
		labeledInfo.PodNamespace)
	if err := m.osInterface.Remove(legacySymlink); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to remove container %q log legacy symbolic link %q: %v",
			containerID, legacySymlink, err)
	}
	return nil
}

// DeleteContainer removes a container.
func (m *kubeGenericRuntimeManager) DeleteContainer(containerID kubecontainer.ContainerID) error {
	return m.removeContainer(containerID.ID)
}
