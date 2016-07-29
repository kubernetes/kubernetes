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
	"io/ioutil"
	"os"
	"path"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

// getContainerLogsPath gets log path for container.
func getContainerLogsPath(containerName, podUID string) string {
	return path.Join(podLogsRootDirectory, podUID, fmt.Sprintf("%s.log", containerName))
}

// generateContainerConfig generates container config for kubelet runtime api.
func (m *kubeGenericRuntimeManager) generateContainerConfig(container *api.Container, pod *api.Pod, restartCount int, podIP string) (*runtimeApi.ContainerConfig, error) {
	opts, err := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP)
	if err != nil {
		return nil, err
	}

	_, containerName, cid := buildContainerName(pod.Name, pod.Namespace, string(pod.UID), container)
	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)
	containerLogsPath := getContainerLogsPath(containerName, string(pod.UID))
	podHasSELinuxLabel := pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxOptions != nil
	config := &runtimeApi.ContainerConfig{
		Name:        &containerName,
		Image:       &runtimeApi.ImageSpec{Image: &container.Image},
		Command:     command,
		Args:        args,
		WorkingDir:  &container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount),
		Mounts:      makeMounts(cid, opts, container, podHasSELinuxLabel),
		LogPath:     &containerLogsPath,
		Stdin:       &container.Stdin,
		StdinOnce:   &container.StdinOnce,
		Tty:         &container.TTY,
	}

	memoryLimit := container.Resources.Limits.Memory().Value()
	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	var cpuShares int64
	if cpuRequest.IsZero() && !cpuLimit.IsZero() {
		cpuShares = milliCPUToShares(cpuLimit.MilliValue())
	} else {
		// if cpuRequest.Amount is nil, then milliCPUToShares will return the minimal number
		// of CPU shares.
		cpuShares = milliCPUToShares(cpuRequest.MilliValue())
	}
	if cpuShares != 0 || memoryLimit != 0 || m.cpuCFSQuota {
		linuxResource := &runtimeApi.LinuxContainerResources{}
		if cpuShares != 0 {
			linuxResource.CpuShares = &cpuShares
		}
		if memoryLimit != 0 {
			linuxResource.MemoryLimitInBytes = &memoryLimit
		}
		if m.cpuCFSQuota {
			// if cpuLimit.Amount is nil, then the appropriate default value is returned
			// to allow full usage of cpu resource.
			cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())
			linuxResource.CpuQuota = &cpuQuota
			linuxResource.CpuPeriod = &cpuPeriod
		}

		config.Linux = &runtimeApi.LinuxContainerConfig{
			Resources: linuxResource,
		}
	}

	if container.SecurityContext != nil {
		securityContext := container.SecurityContext
		if securityContext.Privileged != nil {
			config.Privileged = securityContext.Privileged
		}

		if securityContext.ReadOnlyRootFilesystem != nil {
			config.ReadonlyRootfs = securityContext.ReadOnlyRootFilesystem
		}

		if securityContext.Capabilities != nil {
			if config.Linux == nil {
				config.Linux = &runtimeApi.LinuxContainerConfig{
					Capabilities: &runtimeApi.Capability{
						AddCapabilities:  make([]string, 0, len(securityContext.Capabilities.Add)),
						DropCapabilities: make([]string, 0, len(securityContext.Capabilities.Drop)),
					},
				}
			}

			for index, value := range securityContext.Capabilities.Add {
				config.Linux.Capabilities.AddCapabilities[index] = string(value)
			}
			for index, value := range securityContext.Capabilities.Drop {
				config.Linux.Capabilities.DropCapabilities[index] = string(value)
			}
		}

		if securityContext.SELinuxOptions != nil {
			if config.Linux == nil {
				config.Linux = &runtimeApi.LinuxContainerConfig{}
			}
			config.Linux.SelinuxOptions = &runtimeApi.SELinuxOption{
				User:  &securityContext.SELinuxOptions.User,
				Role:  &securityContext.SELinuxOptions.Role,
				Type:  &securityContext.SELinuxOptions.Type,
				Level: &securityContext.SELinuxOptions.Level,
			}
		}
	}

	envs := make([]*runtimeApi.KeyValue, len(opts.Envs))
	for index, e := range opts.Envs {
		envs[index] = &runtimeApi.KeyValue{
			Key:   &e.Name,
			Value: &e.Value,
		}
	}
	config.Envs = envs

	return config, nil
}

// startContainer starts a container through the following steps:
// * Pull the image
// * Create the container
// * Start the container
// * Run the post start lifecycle hooks (if applicable)
func (m *kubeGenericRuntimeManager) startContainer(podSandboxID string, podSandboxConfig *runtimeApi.PodSandboxConfig, container *api.Container, pod *api.Pod, podStatus *kubecontainer.PodStatus, pullSecrets []api.Secret, podIP string) (string, error) {
	err, msg := m.imagePuller.EnsureImageExists(pod, container, pullSecrets)
	if err != nil {
		return msg, err
	}

	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Can't make a ref to pod %q, container %v: '%v'", format.Pod(pod), container.Name, err)
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
		m.recorder.Eventf(ref, api.EventTypeWarning, events.FailedToCreateContainer, "Failed to create container with error: %v", err)
		return "Generate Container Config Failed", err
	}

	containerID, err := m.runtimeService.CreateContainer(podSandboxID, containerConfig, podSandboxConfig)
	if err != nil {
		m.recorder.Eventf(ref, api.EventTypeWarning, events.FailedToCreateContainer, "Failed to create container with error: %v", err)
		return "Create Container Failed", err
	}
	m.recorder.Eventf(ref, api.EventTypeNormal, events.CreatedContainer, "Created container with id %v", containerID)

	if ref != nil {
		m.containerRefManager.SetRef(kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   containerID,
		}, ref)
	}

	err = m.runtimeService.StartContainer(containerID)
	if err != nil {
		m.recorder.Eventf(ref, api.EventTypeWarning, events.FailedToStartContainer,
			"Failed to start container with id %v with error: %v", containerID, err)
		return "Start Container Failed", err
	}
	m.recorder.Eventf(ref, api.EventTypeNormal, events.StartedContainer, "Started container with id %v", containerID)

	if container.Lifecycle != nil && container.Lifecycle.PostStart != nil {
		kubeContainerID := kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   containerID,
		}
		msg, handlerErr := m.runner.Run(kubeContainerID, pod, container, container.Lifecycle.PostStart)
		if handlerErr != nil {
			err := fmt.Errorf("PostStart handler: %v", handlerErr)
			m.generateFailedContainerEvent(kubeContainerID, pod.Name, events.FailedPostStartHook, msg)
			m.killContainer(pod, kubeContainerID, container, "FailedPostStartHook", nil)
			return "PostStart Hook Failed", err
		}
	}

	return "", nil
}

// makeMounts generates container volume mounts for kubelet runtime api.
func makeMounts(cid string, opts *kubecontainer.RunContainerOptions, container *api.Container, podHasSELinuxLabel bool) []*runtimeApi.Mount {
	volumeMounts := []*runtimeApi.Mount{}

	for _, v := range opts.Mounts {
		m := &runtimeApi.Mount{
			Name:          &v.Name,
			HostPath:      &v.HostPath,
			ContainerPath: &v.ContainerPath,
			Readonly:      &v.ReadOnly,
		}
		if podHasSELinuxLabel && v.SELinuxRelabel {
			m.SelinuxRelabel = &v.SELinuxRelabel
		}

		volumeMounts = append(volumeMounts, m)
	}

	// The reason we create and mount the log file in here (not in kubelet) is because
	// the file's location depends on the ID of the container, and we need to create and
	// mount the file before actually starting the container.
	if opts.PodContainerDir != "" && len(container.TerminationMessagePath) != 0 {
		// Because the PodContainerDir contains pod uid and container name which is unique enough,
		// here we just add an unique container id to make the path unique for different instances
		// of the same container.
		containerLogPath := path.Join(opts.PodContainerDir, cid)
		fs, err := os.Create(containerLogPath)
		if err != nil {
			glog.Errorf("Error on creating termination-log file %q: %v", containerLogPath, err)
		} else {
			fs.Close()
			volumeMounts = append(volumeMounts, &runtimeApi.Mount{
				HostPath:      &containerLogPath,
				ContainerPath: &container.TerminationMessagePath,
			})
		}
	}

	return volumeMounts
}

// getKubeletContainerStatuses gets all containers' status for the pod sandbox.
func (m *kubeGenericRuntimeManager) getKubeletContainerStatuses(podSandboxID string) ([]*kubecontainer.ContainerStatus, error) {
	containers, err := m.runtimeService.ListContainers(&runtimeApi.ContainerFilter{
		PodSandboxId: &podSandboxID,
	})
	if err != nil {
		glog.Errorf("ListContainers error: %v", err)
		return nil, err
	}

	statuses := make([]*kubecontainer.ContainerStatus, len(containers))
	for i, c := range containers {
		status, err := m.runtimeService.ContainerStatus(c.GetId())
		if err != nil {
			glog.Errorf("ContainerStatus for %s error: %v", c.GetId(), err)
			return nil, err
		}

		_, _, _, cName, hash, err := parseContainerName(c.GetName())
		if err != nil {
			glog.V(3).Infof("%s container %s is not managed by kubelet", m.runtimeName, c.GetName())
			continue
		}

		annotatedInfo := getContainerInfoFromAnnotations(c.Annotations)
		cStatus := &kubecontainer.ContainerStatus{
			ID: kubecontainer.ContainerID{
				Type: m.runtimeName,
				ID:   c.GetId(),
			},
			Name:         cName,
			State:        toKubeContainerState(c.GetState()),
			Image:        status.Image.GetImage(),
			ImageID:      status.GetImageRef(),
			Hash:         hash,
			RestartCount: annotatedInfo.RestartCount,
		}
		if status.Reason != nil {
			cStatus.Reason = status.GetReason()
		}
		if status.ExitCode != nil {
			cStatus.ExitCode = int(status.GetExitCode())
		}
		if status.CreatedAt != nil {
			cStatus.CreatedAt = time.Unix(status.GetCreatedAt(), 0)
		}
		if status.StartedAt != nil {
			cStatus.StartedAt = time.Unix(status.GetStartedAt(), 0)
		}
		if status.FinishedAt != nil {
			cStatus.FinishedAt = time.Unix(status.GetFinishedAt(), 0)
		}

		message := ""
		if !cStatus.FinishedAt.IsZero() || cStatus.ExitCode != 0 {
			if annotatedInfo.TerminationMessagePath != "" {
				for _, mount := range status.Mounts {
					if mount.GetContainerPath() == annotatedInfo.TerminationMessagePath {
						path := mount.GetHostPath()
						if data, err := ioutil.ReadFile(path); err != nil {
							message = fmt.Sprintf("Error on reading termination-log %s: %v", path, err)
						} else {
							message = string(data)
						}
					}
				}
			}
		}
		cStatus.Message = message
		statuses[i] = cStatus
	}

	return statuses, nil
}

// getKubeletContainers lists all (or just the running) containers managed by kubelet.
func (m *kubeGenericRuntimeManager) getKubeletContainers(allContainers bool) ([]*runtimeApi.Container, error) {
	var resp []*runtimeApi.Container
	var err error

	if allContainers {
		resp, err = m.runtimeService.ListContainers(nil)
	} else {
		runningState := runtimeApi.ContainerState_RUNNING
		resp, err = m.runtimeService.ListContainers(&runtimeApi.ContainerFilter{
			State: &runningState,
		})
	}

	if err != nil {
		glog.Errorf("ListContainers error: %v", err)
		return nil, err
	}

	result := []*runtimeApi.Container{}
	for _, c := range resp {
		if c.Name == nil || len(c.GetName()) == 0 {
			continue
		}

		containerName := strings.TrimPrefix(c.GetName(), "/")
		if !strings.HasPrefix(containerName, containerNamePrefix+"_") {
			glog.V(3).Infof("%s container %s is not managed by kubelet", m.runtimeName, containerName)
			continue
		}

		if !isContainerManagedByKubelet(containerName) {
			glog.V(3).Infof("%s container %s is not managed by kubelet", m.runtimeName, containerName)
			continue
		}

		result = append(result, c)
	}

	return result, nil
}

// DeleteContainer removes a container.
func (m *kubeGenericRuntimeManager) DeleteContainer(containerID kubecontainer.ContainerID) error {
	return m.runtimeService.RemoveContainer(containerID.ID)
}

// killContainer kills a container through the following steps:
// * Run the pre-stop lifecycle hooks (if applicable).
// * Stop the container.
func (m *kubeGenericRuntimeManager) killContainer(pod *api.Pod, containerID kubecontainer.ContainerID, containerSpec *api.Container, reason string, gracePeriodOverride *int64) error {
	gracePeriod := int64(minimumGracePeriodInSeconds)
	if pod != nil {
		switch {
		case pod.DeletionGracePeriodSeconds != nil:
			gracePeriod = *pod.DeletionGracePeriodSeconds
		case pod.Spec.TerminationGracePeriodSeconds != nil:
			gracePeriod = *pod.Spec.TerminationGracePeriodSeconds
		}
	}

	glog.V(2).Infof("Killing container %q with %d second grace period", containerID.String(), gracePeriod)
	start := unversioned.Now()
	if pod != nil && containerSpec != nil && containerSpec.Lifecycle != nil && containerSpec.Lifecycle.PreStop != nil {
		glog.V(3).Infof("Running preStop hook for container %q", containerID.String())
		done := make(chan struct{})
		go func() {
			defer close(done)
			defer utilruntime.HandleCrash()
			if msg, err := m.runner.Run(containerID, pod, containerSpec, containerSpec.Lifecycle.PreStop); err != nil {
				glog.Errorf("preStop hook for container %q failed: %v", containerSpec.Name, err)
				m.generateFailedContainerEvent(containerID, pod.Name, events.FailedPreStopHook, msg)
			}
		}()
		select {
		case <-time.After(time.Duration(gracePeriod) * time.Second):
			glog.V(2).Infof("preStop hook for container %q did not complete in %d seconds", containerID, gracePeriod)
		case <-done:
			glog.V(3).Infof("preStop hook for container %q completed", containerID)
		}
		gracePeriod -= int64(unversioned.Now().Sub(start.Time).Seconds())
	}

	if gracePeriodOverride == nil {
		// always give containers a minimal shutdown window to avoid unnecessary SIGKILLs
		if gracePeriod < minimumGracePeriodInSeconds {
			gracePeriod = minimumGracePeriodInSeconds
		}
	} else {
		gracePeriod = *gracePeriodOverride
		glog.V(2).Infof("Killing container %q, but using %d second grace period override", containerID, gracePeriod)
	}

	err := m.runtimeService.StopContainer(containerID.ID, gracePeriod)
	if err != nil {
		glog.V(2).Infof("Container %q termination failed after %s: %v", containerID.String(), unversioned.Now().Sub(start.Time), err)
	} else {
		glog.V(2).Infof("Container %q exited after %s", containerID.String(), unversioned.Now().Sub(start.Time))
	}

	ref, ok := m.containerRefManager.GetRef(containerID)
	if !ok {
		glog.Warningf("No ref for container %s", containerID.String())
	} else {
		message := fmt.Sprintf("Killing container with id %s", containerID.String())
		if reason != "" {
			message = fmt.Sprint(message, ":", reason)
		}
		m.recorder.Event(ref, api.EventTypeNormal, events.KillingContainer, message)
		m.containerRefManager.ClearRef(containerID)
	}

	return err
}
