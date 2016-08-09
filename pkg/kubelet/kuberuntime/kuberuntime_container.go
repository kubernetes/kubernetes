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
	"os"
	"path"
	"sort"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/term"
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

	_, containerName, cid := buildContainerName(pod, container)
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
		glog.Errorf("ListContainers failed: %v", err)
		return nil, err
	}

	result := []*runtimeApi.Container{}
	for _, c := range resp {
		if !isContainerManagedByKubelet(c.GetName()) {
			glog.V(5).Infof("Container %s is not managed by kubelet", c.GetName())
			continue
		}

		result = append(result, c)
	}

	return result, nil
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
	// TODO: optimization: set maximum number of containers per container name to examine.
	for i, c := range containers {
		if !isContainerManagedByKubelet(c.GetName()) {
			glog.V(5).Infof("Container %s is not managed by kubelet", c.GetName())
			continue
		}

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
			CreatedAt:    time.Unix(status.GetCreatedAt(), 0),
		}

		if c.GetState() == runtimeApi.ContainerState_RUNNING {
			cStatus.StartedAt = time.Unix(status.GetStartedAt(), 0)
		} else {
			cStatus.Reason = status.GetReason()
			cStatus.ExitCode = int(status.GetExitCode())
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

	sort.Sort(containerStatusByCreated(statuses))
	return statuses, nil
}

// AttachContainer attaches to the container's console
func (m *kubeGenericRuntimeManager) AttachContainer(id kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) (err error) {
	return fmt.Errorf("not implemented")
}

// GetContainerLogs returns logs of a specific container.
func (m *kubeGenericRuntimeManager) GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error) {
	return fmt.Errorf("not implemented")
}

// Runs the command in the container of the specified pod using nsenter.
// Attaches the processes stdin, stdout, and stderr. Optionally uses a
// tty.
// TODO: handle terminal resizing, refer https://github.com/kubernetes/kubernetes/issues/29579
func (m *kubeGenericRuntimeManager) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	return fmt.Errorf("not implemented")
}

// DeleteContainer removes a container.
func (m *kubeGenericRuntimeManager) DeleteContainer(containerID kubecontainer.ContainerID) error {
	return m.runtimeService.RemoveContainer(containerID.ID)
}
