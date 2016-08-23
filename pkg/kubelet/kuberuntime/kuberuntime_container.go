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
	"math/rand"
	"os"
	"path"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/term"
)

// getContainerLogsPath gets log path for container.
func getContainerLogsPath(containerName string, podUID types.UID) string {
	return path.Join(podLogsRootDirectory, string(podUID), fmt.Sprintf("%s.log", containerName))
}

// generateContainerConfig generates container config for kubelet runtime api.
func (m *kubeGenericRuntimeManager) generateContainerConfig(container *api.Container, pod *api.Pod, restartCount int, podIP string) (*runtimeApi.ContainerConfig, error) {
	opts, err := m.runtimeHelper.GenerateRunContainerOptions(pod, container, podIP)
	if err != nil {
		return nil, err
	}

	command, args := kubecontainer.ExpandContainerCommandAndArgs(container, opts.Envs)
	containerLogsPath := getContainerLogsPath(container.Name, pod.UID)
	podHasSELinuxLabel := pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxOptions != nil
	restartCountUint32 := uint32(restartCount)
	config := &runtimeApi.ContainerConfig{
		Metadata: &runtimeApi.ContainerMetadata{
			Name:    &container.Name,
			Attempt: &restartCountUint32,
		},
		Image:       &runtimeApi.ImageSpec{Image: &container.Image},
		Command:     command,
		Args:        args,
		WorkingDir:  &container.WorkingDir,
		Labels:      newContainerLabels(container, pod),
		Annotations: newContainerAnnotations(container, pod, restartCount),
		Mounts:      makeMounts(opts, container, podHasSELinuxLabel),
		LogPath:     &containerLogsPath,
		Stdin:       &container.Stdin,
		StdinOnce:   &container.StdinOnce,
		Tty:         &container.TTY,
		Linux:       m.generateLinuxContainerConfig(container),
	}

	// set privileged and readonlyRootfs
	if container.SecurityContext != nil {
		securityContext := container.SecurityContext
		if securityContext.Privileged != nil {
			config.Privileged = securityContext.Privileged
		}
		if securityContext.ReadOnlyRootFilesystem != nil {
			config.ReadonlyRootfs = securityContext.ReadOnlyRootFilesystem
		}
	}

	// set environment variables
	envs := make([]*runtimeApi.KeyValue, len(opts.Envs))
	for idx := range opts.Envs {
		e := opts.Envs[idx]
		envs[idx] = &runtimeApi.KeyValue{
			Key:   &e.Name,
			Value: &e.Value,
		}
	}
	config.Envs = envs

	return config, nil
}

// generateLinuxContainerConfig generates linux container config for kubelet runtime api.
func (m *kubeGenericRuntimeManager) generateLinuxContainerConfig(container *api.Container) *runtimeApi.LinuxContainerConfig {
	linuxConfig := &runtimeApi.LinuxContainerConfig{
		Resources: &runtimeApi.LinuxContainerResources{},
	}

	// set linux container resources
	var cpuShares int64
	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	memoryLimit := container.Resources.Limits.Memory().Value()
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
	linuxConfig.Resources.CpuShares = &cpuShares
	if memoryLimit != 0 {
		linuxConfig.Resources.MemoryLimitInBytes = &memoryLimit
	}
	if m.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned
		// to allow full usage of cpu resource.
		cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())
		linuxConfig.Resources.CpuQuota = &cpuQuota
		linuxConfig.Resources.CpuPeriod = &cpuPeriod
	}

	// set security context options
	if container.SecurityContext != nil {
		securityContext := container.SecurityContext
		if securityContext.Capabilities != nil {
			linuxConfig.Capabilities = &runtimeApi.Capability{
				AddCapabilities:  make([]string, 0, len(securityContext.Capabilities.Add)),
				DropCapabilities: make([]string, 0, len(securityContext.Capabilities.Drop)),
			}
			for index, value := range securityContext.Capabilities.Add {
				linuxConfig.Capabilities.AddCapabilities[index] = string(value)
			}
			for index, value := range securityContext.Capabilities.Drop {
				linuxConfig.Capabilities.DropCapabilities[index] = string(value)
			}
		}

		if securityContext.SELinuxOptions != nil {
			linuxConfig.SelinuxOptions = &runtimeApi.SELinuxOption{
				User:  &securityContext.SELinuxOptions.User,
				Role:  &securityContext.SELinuxOptions.Role,
				Type:  &securityContext.SELinuxOptions.Type,
				Level: &securityContext.SELinuxOptions.Level,
			}
		}
	}

	return linuxConfig
}

// makeMounts generates container volume mounts for kubelet runtime api.
func makeMounts(opts *kubecontainer.RunContainerOptions, container *api.Container, podHasSELinuxLabel bool) []*runtimeApi.Mount {
	volumeMounts := []*runtimeApi.Mount{}

	for idx := range opts.Mounts {
		v := opts.Mounts[idx]
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
		// here we just add a random id to make the path unique for different instances
		// of the same container.
		cid := makeUID()
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

// getKubeletContainers lists containers managed by kubelet.
// The boolean parameter specifies whether returns all containers including
// those already exited and dead containers (used for garbage collection).
func (m *kubeGenericRuntimeManager) getKubeletContainers(allContainers bool) ([]*runtimeApi.Container, error) {
	filter := &runtimeApi.ContainerFilter{
		LabelSelector: map[string]string{kubernetesManagedLabel: "true"},
	}
	if !allContainers {
		runningState := runtimeApi.ContainerState_RUNNING
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
func (m *kubeGenericRuntimeManager) getContainersHelper(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
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
