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

package dockershim

import (
	"fmt"
	"io"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"
	dockerstrslice "github.com/docker/engine-api/types/strslice"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// ListContainers lists all containers matching the filter.
func (ds *dockerService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	opts := dockertypes.ContainerListOptions{All: true}

	opts.Filter = dockerfilters.NewArgs()
	if filter != nil {
		if filter.Name != nil {
			opts.Filter.Add("name", filter.GetName())
		}
		if filter.Id != nil {
			opts.Filter.Add("id", filter.GetId())
		}
		if filter.State != nil {
			opts.Filter.Add("status", toDockerContainerStatus(filter.GetState()))
		}
		if filter.PodSandboxId != nil {
			// TODO: implement this after sandbox functions are implemented.
		}

		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				opts.Filter.Add("label", fmt.Sprintf("%s=%s", k, v))
			}
		}
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}
	// Convert docker to runtime api containers.
	result := []*runtimeApi.Container{}
	for _, c := range containers {
		result = append(result, toRuntimeAPIContainer(&c))
	}
	return result, nil
}

// CreateContainer creates a new container in the given PodSandbox
// Note: docker doesn't use LogPath yet.
// TODO: check if the default values returned by the runtime API are ok.
func (ds *dockerService) CreateContainer(podSandboxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	if config == nil {
		return "", fmt.Errorf("container config is nil")
	}
	if sandboxConfig == nil {
		return "", fmt.Errorf("sandbox config is nil for container %q", config.GetName())
	}

	// Merge annotations and labels because docker supports only labels.
	// TODO: add a prefix to annotations so that we can distinguish labels and
	// annotations when reading back them from the docker container.
	// TODO: should we apply docker-specific labels?
	labels := config.GetLabels()
	for k, v := range config.GetAnnotations() {
		if _, ok := labels[k]; !ok {
			// Only write to labels if the key doesn't exist.
			labels[k] = v
		}
	}

	image := ""
	if iSpec := config.GetImage(); iSpec != nil {
		image = iSpec.GetImage()
	}
	createConfig := dockertypes.ContainerCreateConfig{
		Name: config.GetName(),
		Config: &dockercontainer.Config{
			// TODO: set User.
			Hostname:   sandboxConfig.GetHostname(),
			Entrypoint: dockerstrslice.StrSlice(config.GetCommand()),
			Cmd:        dockerstrslice.StrSlice(config.GetArgs()),
			Env:        generateEnvList(config.GetEnvs()),
			Image:      image,
			WorkingDir: config.GetWorkingDir(),
			Labels:     labels,
			// Interactive containers:
			OpenStdin: config.GetStdin(),
			StdinOnce: config.GetStdinOnce(),
			Tty:       config.GetTty(),
		},
	}

	// Fill the HostConfig.
	hc := &dockercontainer.HostConfig{
		Binds:          generateMountBindings(config.GetMounts()),
		ReadonlyRootfs: config.GetReadonlyRootfs(),
		Privileged:     config.GetPrivileged(),
	}

	// Apply options derived from the sandbox config.
	if lc := sandboxConfig.GetLinux(); lc != nil {
		// Apply Cgroup options.
		// TODO: Check if this works with per-pod cgroups.
		hc.CgroupParent = lc.GetCgroupParent()

		// Apply namespace options.
		sandboxNSMode := fmt.Sprintf("container:%v", podSandboxID)
		hc.NetworkMode = dockercontainer.NetworkMode(sandboxNSMode)
		hc.IpcMode = dockercontainer.IpcMode(sandboxNSMode)
		hc.UTSMode = ""
		hc.PidMode = ""

		nsOpts := lc.GetNamespaceOptions()
		if nsOpts != nil {
			if nsOpts.GetHostNetwork() {
				hc.UTSMode = namespaceModeHost
			}
			if nsOpts.GetHostPid() {
				hc.PidMode = namespaceModeHost
			}
		}
	}

	// Apply Linux-specific options if applicable.
	if lc := config.GetLinux(); lc != nil {
		// Apply resource options.
		// TODO: Check if the units are correct.
		// TODO: Can we assume the defaults are sane?
		rOpts := lc.GetResources()
		if rOpts != nil {
			hc.Resources = dockercontainer.Resources{
				Memory:     rOpts.GetMemoryLimitInBytes(),
				MemorySwap: -1, // Always disable memory swap.
				CPUShares:  rOpts.GetCpuShares(),
				CPUQuota:   rOpts.GetCpuQuota(),
				CPUPeriod:  rOpts.GetCpuPeriod(),
				// TODO: Need to set devices.
			}
			hc.OomScoreAdj = int(rOpts.GetOomScoreAdj())
		}
		// Note: ShmSize is handled in kube_docker_client.go
	}

	// TODO: Seccomp support. Need to figure out how to pass seccomp options
	// through the runtime API (annotations?).See dockerManager.getSecurityOpts()
	// for the details. Always set the default seccomp profile for now.
	hc.SecurityOpt = []string{fmt.Sprintf("%s=%s", "seccomp", defaultSeccompProfile)}
	// TODO: Add or drop capabilities.

	createConfig.HostConfig = hc
	createResp, err := ds.client.CreateContainer(createConfig)
	if createResp != nil {
		return createResp.ID, err
	}
	return "", err
}

// StartContainer starts the container.
func (ds *dockerService) StartContainer(rawContainerID string) error {
	return ds.client.StartContainer(rawContainerID)
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (ds *dockerService) StopContainer(rawContainerID string, timeout int64) error {
	return ds.client.StopContainer(rawContainerID, int(timeout))
}

// RemoveContainer removes the container.
// TODO: If a container is still running, should we forcibly remove it?
func (ds *dockerService) RemoveContainer(rawContainerID string) error {
	return ds.client.RemoveContainer(rawContainerID, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
}

// ContainerStatus returns the container status.
// TODO: Implement the function.
func (ds *dockerService) ContainerStatus(rawContainerID string) (*runtimeApi.ContainerStatus, error) {
	return nil, fmt.Errorf("not implemented")
}

// Exec execute a command in the container.
// TODO: Implement the function.
func (ds *dockerService) Exec(rawContainerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	return fmt.Errorf("not implemented")
}
