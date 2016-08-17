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
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"
	dockerstrslice "github.com/docker/engine-api/types/strslice"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

// ListContainers lists all containers matching the filter.
func (ds *dockerService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	opts := dockertypes.ContainerListOptions{All: true}

	opts.Filter = dockerfilters.NewArgs()
	f := newDockerFilter(&opts.Filter)

	if filter != nil {
		if filter.Id != nil {
			f.Add("id", filter.GetId())
		}
		if filter.State != nil {
			f.Add("status", toDockerContainerStatus(filter.GetState()))
		}
		if filter.PodSandboxId != nil {
			// TODO: implement this after sandbox functions are implemented.
		}

		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				f.AddLabel(k, v)
			}
		}
		// Filter out sandbox containers.
		f.AddLabel(containerTypeLabelKey, containerTypeLabelContainer)
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}
	// Convert docker to runtime api containers.
	result := []*runtimeApi.Container{}
	for _, c := range containers {
		if len(filter.GetName()) > 0 {
			_, _, _, containerName, _, err := parseContainerName(c.Names[0])
			if err != nil || containerName != filter.GetName() {
				continue
			}
		}

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
		return "", fmt.Errorf("sandbox config is nil for container %q", config.Metadata.GetName())
	}

	// Merge annotations and labels because docker supports only labels.
	// TODO: add a prefix to annotations so that we can distinguish labels and
	// annotations when reading back them from the docker container.
	labels := makeLabels(config.GetLabels(), config.GetAnnotations())
	// Apply a the container type label.
	labels[containerTypeLabelKey] = containerTypeLabelContainer

	image := ""
	if iSpec := config.GetImage(); iSpec != nil {
		image = iSpec.GetImage()
	}
	createConfig := dockertypes.ContainerCreateConfig{
		Name: buildContainerName(sandboxConfig, config),
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

	hc.SecurityOpt = []string{getSeccompOpts()}
	// TODO: Add or drop capabilities.

	createConfig.HostConfig = hc
	createResp, err := ds.client.CreateContainer(createConfig)
	if createResp != nil {
		return createResp.ID, err
	}
	return "", err
}

// StartContainer starts the container.
func (ds *dockerService) StartContainer(containerID string) error {
	return ds.client.StartContainer(containerID)
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (ds *dockerService) StopContainer(containerID string, timeout int64) error {
	return ds.client.StopContainer(containerID, int(timeout))
}

// RemoveContainer removes the container.
// TODO: If a container is still running, should we forcibly remove it?
func (ds *dockerService) RemoveContainer(containerID string) error {
	return ds.client.RemoveContainer(containerID, dockertypes.ContainerRemoveOptions{RemoveVolumes: true})
}

func getContainerTimestamps(r *dockertypes.ContainerJSON) (time.Time, time.Time, time.Time, error) {
	var createdAt, startedAt, finishedAt time.Time
	var err error

	createdAt, err = dockertools.ParseDockerTimestamp(r.Created)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	startedAt, err = dockertools.ParseDockerTimestamp(r.State.StartedAt)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	finishedAt, err = dockertools.ParseDockerTimestamp(r.State.FinishedAt)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	return createdAt, startedAt, finishedAt, nil
}

// ContainerStatus returns the container status.
func (ds *dockerService) ContainerStatus(containerID string) (*runtimeApi.ContainerStatus, error) {
	r, err := ds.client.InspectContainer(containerID)
	if err != nil {
		return nil, err
	}

	// Parse the timstamps.
	createdAt, startedAt, finishedAt, err := getContainerTimestamps(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse timestamp for container %q: %v", containerID, err)
	}

	// Convert the mounts.
	mounts := []*runtimeApi.Mount{}
	for _, m := range r.Mounts {
		readonly := !m.RW
		mounts = append(mounts, &runtimeApi.Mount{
			Name:          &m.Name,
			HostPath:      &m.Source,
			ContainerPath: &m.Destination,
			Readonly:      &readonly,
			// Note: Can't set SeLinuxRelabel
		})
	}
	// Interpret container states.
	var state runtimeApi.ContainerState
	var reason string
	if r.State.Running {
		// Container is running.
		state = runtimeApi.ContainerState_RUNNING
	} else {
		// Container is *not* running. We need to get more details.
		//    * Case 1: container has run and exited with non-zero finishedAt
		//              time.
		//    * Case 2: container has failed to start; it has a zero finishedAt
		//              time, but a non-zero exit code.
		//    * Case 3: container has been created, but not started (yet).
		if !finishedAt.IsZero() { // Case 1
			state = runtimeApi.ContainerState_EXITED
			switch {
			case r.State.OOMKilled:
				// TODO: consider exposing OOMKilled via the runtimeAPI.
				// Note: if an application handles OOMKilled gracefully, the
				// exit code could be zero.
				reason = "OOMKilled"
			case r.State.ExitCode == 0:
				reason = "Completed"
			default:
				reason = fmt.Sprintf("Error: %s", r.State.Error)
			}
		} else if !finishedAt.IsZero() && r.State.ExitCode != 0 { // Case 2
			state = runtimeApi.ContainerState_EXITED
			// Adjust finshedAt and startedAt time to createdAt time to avoid
			// the confusion.
			finishedAt, startedAt = createdAt, createdAt
			reason = "ContainerCannotRun"
		} else { // Case 3
			state = runtimeApi.ContainerState_CREATED
		}
	}

	// Convert to unix timestamps.
	ct, st, ft := createdAt.Unix(), startedAt.Unix(), finishedAt.Unix()
	exitCode := int32(r.State.ExitCode)

	_, _, _, containerName, attempt, err := parseContainerName(r.Name)
	if err != nil {
		return nil, err
	}

	return &runtimeApi.ContainerStatus{
		Id: &r.ID,
		Metadata: &runtimeApi.ContainerMetadata{
			Name:    &containerName,
			Attempt: &attempt,
		},
		Image:      &runtimeApi.ImageSpec{Image: &r.Config.Image},
		ImageRef:   &r.Image,
		Mounts:     mounts,
		ExitCode:   &exitCode,
		State:      &state,
		CreatedAt:  &ct,
		StartedAt:  &st,
		FinishedAt: &ft,
		Reason:     &reason,
		// TODO: We write annotations as labels on the docker containers. All
		// these annotations will be read back as labels. Need to fix this.
		Labels: r.Config.Labels,
	}, nil
}

// Exec execute a command in the container.
// TODO: Need to handle terminal resizing before implementing this function.
// https://github.com/kubernetes/kubernetes/issues/29579.
func (ds *dockerService) Exec(containerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	return fmt.Errorf("not implemented")
}
