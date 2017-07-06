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
	"os"
	"path/filepath"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"
	dockerstrslice "github.com/docker/engine-api/types/strslice"
	"github.com/golang/glog"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// ListContainers lists all containers matching the filter.
func (ds *dockerService) ListContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	opts := dockertypes.ContainerListOptions{All: true}

	opts.Filter = dockerfilters.NewArgs()
	f := newDockerFilter(&opts.Filter)
	// Add filter to get *only* (non-sandbox) containers.
	f.AddLabel(containerTypeLabelKey, containerTypeLabelContainer)

	if filter != nil {
		if filter.Id != "" {
			f.Add("id", filter.Id)
		}
		if filter.State != nil {
			f.Add("status", toDockerContainerStatus(filter.GetState().State))
		}
		if filter.PodSandboxId != "" {
			f.AddLabel(sandboxIDLabelKey, filter.PodSandboxId)
		}

		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				f.AddLabel(k, v)
			}
		}
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}
	// Convert docker to runtime api containers.
	result := []*runtimeapi.Container{}
	for i := range containers {
		c := containers[i]

		converted, err := toRuntimeAPIContainer(&c)
		if err != nil {
			glog.V(4).Infof("Unable to convert docker to runtime API container: %v", err)
			continue
		}

		result = append(result, converted)
	}
	// Include legacy containers if there are still legacy containers not cleaned up yet.
	if !ds.legacyCleanup.Done() {
		legacyContainers, err := ds.ListLegacyContainers(filter)
		if err != nil {
			return nil, err
		}
		// Legacy containers are always older, so we can safely append them to the end.
		result = append(result, legacyContainers...)
	}
	return result, nil
}

// CreateContainer creates a new container in the given PodSandbox
// Docker cannot store the log to an arbitrary location (yet), so we create an
// symlink at LogPath, linking to the actual path of the log.
// TODO: check if the default values returned by the runtime API are ok.
func (ds *dockerService) CreateContainer(podSandboxID string, config *runtimeapi.ContainerConfig, sandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	if config == nil {
		return "", fmt.Errorf("container config is nil")
	}
	if sandboxConfig == nil {
		return "", fmt.Errorf("sandbox config is nil for container %q", config.Metadata.Name)
	}

	labels := makeLabels(config.GetLabels(), config.GetAnnotations())
	// Apply a the container type label.
	labels[containerTypeLabelKey] = containerTypeLabelContainer
	// Write the container log path in the labels.
	labels[containerLogPathLabelKey] = filepath.Join(sandboxConfig.LogDirectory, config.LogPath)
	// Write the sandbox ID in the labels.
	labels[sandboxIDLabelKey] = podSandboxID

	apiVersion, err := ds.getDockerAPIVersion()
	if err != nil {
		return "", fmt.Errorf("unable to get the docker API version: %v", err)
	}
	securityOptSep := getSecurityOptSeparator(apiVersion)

	image := ""
	if iSpec := config.GetImage(); iSpec != nil {
		image = iSpec.Image
	}
	createConfig := dockertypes.ContainerCreateConfig{
		Name: makeContainerName(sandboxConfig, config),
		Config: &dockercontainer.Config{
			// TODO: set User.
			Entrypoint: dockerstrslice.StrSlice(config.Command),
			Cmd:        dockerstrslice.StrSlice(config.Args),
			Env:        generateEnvList(config.GetEnvs()),
			Image:      image,
			WorkingDir: config.WorkingDir,
			Labels:     labels,
			// Interactive containers:
			OpenStdin: config.Stdin,
			StdinOnce: config.StdinOnce,
			Tty:       config.Tty,
		},
		HostConfig: &dockercontainer.HostConfig{
			Binds: generateMountBindings(config.GetMounts()),
		},
	}

	hc := createConfig.HostConfig
	ds.updateCreateConfig(&createConfig, config, sandboxConfig, podSandboxID, securityOptSep, apiVersion)
	// Set devices for container.
	devices := make([]dockercontainer.DeviceMapping, len(config.Devices))
	for i, device := range config.Devices {
		devices[i] = dockercontainer.DeviceMapping{
			PathOnHost:        device.HostPath,
			PathInContainer:   device.ContainerPath,
			CgroupPermissions: device.Permissions,
		}
	}
	hc.Resources.Devices = devices

	securityOpts, err := ds.getSecurityOpts(config.Metadata.Name, sandboxConfig, securityOptSep)
	if err != nil {
		return "", fmt.Errorf("failed to generate security options for container %q: %v", config.Metadata.Name, err)
	}

	hc.SecurityOpt = append(hc.SecurityOpt, securityOpts...)

	createResp, err := ds.client.CreateContainer(createConfig)
	if err != nil {
		createResp, err = recoverFromCreationConflictIfNeeded(ds.client, createConfig, err)
	}

	if createResp != nil {
		return createResp.ID, err
	}
	return "", err
}

// getContainerLogPath returns the container log path specified by kubelet and the real
// path where docker stores the container log.
func (ds *dockerService) getContainerLogPath(containerID string) (string, string, error) {
	info, err := ds.client.InspectContainer(containerID)
	if err != nil {
		return "", "", fmt.Errorf("failed to inspect container %q: %v", containerID, err)
	}
	return info.Config.Labels[containerLogPathLabelKey], info.LogPath, nil
}

// createContainerLogSymlink creates the symlink for docker container log.
func (ds *dockerService) createContainerLogSymlink(containerID string) error {
	path, realPath, err := ds.getContainerLogPath(containerID)
	if err != nil {
		return fmt.Errorf("failed to get container %q log path: %v", containerID, err)
	}

	if path == "" {
		glog.V(5).Infof("Container %s log path isn't specified, will not create the symlink", containerID)
		return nil
	}

	if realPath != "" {
		// Only create the symlink when container log path is specified and log file exists.
		if err = ds.os.Symlink(realPath, path); err != nil {
			return fmt.Errorf("failed to create symbolic link %q to the container log file %q for container %q: %v",
				path, realPath, containerID, err)
		}
	} else {
		supported, err := IsCRISupportedLogDriver(ds.client)
		if err != nil {
			glog.Warningf("Failed to check supported logging driver by CRI: %v", err)
			return nil
		}

		if supported {
			glog.Warningf("Cannot create symbolic link because container log file doesn't exist!")
		} else {
			glog.V(5).Infof("Unsupported logging driver by CRI")
		}
	}

	return nil
}

// removeContainerLogSymlink removes the symlink for docker container log.
func (ds *dockerService) removeContainerLogSymlink(containerID string) error {
	path, _, err := ds.getContainerLogPath(containerID)
	if err != nil {
		return fmt.Errorf("failed to get container %q log path: %v", containerID, err)
	}
	if path != "" {
		// Only remove the symlink when container log path is specified.
		err := ds.os.Remove(path)
		if err != nil && !os.IsNotExist(err) {
			return fmt.Errorf("failed to remove container %q log symlink %q: %v", containerID, path, err)
		}
	}
	return nil
}

// StartContainer starts the container.
func (ds *dockerService) StartContainer(containerID string) error {
	err := ds.client.StartContainer(containerID)
	if err != nil {
		err = transformStartContainerError(err)
		return fmt.Errorf("failed to start container %q: %v", containerID, err)
	}
	// Create container log symlink.
	if err := ds.createContainerLogSymlink(containerID); err != nil {
		// Do not stop the container if we failed to create symlink because:
		//   1. This is not a critical failure.
		//   2. We don't have enough information to properly stop container here.
		// Kubelet will surface this error to user via an event.
		return err
	}
	return nil
}

// StopContainer stops a running container with a grace period (i.e., timeout).
func (ds *dockerService) StopContainer(containerID string, timeout int64) error {
	return ds.client.StopContainer(containerID, int(timeout))
}

// RemoveContainer removes the container.
func (ds *dockerService) RemoveContainer(containerID string) error {
	// Ideally, log lifecycle should be independent of container lifecycle.
	// However, docker will remove container log after container is removed,
	// we can't prevent that now, so we also clean up the symlink here.
	err := ds.removeContainerLogSymlink(containerID)
	if err != nil {
		return err
	}
	err = ds.client.RemoveContainer(containerID, dockertypes.ContainerRemoveOptions{RemoveVolumes: true, Force: true})
	if err != nil {
		return fmt.Errorf("failed to remove container %q: %v", containerID, err)
	}
	return nil
}

func getContainerTimestamps(r *dockertypes.ContainerJSON) (time.Time, time.Time, time.Time, error) {
	var createdAt, startedAt, finishedAt time.Time
	var err error

	createdAt, err = libdocker.ParseDockerTimestamp(r.Created)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	startedAt, err = libdocker.ParseDockerTimestamp(r.State.StartedAt)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	finishedAt, err = libdocker.ParseDockerTimestamp(r.State.FinishedAt)
	if err != nil {
		return createdAt, startedAt, finishedAt, err
	}
	return createdAt, startedAt, finishedAt, nil
}

// ContainerStatus inspects the docker container and returns the status.
func (ds *dockerService) ContainerStatus(containerID string) (*runtimeapi.ContainerStatus, error) {
	r, err := ds.client.InspectContainer(containerID)
	if err != nil {
		return nil, err
	}

	// Parse the timestamps.
	createdAt, startedAt, finishedAt, err := getContainerTimestamps(r)
	if err != nil {
		return nil, fmt.Errorf("failed to parse timestamp for container %q: %v", containerID, err)
	}

	// Convert the image id to a pullable id.
	ir, err := ds.client.InspectImageByID(r.Image)
	if err != nil {
		return nil, fmt.Errorf("unable to inspect docker image %q while inspecting docker container %q: %v", r.Image, containerID, err)
	}
	imageID := toPullableImageID(r.Image, ir)

	// Convert the mounts.
	mounts := make([]*runtimeapi.Mount, 0, len(r.Mounts))
	for i := range r.Mounts {
		m := r.Mounts[i]
		readonly := !m.RW
		mounts = append(mounts, &runtimeapi.Mount{
			HostPath:      m.Source,
			ContainerPath: m.Destination,
			Readonly:      readonly,
			// Note: Can't set SeLinuxRelabel
		})
	}
	// Interpret container states.
	var state runtimeapi.ContainerState
	var reason, message string
	if r.State.Running {
		// Container is running.
		state = runtimeapi.ContainerState_CONTAINER_RUNNING
	} else {
		// Container is *not* running. We need to get more details.
		//    * Case 1: container has run and exited with non-zero finishedAt
		//              time.
		//    * Case 2: container has failed to start; it has a zero finishedAt
		//              time, but a non-zero exit code.
		//    * Case 3: container has been created, but not started (yet).
		if !finishedAt.IsZero() { // Case 1
			state = runtimeapi.ContainerState_CONTAINER_EXITED
			switch {
			case r.State.OOMKilled:
				// TODO: consider exposing OOMKilled via the runtimeAPI.
				// Note: if an application handles OOMKilled gracefully, the
				// exit code could be zero.
				reason = "OOMKilled"
			case r.State.ExitCode == 0:
				reason = "Completed"
			default:
				reason = "Error"
			}
		} else if r.State.ExitCode != 0 { // Case 2
			state = runtimeapi.ContainerState_CONTAINER_EXITED
			// Adjust finshedAt and startedAt time to createdAt time to avoid
			// the confusion.
			finishedAt, startedAt = createdAt, createdAt
			reason = "ContainerCannotRun"
		} else { // Case 3
			state = runtimeapi.ContainerState_CONTAINER_CREATED
		}
		message = r.State.Error
	}

	// Convert to unix timestamps.
	ct, st, ft := createdAt.UnixNano(), startedAt.UnixNano(), finishedAt.UnixNano()
	exitCode := int32(r.State.ExitCode)

	// If the container has no containerTypeLabelKey label, treat it as a legacy container.
	if _, ok := r.Config.Labels[containerTypeLabelKey]; !ok {
		names, labels, err := convertLegacyNameAndLabels([]string{r.Name}, r.Config.Labels)
		if err != nil {
			return nil, err
		}
		r.Name, r.Config.Labels = names[0], labels
	}

	metadata, err := parseContainerName(r.Name)
	if err != nil {
		return nil, err
	}

	labels, annotations := extractLabels(r.Config.Labels)
	imageName := r.Config.Image
	if len(ir.RepoTags) > 0 {
		imageName = ir.RepoTags[0]
	}
	return &runtimeapi.ContainerStatus{
		Id:          r.ID,
		Metadata:    metadata,
		Image:       &runtimeapi.ImageSpec{Image: imageName},
		ImageRef:    imageID,
		Mounts:      mounts,
		ExitCode:    exitCode,
		State:       state,
		CreatedAt:   ct,
		StartedAt:   st,
		FinishedAt:  ft,
		Reason:      reason,
		Message:     message,
		Labels:      labels,
		Annotations: annotations,
		LogPath:     r.Config.Labels[containerLogPathLabelKey],
	}, nil
}
