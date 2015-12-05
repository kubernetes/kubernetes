/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"fmt"
	"strings"

	docker "github.com/fsouza/go-dockerclient"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// This file contains helper functions to convert docker API types to runtime
// (kubecontainer) types.

func mapState(state string) kubecontainer.ContainerState {
	// Parse the state string in docker.APIContainers. This could break when
	// we upgrade docker.
	switch {
	case strings.HasPrefix(state, "Up"):
		return kubecontainer.ContainerStateRunning
	case strings.HasPrefix(state, "Exited"):
		return kubecontainer.ContainerStateExited
	default:
		return kubecontainer.ContainerStateUnknown
	}
}

// Converts docker.APIContainers to kubecontainer.Container.
func toRuntimeContainer(c *docker.APIContainers) (*kubecontainer.Container, error) {
	if c == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	dockerName, hash, err := getDockerContainerNameInfo(c)
	if err != nil {
		return nil, err
	}

	return &kubecontainer.Container{
		ID:      kubetypes.DockerID(c.ID).ContainerID(),
		Name:    dockerName.ContainerName,
		Image:   c.Image,
		Hash:    hash,
		Created: c.Created,
		// (random-liu) docker uses status to indicate whether a container is running or exited.
		// However, in kubernetes we usually use state to indicate whether a container is running or exited,
		// while use status to indicate the comprehensive status of the container. So we have different naming
		// norm here.
		State: mapState(c.Status),
	}, nil
}

// Converts docker.APIImages to kubecontainer.Image.
func toRuntimeImage(image *docker.APIImages) (*kubecontainer.Image, error) {
	if image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime image")
	}

	return &kubecontainer.Image{
		ID:   image.ID,
		Tags: image.RepoTags,
		Size: image.VirtualSize,
	}, nil
}

// convert ContainerStatus to api.ContainerStatus.
func containerStatusToAPIContainerStatus(containerStatus *kubecontainer.ContainerStatus) *api.ContainerStatus {
	containerID := DockerPrefix + containerStatus.ID.ID
	status := api.ContainerStatus{
		Name:         containerStatus.Name,
		RestartCount: containerStatus.RestartCount,
		Image:        containerStatus.Image,
		ImageID:      containerStatus.ImageID,
		ContainerID:  containerID,
	}
	switch containerStatus.State {
	case kubecontainer.ContainerStateRunning:
		status.State.Running = &api.ContainerStateRunning{StartedAt: unversioned.NewTime(containerStatus.StartedAt)}
	case kubecontainer.ContainerStateExited:
		status.State.Terminated = &api.ContainerStateTerminated{
			ExitCode:    containerStatus.ExitCode,
			Reason:      containerStatus.Reason,
			Message:     containerStatus.Message,
			StartedAt:   unversioned.NewTime(containerStatus.StartedAt),
			FinishedAt:  unversioned.NewTime(containerStatus.FinishedAt),
			ContainerID: containerID,
		}
	default:
		status.State.Waiting = &api.ContainerStateWaiting{}
	}
	return &status
}
