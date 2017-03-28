/*
Copyright 2015 The Kubernetes Authors.

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

	dockertypes "github.com/docker/engine-api/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// This file contains helper functions to convert docker API types to runtime
// (kubecontainer) types.
const (
	statusRunningPrefix = "Up"
	statusCreatedPrefix = "Created"
	statusExitedPrefix  = "Exited"
)

func mapState(state string) kubecontainer.ContainerState {
	// Parse the state string in dockertypes.Container. This could break when
	// we upgrade docker.
	switch {
	case strings.HasPrefix(state, statusRunningPrefix):
		return kubecontainer.ContainerStateRunning
	case strings.HasPrefix(state, statusExitedPrefix):
		return kubecontainer.ContainerStateExited
	default:
		return kubecontainer.ContainerStateUnknown
	}
}

// Converts dockertypes.Container to kubecontainer.Container.
func toRuntimeContainer(c *dockertypes.Container) (*kubecontainer.Container, error) {
	if c == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime container")
	}

	dockerName, hash, err := getDockerContainerNameInfo(c)
	if err != nil {
		return nil, err
	}

	return &kubecontainer.Container{
		ID:      kubecontainer.DockerID(c.ID).ContainerID(),
		Name:    dockerName.ContainerName,
		Image:   c.Image,
		ImageID: c.ImageID,
		Hash:    hash,
		// (random-liu) docker uses status to indicate whether a container is running or exited.
		// However, in kubernetes we usually use state to indicate whether a container is running or exited,
		// while use status to indicate the comprehensive status of the container. So we have different naming
		// norm here.
		State: mapState(c.Status),
	}, nil
}

// Converts dockertypes.Image to kubecontainer.Image.
func toRuntimeImage(image *dockertypes.Image) (*kubecontainer.Image, error) {
	if image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime image")
	}

	return &kubecontainer.Image{
		ID:          image.ID,
		RepoTags:    image.RepoTags,
		RepoDigests: image.RepoDigests,
		Size:        image.VirtualSize,
	}, nil
}
