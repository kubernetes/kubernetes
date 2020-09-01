// +build !dockerless

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
	"strings"
	"time"

	dockertypes "github.com/docker/docker/api/types"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// This file contains helper functions to convert docker API types to runtime
// API types, or vice versa.

func imageToRuntimeAPIImage(image *dockertypes.ImageSummary) (*runtimeapi.Image, error) {
	if image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime API image")
	}

	size := uint64(image.VirtualSize)
	return &runtimeapi.Image{
		Id:          image.ID,
		RepoTags:    image.RepoTags,
		RepoDigests: image.RepoDigests,
		Size_:       size,
	}, nil
}

func imageInspectToRuntimeAPIImage(image *dockertypes.ImageInspect) (*runtimeapi.Image, error) {
	if image == nil || image.Config == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime API image")
	}

	size := uint64(image.VirtualSize)
	runtimeImage := &runtimeapi.Image{
		Id:          image.ID,
		RepoTags:    image.RepoTags,
		RepoDigests: image.RepoDigests,
		Size_:       size,
	}

	uid, username := getUserFromImageUser(image.Config.User)
	if uid != nil {
		runtimeImage.Uid = &runtimeapi.Int64Value{Value: *uid}
	}
	runtimeImage.Username = username
	return runtimeImage, nil
}

func toPullableImageID(id string, image *dockertypes.ImageInspect) string {
	// Default to the image ID, but if RepoDigests is not empty, use
	// the first digest instead.
	imageID := DockerImageIDPrefix + id
	if image != nil && len(image.RepoDigests) > 0 {
		imageID = DockerPullableImageIDPrefix + image.RepoDigests[0]
	}
	return imageID
}

func toRuntimeAPIContainer(c *dockertypes.Container) (*runtimeapi.Container, error) {
	state := toRuntimeAPIContainerState(c.Status)
	if len(c.Names) == 0 {
		return nil, fmt.Errorf("unexpected empty container name: %+v", c)
	}
	metadata, err := parseContainerName(c.Names[0])
	if err != nil {
		return nil, err
	}
	labels, annotations := extractLabels(c.Labels)
	sandboxID := c.Labels[sandboxIDLabelKey]
	// The timestamp in dockertypes.Container is in seconds.
	createdAt := c.Created * int64(time.Second)
	return &runtimeapi.Container{
		Id:           c.ID,
		PodSandboxId: sandboxID,
		Metadata:     metadata,
		Image:        &runtimeapi.ImageSpec{Image: c.Image},
		ImageRef:     c.ImageID,
		State:        state,
		CreatedAt:    createdAt,
		Labels:       labels,
		Annotations:  annotations,
	}, nil
}

func toDockerContainerStatus(state runtimeapi.ContainerState) string {
	switch state {
	case runtimeapi.ContainerState_CONTAINER_CREATED:
		return "created"
	case runtimeapi.ContainerState_CONTAINER_RUNNING:
		return "running"
	case runtimeapi.ContainerState_CONTAINER_EXITED:
		return "exited"
	case runtimeapi.ContainerState_CONTAINER_UNKNOWN:
		fallthrough
	default:
		return "unknown"
	}
}

func toRuntimeAPIContainerState(state string) runtimeapi.ContainerState {
	// Parse the state string in dockertypes.Container. This could break when
	// we upgrade docker.
	switch {
	case strings.HasPrefix(state, libdocker.StatusRunningPrefix):
		return runtimeapi.ContainerState_CONTAINER_RUNNING
	case strings.HasPrefix(state, libdocker.StatusExitedPrefix):
		return runtimeapi.ContainerState_CONTAINER_EXITED
	case strings.HasPrefix(state, libdocker.StatusCreatedPrefix):
		return runtimeapi.ContainerState_CONTAINER_CREATED
	default:
		return runtimeapi.ContainerState_CONTAINER_UNKNOWN
	}
}

func toRuntimeAPISandboxState(state string) runtimeapi.PodSandboxState {
	// Parse the state string in dockertypes.Container. This could break when
	// we upgrade docker.
	switch {
	case strings.HasPrefix(state, libdocker.StatusRunningPrefix):
		return runtimeapi.PodSandboxState_SANDBOX_READY
	default:
		return runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	}
}

func containerToRuntimeAPISandbox(c *dockertypes.Container) (*runtimeapi.PodSandbox, error) {
	state := toRuntimeAPISandboxState(c.Status)
	if len(c.Names) == 0 {
		return nil, fmt.Errorf("unexpected empty sandbox name: %+v", c)
	}
	metadata, err := parseSandboxName(c.Names[0])
	if err != nil {
		return nil, err
	}
	labels, annotations := extractLabels(c.Labels)
	// The timestamp in dockertypes.Container is in seconds.
	createdAt := c.Created * int64(time.Second)
	return &runtimeapi.PodSandbox{
		Id:          c.ID,
		Metadata:    metadata,
		State:       state,
		CreatedAt:   createdAt,
		Labels:      labels,
		Annotations: annotations,
	}, nil
}

func checkpointToRuntimeAPISandbox(id string, checkpoint ContainerCheckpoint) *runtimeapi.PodSandbox {
	state := runtimeapi.PodSandboxState_SANDBOX_NOTREADY
	_, name, namespace, _, _ := checkpoint.GetData()
	return &runtimeapi.PodSandbox{
		Id: id,
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      name,
			Namespace: namespace,
		},
		State: state,
	}
}
