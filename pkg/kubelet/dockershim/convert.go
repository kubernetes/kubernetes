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

	dockertypes "github.com/docker/engine-api/types"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// This file contains helper functions to convert docker API types to runtime
// API types, or vice versa.

func imageToRuntimeAPIImage(image *dockertypes.Image) (*runtimeApi.Image, error) {
	if image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime API image")
	}

	size := uint64(image.VirtualSize)
	return &runtimeApi.Image{
		Id:          &image.ID,
		RepoTags:    image.RepoTags,
		RepoDigests: image.RepoDigests,
		Size_:       &size,
	}, nil
}

func imageInspectToRuntimeAPIImage(image *dockertypes.ImageInspect) (*runtimeApi.Image, error) {
	if image == nil {
		return nil, fmt.Errorf("unable to convert a nil pointer to a runtime API image")
	}

	size := uint64(image.VirtualSize)
	return &runtimeApi.Image{
		Id:          &image.ID,
		RepoTags:    image.RepoTags,
		RepoDigests: image.RepoDigests,
		Size_:       &size,
	}, nil

}

func toPullableImageID(id string, image *dockertypes.ImageInspect) string {
	// Default to the image ID, but if RepoDigests is not empty, use
	// the first digest instead.
	imageID := DockerImageIDPrefix + id
	if len(image.RepoDigests) > 0 {
		imageID = DockerPullableImageIDPrefix + image.RepoDigests[0]
	}
	return imageID
}

func toRuntimeAPIContainer(c *dockertypes.Container) (*runtimeApi.Container, error) {
	state := toRuntimeAPIContainerState(c)
	metadata, err := parseContainerName(c.Names[0])
	if err != nil {
		return nil, err
	}
	labels, annotations := extractLabels(c.Labels)
	sandboxID := c.Labels[sandboxIDLabelKey]
	// The timestamp in dockertypes.Container is in seconds.
	createdAt := c.Created * int64(time.Second)
	return &runtimeApi.Container{
		Id:           &c.ID,
		PodSandboxId: &sandboxID,
		Metadata:     metadata,
		Image:        &runtimeApi.ImageSpec{Image: &c.Image},
		ImageRef:     &c.ImageID,
		State:        &state,
		CreatedAt:    &createdAt,
		Labels:       labels,
		Annotations:  annotations,
	}, nil
}

func toRuntimeAPIContainerState(c *dockertypes.Container) runtimeApi.ContainerState {
	if len(c.State) == 0 {
		// Docker <= 1.10 (remote API <= 1.22) does not include "State" in the
		// response when listing the containers. We have to rely on translating
		// the "Status" string into a state.
		// TODO: Remove this once we determine not to support Docker 1.10.
		switch {
		case strings.HasPrefix(c.Status, "Up"):
			return runtimeApi.ContainerState_RUNNING
		case strings.HasPrefix(c.Status, "Exited"):
			return runtimeApi.ContainerState_EXITED
		case strings.HasPrefix(c.Status, "Created"):
			return runtimeApi.ContainerState_CREATED
		default:
			return runtimeApi.ContainerState_UNKNOWN
		}
	}

	switch c.State {
	case "running":
		return runtimeApi.ContainerState_RUNNING
	case "exited":
		return runtimeApi.ContainerState_EXITED
	case "created":
		return runtimeApi.ContainerState_CREATED
	default:
		return runtimeApi.ContainerState_UNKNOWN
	}
}

func toRuntimeAPISandboxState(c *dockertypes.Container) runtimeApi.PodSandBoxState {
	if len(c.State) == 0 {
		// Docker <= 1.10 (remote API <= 1.22) does not include "State" in the
		// response when listing the containers. We have to rely on translating
		// the "Status" string into a state.
		// TODO: Remove this once we determine not to support Docker 1.10.
		switch {
		case strings.HasPrefix(c.Status, "Up"):
			return runtimeApi.PodSandBoxState_READY
		default:
			return runtimeApi.PodSandBoxState_NOTREADY
		}
	}
	switch c.State {
	case "running":
		return runtimeApi.PodSandBoxState_READY
	default:
		return runtimeApi.PodSandBoxState_NOTREADY
	}
}

func toRuntimeAPISandbox(c *dockertypes.Container) (*runtimeApi.PodSandbox, error) {
	state := toRuntimeAPISandboxState(c)
	metadata, err := parseSandboxName(c.Names[0])
	if err != nil {
		return nil, err
	}
	labels, annotations := extractLabels(c.Labels)
	// The timestamp in dockertypes.Container is in seconds.
	createdAt := c.Created * int64(time.Second)
	return &runtimeApi.PodSandbox{
		Id:          &c.ID,
		Metadata:    metadata,
		State:       &state,
		CreatedAt:   &createdAt,
		Labels:      labels,
		Annotations: annotations,
	}, nil
}
