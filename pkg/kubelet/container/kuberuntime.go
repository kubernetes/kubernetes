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

package container

import (
	"io"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// RuntimeService interface defines the interfaces that should be implemented
// by a container runtime.
// Thread safety is required from implementations of this interface.
type RuntimeService interface {
	// Version returns the runtime name, runtime version and runtime API version
	Version(apiVersion string) (*runtimeApi.VersionResponse, error)
	// CreatePodSandbox creates a pod-level sandbox.
	// The definition of PodSandbox is at https://github.com/kubernetes/kubernetes/pull/25899
	CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error)
	// StopPodSandbox stops the sandbox. If there are any running containers in the
	// sandbox, they should be force terminated.
	StopPodSandbox(podSandBoxID string) error
	// DeletePodSandbox deletes the sandbox. If there are any running containers in the
	// sandbox, they should be force deleted.
	DeletePodSandbox(podSandBoxID string) error
	// PodSandboxStatus returns the Status of the PodSandbox.
	PodSandboxStatus(podSandBoxID string) (*runtimeApi.PodSandboxStatus, error)
	// ListPodSandbox returns a list of SandBox.
	ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error)
	// CreateContainer creates a new container in specified PodSandbox
	CreateContainer(podSandBoxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error)
	// StartContainer starts the container.
	StartContainer(rawContainerID string) error
	// StopContainer stops a running container with a grace period (i.e., timeout).
	StopContainer(rawContainerID string, timeout int64) error
	// RemoveContainer removes the container. If the container is running, the container
	// should be force removed.
	RemoveContainer(rawContainerID string) error
	// ListContainers lists all containers by filters.
	ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error)
	// ContainerStatus returns the container status.
	ContainerStatus(rawContainerID string) (*runtimeApi.ContainerStatus, error)
	// Exec execute a command in the container.
	Exec(rawContainerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error
}

// ImageManagerService interface defines the interfaces that should be implemented
// by a container image manager.
// Thread safety is required from implementations of this interface.
type ImageManagerService interface {
	// ListImages lists existing images.
	ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error)
	// ImageStatus returns the status of the image.
	ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error)
	// PullImage pulls a image with authentication config.
	PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error
	// RemoveImage removes the image.
	RemoveImage(image *runtimeApi.ImageSpec) error
}
