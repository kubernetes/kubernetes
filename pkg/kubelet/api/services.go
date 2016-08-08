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

package api

import (
	"io"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// RuntimeService interface should be implemented by a container runtime.
// The methods should be thread-safe.
type RuntimeService interface {
	// Version returns the runtime name, runtime version and runtime API version
	Version(apiVersion string) (*runtimeApi.VersionResponse, error)
	// CreatePodSandbox creates a pod-level sandbox.
	// The definition of PodSandbox is at https://github.com/kubernetes/kubernetes/pull/25899
	CreatePodSandbox(config *runtimeApi.PodSandboxConfig) (string, error)
	// StopPodSandbox stops the sandbox. If there are any running containers in the
	// sandbox, they should be force terminated.
	StopPodSandbox(podSandboxID string) error
	// RemovePodSandbox removes the sandbox. If there are running containers in the
	// sandbox, they should be forcibly removed.
	RemovePodSandbox(podSandboxID string) error
	// PodSandboxStatus returns the Status of the PodSandbox.
	PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error)
	// ListPodSandbox returns a list of Sandbox.
	ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error)
	// CreateContainer creates a new container in specified PodSandbox.
	CreateContainer(podSandboxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error)
	// StartContainer starts the container.
	StartContainer(containerID string) error
	// StopContainer stops a running container with a grace period (i.e., timeout).
	StopContainer(containerID string, timeout int64) error
	// RemoveContainer removes the container.
	RemoveContainer(containerID string) error
	// ListContainers lists all containers by filters.
	ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error)
	// ContainerStatus returns the status of the container.
	ContainerStatus(containerID string) (*runtimeApi.ContainerStatus, error)
	// Exec executes a command in the container.
	Exec(containerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error
}

// ImageManagerService interface should be implemented by a container image
// manager.
// The methods should be thread-safe.
type ImageManagerService interface {
	// ListImages lists the existing images.
	ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error)
	// ImageStatus returns the status of the image.
	ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error)
	// PullImage pulls an image with the authentication config.
	PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error
	// RemoveImage removes the image.
	RemoveImage(image *runtimeApi.ImageSpec) error
}
