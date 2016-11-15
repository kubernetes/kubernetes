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
	"time"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// RuntimeVersioner contains methods for runtime name, version and API version.
type RuntimeVersioner interface {
	// Version returns the runtime name, runtime version and runtime API version
	Version(apiVersion string) (*runtimeApi.VersionResponse, error)
}

// ContainerManager contains methods to manipulate containers managed by a
// container runtime. The methods are thread-safe.
type ContainerManager interface {
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
	// ExecSync executes a command in the container, and returns the stdout output.
	// If command exits with a non-zero exit code, an error is returned.
	ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error)
	// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
	Exec(*runtimeApi.ExecRequest) (*runtimeApi.ExecResponse, error)
	// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
	Attach(req *runtimeApi.AttachRequest) (*runtimeApi.AttachResponse, error)
}

// PodSandboxManager contains methods for operating on PodSandboxes. The methods
// are thread-safe.
type PodSandboxManager interface {
	// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
	// the sandbox is in ready state.
	RunPodSandbox(config *runtimeApi.PodSandboxConfig) (string, error)
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
	// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
	PortForward(*runtimeApi.PortForwardRequest) (*runtimeApi.PortForwardResponse, error)
}

// RuntimeService interface should be implemented by a container runtime.
// The methods should be thread-safe.
type RuntimeService interface {
	RuntimeVersioner
	ContainerManager
	PodSandboxManager

	// UpdateRuntimeConfig updates runtime configuration if specified
	UpdateRuntimeConfig(runtimeConfig *runtimeApi.RuntimeConfig) error
	// Status returns the status of the runtime.
	Status() (*runtimeApi.RuntimeStatus, error)
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
