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

// This package is originally inherited by k8s.io/cri-api/pkg/apis:
// https://github.com/kubernetes/kubernetes/blob/2dc2b1e/staging/src/k8s.io/cri-api/pkg/apis/services.go
//
// We made this API kubelet internal, because it maps to the internal
// intermediate CRI types of the kubelet.
package cri

import (
	"time"
)

// RuntimeVersioner contains methods for runtime name, version and API version.
type RuntimeVersioner interface {
	// Version returns the runtime name, runtime version and runtime API version
	Version(apiVersion string) (*VersionResponse, error)
}

// ContainerManager contains methods to manipulate containers managed by a
// container runtime. The methods are thread-safe.
type ContainerManager interface {
	// CreateContainer creates a new container in specified PodSandbox.
	CreateContainer(podSandboxID string, config *ContainerConfig, sandboxConfig *PodSandboxConfig) (string, error)
	// StartContainer starts the container.
	StartContainer(containerID string) error
	// StopContainer stops a running container with a grace period (i.e., timeout).
	StopContainer(containerID string, timeout int64) error
	// RemoveContainer removes the container.
	RemoveContainer(containerID string) error
	// ListContainers lists all containers by filters.
	ListContainers(filter *ContainerFilter) ([]*Container, error)
	// ContainerStatus returns the status of the container.
	ContainerStatus(containerID string) (*ContainerStatus, error)
	// UpdateContainerResources updates the cgroup resources for the container.
	UpdateContainerResources(containerID string, resources *LinuxContainerResources) error
	// ExecSync executes a command in the container, and returns the stdout output.
	// If command exits with a non-zero exit code, an error is returned.
	ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error)
	// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
	Exec(*ExecRequest) (*ExecResponse, error)
	// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
	Attach(req *AttachRequest) (*AttachResponse, error)
	// ReopenContainerLog asks runtime to reopen the stdout/stderr log file
	// for the container. If it returns error, new container log file MUST NOT
	// be created.
	ReopenContainerLog(ContainerID string) error
}

// PodSandboxManager contains methods for operating on PodSandboxes. The methods
// are thread-safe.
type PodSandboxManager interface {
	// RunPodSandbox creates and starts a pod-level sandbox. Runtimes should ensure
	// the sandbox is in ready state.
	RunPodSandbox(config *PodSandboxConfig, runtimeHandler string) (string, error)
	// StopPodSandbox stops the sandbox. If there are any running containers in the
	// sandbox, they should be force terminated.
	StopPodSandbox(podSandboxID string) error
	// RemovePodSandbox removes the sandbox. If there are running containers in the
	// sandbox, they should be forcibly removed.
	RemovePodSandbox(podSandboxID string) error
	// PodSandboxStatus returns the Status of the PodSandbox.
	PodSandboxStatus(podSandboxID string) (*PodSandboxStatus, error)
	// ListPodSandbox returns a list of Sandbox.
	ListPodSandbox(filter *PodSandboxFilter) ([]*PodSandbox, error)
	// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
	PortForward(*PortForwardRequest) (*PortForwardResponse, error)
}

// ContainerStatsManager contains methods for retrieving the container
// statistics.
type ContainerStatsManager interface {
	// ContainerStats returns stats of the container. If the container does not
	// exist, the call returns an error.
	ContainerStats(containerID string) (*ContainerStats, error)
	// ListContainerStats returns stats of all running containers.
	ListContainerStats(filter *ContainerStatsFilter) ([]*ContainerStats, error)
	// PodSandboxStats returns stats of the pod. If the pod does not
	// exist, the call returns an error.
	PodSandboxStats(podSandboxID string) (*PodSandboxStats, error)
	// ListPodSandboxStats returns stats of all running pods.
	ListPodSandboxStats(filter *PodSandboxStatsFilter) ([]*PodSandboxStats, error)
}

// RuntimeService interface should be implemented by a container runtime.
// The methods should be thread-safe.
type RuntimeService interface {
	RuntimeVersioner
	ContainerManager
	PodSandboxManager
	ContainerStatsManager

	// UpdateRuntimeConfig updates runtime configuration if specified
	UpdateRuntimeConfig(runtimeConfig *RuntimeConfig) error
	// Status returns the status of the runtime.
	Status() (*RuntimeStatus, error)
	// Retrieve the currently used CRI API version.
	APIVersion() APIVersion
}

// APIVersion is the type for the CRI API version.
type APIVersion string

const (
	// APIVersionV1 references the v1 CRI API.
	APIVersionV1 APIVersion = "v1"

	// APIVersionV1 references the v1alpha2 CRI API.
	APIVersionV1alpha2 APIVersion = "v1alpha2"
)

// ImageManagerService interface should be implemented by a container image
// manager.
// The methods should be thread-safe.
type ImageManagerService interface {
	// ListImages lists the existing images.
	ListImages(filter *ImageFilter) ([]*Image, error)
	// ImageStatus returns the status of the image.
	ImageStatus(image *ImageSpec) (*Image, error)
	// PullImage pulls an image with the authentication config.
	PullImage(image *ImageSpec, auth *AuthConfig, podSandboxConfig *PodSandboxConfig) (string, error)
	// RemoveImage removes the image.
	RemoveImage(image *ImageSpec) error
	// ImageFsInfo returns information of the filesystem that is used to store images.
	ImageFsInfo() ([]*FilesystemUsage, error)
}
