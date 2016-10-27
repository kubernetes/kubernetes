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

package kuberuntime

import (
	"io"
	"time"

	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// instrumentedRuntimeService wraps the RuntimeService and records the operations
// and errors metrics.
type instrumentedRuntimeService struct {
	service internalApi.RuntimeService
}

// Creates an instrumented RuntimeInterface from an existing RuntimeService.
func NewInstrumentedRuntimeService(service internalApi.RuntimeService) internalApi.RuntimeService {
	return &instrumentedRuntimeService{service: service}
}

// instrumentedImageManagerService wraps the ImageManagerService and records the operations
// and errors metrics.
type instrumentedImageManagerService struct {
	service internalApi.ImageManagerService
}

// Creates an instrumented ImageManagerService from an existing ImageManagerService.
func NewInstrumentedImageManagerService(service internalApi.ImageManagerService) internalApi.ImageManagerService {
	return &instrumentedImageManagerService{service: service}
}

// recordOperation records the duration of the operation.
func recordOperation(operation string, start time.Time) {
	metrics.RuntimeOperations.WithLabelValues(operation).Inc()
	metrics.RuntimeOperationsLatency.WithLabelValues(operation).Observe(metrics.SinceInMicroseconds(start))
}

// recordError records error for metric if an error occurred.
func recordError(operation string, err error) {
	if err != nil {
		metrics.RuntimeOperationsErrors.WithLabelValues(operation).Inc()
	}
}

func (in instrumentedRuntimeService) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	const operation = "version"
	defer recordOperation(operation, time.Now())

	out, err := in.service.Version(apiVersion)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) CreateContainer(podSandboxID string, config *runtimeApi.ContainerConfig, sandboxConfig *runtimeApi.PodSandboxConfig) (string, error) {
	const operation = "create_container"
	defer recordOperation(operation, time.Now())

	out, err := in.service.CreateContainer(podSandboxID, config, sandboxConfig)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) StartContainer(containerID string) error {
	const operation = "start_container"
	defer recordOperation(operation, time.Now())

	err := in.service.StartContainer(containerID)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) StopContainer(containerID string, timeout int64) error {
	const operation = "stop_container"
	defer recordOperation(operation, time.Now())

	err := in.service.StopContainer(containerID, timeout)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) RemoveContainer(containerID string) error {
	const operation = "remove_container"
	defer recordOperation(operation, time.Now())

	err := in.service.RemoveContainer(containerID)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) ListContainers(filter *runtimeApi.ContainerFilter) ([]*runtimeApi.Container, error) {
	const operation = "list_containers"
	defer recordOperation(operation, time.Now())

	out, err := in.service.ListContainers(filter)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) ContainerStatus(containerID string) (*runtimeApi.ContainerStatus, error) {
	const operation = "container_status"
	defer recordOperation(operation, time.Now())

	out, err := in.service.ContainerStatus(containerID)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) ExecLegacy(containerID string, cmd []string, tty bool, stdin io.Reader, stdout, stderr io.WriteCloser) error {
	const operation = "exec_legacy"
	defer recordOperation(operation, time.Now())

	err := in.service.ExecLegacy(containerID, cmd, tty, stdin, stdout, stderr)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) RunPodSandbox(config *runtimeApi.PodSandboxConfig) (string, error) {
	const operation = "run_podsandbox"
	defer recordOperation(operation, time.Now())

	out, err := in.service.RunPodSandbox(config)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) StopPodSandbox(podSandboxID string) error {
	const operation = "stop_podsandbox"
	defer recordOperation(operation, time.Now())

	err := in.service.StopPodSandbox(podSandboxID)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) RemovePodSandbox(podSandboxID string) error {
	const operation = "remove_podsandbox"
	defer recordOperation(operation, time.Now())

	err := in.service.RemovePodSandbox(podSandboxID)
	recordError(operation, err)
	return err
}

func (in instrumentedRuntimeService) PodSandboxStatus(podSandboxID string) (*runtimeApi.PodSandboxStatus, error) {
	const operation = "podsandbox_status"
	defer recordOperation(operation, time.Now())

	out, err := in.service.PodSandboxStatus(podSandboxID)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) ListPodSandbox(filter *runtimeApi.PodSandboxFilter) ([]*runtimeApi.PodSandbox, error) {
	const operation = "list_podsandbox"
	defer recordOperation(operation, time.Now())

	out, err := in.service.ListPodSandbox(filter)
	recordError(operation, err)
	return out, err
}

func (in instrumentedRuntimeService) UpdateRuntimeConfig(runtimeConfig *runtimeApi.RuntimeConfig) error {
	const operation = "update_runtime_config"
	defer recordOperation(operation, time.Now())

	err := in.service.UpdateRuntimeConfig(runtimeConfig)
	recordError(operation, err)
	return err
}

// Exec prepares a streaming endpoint to execute a command in the container.
func (in instrumentedRuntimeService) Exec(containerID string, cmd []string, tty, stdin bool) (string, error) {
	const operation = "exec"
	defer recordOperation(operation, time.Now())

	url, err := in.service.Exec(containerID, cmd, tty, stdin)
	recordError(operation, err)
	return url, err
}

// ExecSync runs a command in a container synchronously and returns stdout, stderr and exit code.
func (in instrumentedRuntimeService) ExecSync(containerID string, cmd []string, timeout int64) (string, string, int32, error) {
	const operation = "exec_sync"
	defer recordOperation(operation, time.Now())

	stdout, stderr, code, err := in.service.ExecSync(containerID, cmd, timeout)
	recordError(operation, err)
	return stdout, stderr, code, err
}

// Attach prepares a streaming endpoint to attach to a running container.
func (in instrumentedRuntimeService) Attach(containerID string, stdin bool) (string, error) {
	const operation = "attach"
	defer recordOperation(operation, time.Now())

	url, err := in.service.Attach(containerID, stdin)
	recordError(operation, err)
	return url, err
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox.
func (in instrumentedRuntimeService) PortForward(podSandboxID string, port int32) (string, error) {
	const operation = "portforward"
	defer recordOperation(operation, time.Now())

	url, err := in.service.PortForward(podSandboxID, port)
	recordError(operation, err)
	return url, err
}

func (in instrumentedImageManagerService) ListImages(filter *runtimeApi.ImageFilter) ([]*runtimeApi.Image, error) {
	const operation = "list_images"
	defer recordOperation(operation, time.Now())

	out, err := in.service.ListImages(filter)
	recordError(operation, err)
	return out, err
}

func (in instrumentedImageManagerService) ImageStatus(image *runtimeApi.ImageSpec) (*runtimeApi.Image, error) {
	const operation = "image_status"
	defer recordOperation(operation, time.Now())

	out, err := in.service.ImageStatus(image)
	recordError(operation, err)
	return out, err
}

func (in instrumentedImageManagerService) PullImage(image *runtimeApi.ImageSpec, auth *runtimeApi.AuthConfig) error {
	const operation = "pull_image"
	defer recordOperation(operation, time.Now())

	err := in.service.PullImage(image, auth)
	recordError(operation, err)
	return err
}

func (in instrumentedImageManagerService) RemoveImage(image *runtimeApi.ImageSpec) error {
	const operation = "remove_image"
	defer recordOperation(operation, time.Now())

	err := in.service.RemoveImage(image)
	recordError(operation, err)
	return err
}
