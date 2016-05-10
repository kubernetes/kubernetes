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
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// instrumentedDockerInterface wraps the DockerInterface and records the operations
// and errors metrics.
type instrumentedDockerInterface struct {
	client DockerInterface
}

// Creates an instrumented DockerInterface from an existing DockerInterface.
func newInstrumentedDockerInterface(dockerClient DockerInterface) DockerInterface {
	return instrumentedDockerInterface{
		client: dockerClient,
	}
}

// recordOperation records the duration of the operation.
func recordOperation(operation string, start time.Time) {
	metrics.DockerOperations.WithLabelValues(operation).Inc()
	metrics.DockerOperationsLatency.WithLabelValues(operation).Observe(metrics.SinceInMicroseconds(start))
}

// recordError records error for metric if an error occurred.
func recordError(operation string, err error) {
	if err != nil {
		if _, ok := err.(operationTimeout); ok {
			metrics.DockerOperationsTimeout.WithLabelValues(operation).Inc()
		}
		// Docker operation timeout error is also a docker error, so we don't add else here.
		metrics.DockerOperationsErrors.WithLabelValues(operation).Inc()
	}
}

func (in instrumentedDockerInterface) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	const operation = "list_containers"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListContainers(options)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	const operation = "inspect_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectContainer(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error) {
	const operation = "create_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateContainer(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) StartContainer(id string) error {
	const operation = "start_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StartContainer(id)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) StopContainer(id string, timeout int) error {
	const operation = "stop_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StopContainer(id, timeout)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	const operation = "remove_container"
	defer recordOperation(operation, time.Now())

	err := in.client.RemoveContainer(id, opts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) InspectImage(image string) (*dockertypes.ImageInspect, error) {
	const operation = "inspect_image"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectImage(image)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error) {
	const operation = "list_images"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListImages(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) PullImage(imageID string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	const operation = "pull_image"
	defer recordOperation(operation, time.Now())
	err := in.client.PullImage(imageID, auth, opts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error) {
	const operation = "remove_image"
	defer recordOperation(operation, time.Now())

	imageDelete, err := in.client.RemoveImage(image, opts)
	recordError(operation, err)
	return imageDelete, err
}

func (in instrumentedDockerInterface) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	const operation = "logs"
	defer recordOperation(operation, time.Now())

	err := in.client.Logs(id, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) Version() (*dockertypes.Version, error) {
	const operation = "version"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Version()
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) Info() (*dockertypes.Info, error) {
	const operation = "info"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Info()
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error) {
	const operation = "create_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateExec(id, opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	const operation = "start_exec"
	defer recordOperation(operation, time.Now())

	err := in.client.StartExec(startExec, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	const operation = "inspect_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectExec(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	const operation = "attach"
	defer recordOperation(operation, time.Now())

	err := in.client.AttachToContainer(id, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	const operation = "image_history"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ImageHistory(id)
	recordError(operation, err)
	return out, err
}
