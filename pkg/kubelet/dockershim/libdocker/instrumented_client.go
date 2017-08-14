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

package libdocker

import (
	"time"

	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

// instrumentedInterface wraps the Interface and records the operations
// and errors metrics.
type instrumentedInterface struct {
	client Interface
}

// Creates an instrumented Interface from an existing Interface.
func NewInstrumentedInterface(dockerClient Interface) Interface {
	return instrumentedInterface{
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

func (in instrumentedInterface) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	const operation = "list_containers"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListContainers(options)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	const operation = "inspect_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectContainer(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockercontainer.ContainerCreateCreatedBody, error) {
	const operation = "create_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateContainer(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) StartContainer(id string) error {
	const operation = "start_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StartContainer(id)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) StopContainer(id string, timeout time.Duration) error {
	const operation = "stop_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StopContainer(id, timeout)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	const operation = "remove_container"
	defer recordOperation(operation, time.Now())

	err := in.client.RemoveContainer(id, opts)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) UpdateContainerResources(id string, updateConfig dockercontainer.UpdateConfig) error {
	const operation = "update_container"
	defer recordOperation(operation, time.Now())

	err := in.client.UpdateContainerResources(id, updateConfig)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) InspectImageByRef(image string) (*dockertypes.ImageInspect, error) {
	const operation = "inspect_image"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectImageByRef(image)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) InspectImageByID(image string) (*dockertypes.ImageInspect, error) {
	const operation = "inspect_image"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectImageByID(image)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.ImageSummary, error) {
	const operation = "list_images"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListImages(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) PullImage(imageID string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	const operation = "pull_image"
	defer recordOperation(operation, time.Now())
	err := in.client.PullImage(imageID, auth, opts)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error) {
	const operation = "remove_image"
	defer recordOperation(operation, time.Now())

	imageDelete, err := in.client.RemoveImage(image, opts)
	recordError(operation, err)
	return imageDelete, err
}

func (in instrumentedInterface) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	const operation = "logs"
	defer recordOperation(operation, time.Now())

	err := in.client.Logs(id, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) Version() (*dockertypes.Version, error) {
	const operation = "version"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Version()
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) Info() (*dockertypes.Info, error) {
	const operation = "info"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Info()
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.IDResponse, error) {
	const operation = "create_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateExec(id, opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	const operation = "start_exec"
	defer recordOperation(operation, time.Now())

	err := in.client.StartExec(startExec, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	const operation = "inspect_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectExec(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	const operation = "attach"
	defer recordOperation(operation, time.Now())

	err := in.client.AttachToContainer(id, opts, sopts)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	const operation = "image_history"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ImageHistory(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedInterface) ResizeExecTTY(id string, height, width uint) error {
	const operation = "resize_exec"
	defer recordOperation(operation, time.Now())

	err := in.client.ResizeExecTTY(id, height, width)
	recordError(operation, err)
	return err
}

func (in instrumentedInterface) ResizeContainerTTY(id string, height, width uint) error {
	const operation = "resize_container"
	defer recordOperation(operation, time.Now())

	err := in.client.ResizeContainerTTY(id, height, width)
	recordError(operation, err)
	return err
}
