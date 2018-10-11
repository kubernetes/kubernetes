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
	dockerimagetypes "github.com/docker/docker/api/types/image"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/metrics"
)

//actions of docker
const (
	list_containers            = "list_containers"
	inspect_container          = "inspect_container"
	inspect_container_withsize = "inspect_container_withsize"
	create_container           = "create_container"
	start_container            = "start_container"
	stop_container             = "stop_container"
	remove_container           = "remove_container"
	update_container           = "update_container"
	inspect_image              = "inspect_image"
	list_images                = "list_images"
	pull_image                 = "pull_image"
	remove_image               = "remove_image"
	logs                       = "logs"
	version                    = "version"
	info                       = "info"
	create_exec                = "create_exec"
	start_exec                 = "start_exec"
	inspect_exec               = "inspect_exec"
	attach                     = "attach"
	image_history              = "image_history"
	resize_exec                = "resize_exec"
	resize_container           = "resize_container"
	stats                      = "stats"
)

// instrumentedInterface wraps the Interface and records the operations
// and errors metrics.
type instrumentedInterface struct {
	client Interface
}

// NewInstrumentedInterface creates an instrumented Interface from an existing Interface.
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
	defer recordOperation(list_containers, time.Now())

	out, err := in.client.ListContainers(options)
	recordError(list_containers, err)
	return out, err
}

func (in instrumentedInterface) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	defer recordOperation(inspect_container, time.Now())

	out, err := in.client.InspectContainer(id)
	recordError(inspect_container, err)
	return out, err
}

func (in instrumentedInterface) InspectContainerWithSize(id string) (*dockertypes.ContainerJSON, error) {
	defer recordOperation(inspect_container_withsize, time.Now())

	out, err := in.client.InspectContainerWithSize(id)
	recordError(inspect_container_withsize, err)
	return out, err
}

func (in instrumentedInterface) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockercontainer.ContainerCreateCreatedBody, error) {
	defer recordOperation(create_container, time.Now())

	out, err := in.client.CreateContainer(opts)
	recordError(create_container, err)
	return out, err
}

func (in instrumentedInterface) StartContainer(id string) error {
	defer recordOperation(start_container, time.Now())

	err := in.client.StartContainer(id)
	recordError(start_container, err)
	return err
}

func (in instrumentedInterface) StopContainer(id string, timeout time.Duration) error {
	defer recordOperation(stop_container, time.Now())

	err := in.client.StopContainer(id, timeout)
	recordError(stop_container, err)
	return err
}

func (in instrumentedInterface) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	defer recordOperation(remove_container, time.Now())

	err := in.client.RemoveContainer(id, opts)
	recordError(remove_container, err)
	return err
}

func (in instrumentedInterface) UpdateContainerResources(id string, updateConfig dockercontainer.UpdateConfig) error {
	defer recordOperation(update_container, time.Now())

	err := in.client.UpdateContainerResources(id, updateConfig)
	recordError(update_container, err)
	return err
}

func (in instrumentedInterface) InspectImageByRef(image string) (*dockertypes.ImageInspect, error) {
	defer recordOperation(inspect_image, time.Now())

	out, err := in.client.InspectImageByRef(image)
	recordError(inspect_image, err)
	return out, err
}

func (in instrumentedInterface) InspectImageByID(image string) (*dockertypes.ImageInspect, error) {
	defer recordOperation(inspect_image, time.Now())

	out, err := in.client.InspectImageByID(image)
	recordError(inspect_image, err)
	return out, err
}

func (in instrumentedInterface) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.ImageSummary, error) {
	defer recordOperation(list_images, time.Now())

	out, err := in.client.ListImages(opts)
	recordError(list_images, err)
	return out, err
}

func (in instrumentedInterface) PullImage(imageID string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	defer recordOperation(pull_image, time.Now())
	err := in.client.PullImage(imageID, auth, opts)
	recordError(pull_image, err)
	return err
}

func (in instrumentedInterface) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDeleteResponseItem, error) {
	defer recordOperation(remove_image, time.Now())

	imageDelete, err := in.client.RemoveImage(image, opts)
	recordError(remove_image, err)
	return imageDelete, err
}

func (in instrumentedInterface) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	defer recordOperation(logs, time.Now())

	err := in.client.Logs(id, opts, sopts)
	recordError(logs, err)
	return err
}

func (in instrumentedInterface) Version() (*dockertypes.Version, error) {
	defer recordOperation(version, time.Now())

	out, err := in.client.Version()
	recordError(version, err)
	return out, err
}

func (in instrumentedInterface) Info() (*dockertypes.Info, error) {
	defer recordOperation(info, time.Now())

	out, err := in.client.Info()
	recordError(info, err)
	return out, err
}

func (in instrumentedInterface) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.IDResponse, error) {
	defer recordOperation(create_exec, time.Now())

	out, err := in.client.CreateExec(id, opts)
	recordError(create_exec, err)
	return out, err
}

func (in instrumentedInterface) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	defer recordOperation(start_exec, time.Now())

	err := in.client.StartExec(startExec, opts, sopts)
	recordError(start_exec, err)
	return err
}

func (in instrumentedInterface) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	defer recordOperation(inspect_exec, time.Now())

	out, err := in.client.InspectExec(id)
	recordError(inspect_exec, err)
	return out, err
}

func (in instrumentedInterface) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	defer recordOperation(attach, time.Now())

	err := in.client.AttachToContainer(id, opts, sopts)
	recordError(attach, err)
	return err
}

func (in instrumentedInterface) ImageHistory(id string) ([]dockerimagetypes.HistoryResponseItem, error) {
	defer recordOperation(image_history, time.Now())

	out, err := in.client.ImageHistory(id)
	recordError(image_history, err)
	return out, err
}

func (in instrumentedInterface) ResizeExecTTY(id string, height, width uint) error {
	defer recordOperation(resize_exec, time.Now())

	err := in.client.ResizeExecTTY(id, height, width)
	recordError(resize_exec, err)
	return err
}

func (in instrumentedInterface) ResizeContainerTTY(id string, height, width uint) error {
	defer recordOperation(resize_container, time.Now())

	err := in.client.ResizeContainerTTY(id, height, width)
	recordError(resize_container, err)
	return err
}

func (in instrumentedInterface) GetContainerStats(id string) (*dockertypes.StatsJSON, error) {
	defer recordOperation(stats, time.Now())

	out, err := in.client.GetContainerStats(id)
	recordError(stats, err)
	return out, err
}
