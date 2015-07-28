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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	docker "github.com/fsouza/go-dockerclient"
)

type instrumentedDockerInterface struct {
	client DockerInterface
}

// Creates an instrumented DockerInterface from an existing DockerInterface.
func NewInstrumentedDockerInterface(dockerClient DockerInterface) DockerInterface {
	return instrumentedDockerInterface{
		client: dockerClient,
	}
}

// Record the duration of the operation.
func recordOperation(operation string, start time.Time) {
	metrics.DockerOperationsLatency.WithLabelValues(operation).Observe(metrics.SinceInMicroseconds(start))
}

// Record error for metric if an error occured.
func recordError(operation string, err error) {
	if err != nil {
		metrics.DockerErrors.WithLabelValues(operation).Inc()
	}
}

func (in instrumentedDockerInterface) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	const operation = "list_containers"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListContainers(options)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) InspectContainer(id string) (*docker.Container, error) {
	const operation = "inspect_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectContainer(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) CreateContainer(opts docker.CreateContainerOptions) (*docker.Container, error) {
	const operation = "create_container"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateContainer(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) StartContainer(id string, hostConfig *docker.HostConfig) error {
	const operation = "start_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StartContainer(id, hostConfig)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) StopContainer(id string, timeout uint) error {
	const operation = "stop_container"
	defer recordOperation(operation, time.Now())

	err := in.client.StopContainer(id, timeout)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) RemoveContainer(opts docker.RemoveContainerOptions) error {
	const operation = "remove_container"
	defer recordOperation(operation, time.Now())

	err := in.client.RemoveContainer(opts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) InspectImage(image string) (*docker.Image, error) {
	const operation = "inspect_image"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectImage(image)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error) {
	const operation = "list_images"
	defer recordOperation(operation, time.Now())

	out, err := in.client.ListImages(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	const operation = "pull_image"
	defer recordOperation(operation, time.Now())

	err := in.client.PullImage(opts, auth)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) RemoveImage(image string) error {
	const operation = "remove_image"
	defer recordOperation(operation, time.Now())

	err := in.client.RemoveImage(image)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) Logs(opts docker.LogsOptions) error {
	const operation = "logs"
	defer recordOperation(operation, time.Now())

	err := in.client.Logs(opts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) Version() (*docker.Env, error) {
	const operation = "version"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Version()
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) Info() (*docker.Env, error) {
	const operation = "info"
	defer recordOperation(operation, time.Now())

	out, err := in.client.Info()
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	const operation = "create_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.CreateExec(opts)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) StartExec(startExec string, opts docker.StartExecOptions) error {
	const operation = "start_exec"
	defer recordOperation(operation, time.Now())

	err := in.client.StartExec(startExec, opts)
	recordError(operation, err)
	return err
}

func (in instrumentedDockerInterface) InspectExec(id string) (*docker.ExecInspect, error) {
	const operation = "inspect_exec"
	defer recordOperation(operation, time.Now())

	out, err := in.client.InspectExec(id)
	recordError(operation, err)
	return out, err
}

func (in instrumentedDockerInterface) AttachToContainer(opts docker.AttachToContainerOptions) error {
	const operation = "attach"
	defer recordOperation(operation, time.Now())

	err := in.client.AttachToContainer(opts)
	recordError(operation, err)
	return err
}
