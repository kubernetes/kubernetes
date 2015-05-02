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
	"github.com/fsouza/go-dockerclient"
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

func (in instrumentedDockerInterface) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("list_containers").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.ListContainers(options)
}

func (in instrumentedDockerInterface) InspectContainer(id string) (*docker.Container, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("inspect_container").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.InspectContainer(id)
}

func (in instrumentedDockerInterface) CreateContainer(opts docker.CreateContainerOptions) (*docker.Container, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("create_container").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.CreateContainer(opts)
}

func (in instrumentedDockerInterface) StartContainer(id string, hostConfig *docker.HostConfig) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("start_container").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.StartContainer(id, hostConfig)
}

func (in instrumentedDockerInterface) StopContainer(id string, timeout uint) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("stop_container").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.StopContainer(id, timeout)
}

func (in instrumentedDockerInterface) RemoveContainer(opts docker.RemoveContainerOptions) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("remove_container").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.RemoveContainer(opts)
}

func (in instrumentedDockerInterface) InspectImage(image string) (*docker.Image, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("inspect_image").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.InspectImage(image)
}

func (in instrumentedDockerInterface) ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("list_images").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.ListImages(opts)
}

func (in instrumentedDockerInterface) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("pull_image").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.PullImage(opts, auth)
}

func (in instrumentedDockerInterface) RemoveImage(image string) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("remove_image").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.RemoveImage(image)
}

func (in instrumentedDockerInterface) Logs(opts docker.LogsOptions) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("logs").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.Logs(opts)
}

func (in instrumentedDockerInterface) Version() (*docker.Env, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("version").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.Version()
}

func (in instrumentedDockerInterface) Info() (*docker.Env, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("version").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.Info()
}

func (in instrumentedDockerInterface) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("create_exec").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.CreateExec(opts)
}

func (in instrumentedDockerInterface) StartExec(startExec string, opts docker.StartExecOptions) error {
	start := time.Now()
	defer func() {
		metrics.DockerOperationsLatency.WithLabelValues("start_exec").Observe(metrics.SinceInMicroseconds(start))
	}()
	return in.client.StartExec(startExec, opts)
}
