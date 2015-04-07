/*
Copyright 2015 Google Inc. All rights reserved.

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

package metrics

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	docker "github.com/fsouza/go-dockerclient"
)

var _ dockertools.DockerInterface = instrumentedDockerInterface{}

type instrumentedDockerInterface struct {
	client dockertools.DockerInterface
}

// Creates an instrumented DockerInterface from an existing DockerInterface.
func NewInstrumentedDockerInterface(dockerClient dockertools.DockerInterface) dockertools.DockerInterface {
	return instrumentedDockerInterface{
		client: dockerClient,
	}
}

func (self instrumentedDockerInterface) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("list_containers").Observe(SinceInMicroseconds(start))
	}()
	return self.client.ListContainers(options)
}

func (self instrumentedDockerInterface) InspectContainer(id string) (*docker.Container, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("inspect_container").Observe(SinceInMicroseconds(start))
	}()
	return self.client.InspectContainer(id)
}

func (self instrumentedDockerInterface) CreateContainer(opts docker.CreateContainerOptions) (*docker.Container, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("create_container").Observe(SinceInMicroseconds(start))
	}()
	return self.client.CreateContainer(opts)
}

func (self instrumentedDockerInterface) StartContainer(id string, hostConfig *docker.HostConfig) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("start_container").Observe(SinceInMicroseconds(start))
	}()
	return self.client.StartContainer(id, hostConfig)
}

func (self instrumentedDockerInterface) StopContainer(id string, timeout uint) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("stop_container").Observe(SinceInMicroseconds(start))
	}()
	return self.client.StopContainer(id, timeout)
}

func (self instrumentedDockerInterface) RemoveContainer(opts docker.RemoveContainerOptions) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("remove_container").Observe(SinceInMicroseconds(start))
	}()
	return self.client.RemoveContainer(opts)
}

func (self instrumentedDockerInterface) InspectImage(image string) (*docker.Image, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("inspect_image").Observe(SinceInMicroseconds(start))
	}()
	return self.client.InspectImage(image)
}

func (self instrumentedDockerInterface) ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("list_images").Observe(SinceInMicroseconds(start))
	}()
	return self.client.ListImages(opts)
}

func (self instrumentedDockerInterface) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("pull_image").Observe(SinceInMicroseconds(start))
	}()
	return self.client.PullImage(opts, auth)
}

func (self instrumentedDockerInterface) RemoveImage(image string) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("remove_image").Observe(SinceInMicroseconds(start))
	}()
	return self.client.RemoveImage(image)
}

func (self instrumentedDockerInterface) Logs(opts docker.LogsOptions) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("logs").Observe(SinceInMicroseconds(start))
	}()
	return self.client.Logs(opts)
}

func (self instrumentedDockerInterface) Version() (*docker.Env, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("version").Observe(SinceInMicroseconds(start))
	}()
	return self.client.Version()
}

func (self instrumentedDockerInterface) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("create_exec").Observe(SinceInMicroseconds(start))
	}()
	return self.client.CreateExec(opts)
}

func (self instrumentedDockerInterface) StartExec(startExec string, opts docker.StartExecOptions) error {
	start := time.Now()
	defer func() {
		DockerOperationsLatency.WithLabelValues("start_exec").Observe(SinceInMicroseconds(start))
	}()
	return self.client.StartExec(startExec, opts)
}
