/*
Copyright 2014 Google Inc. All rights reserved.

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

package container

import "github.com/fsouza/go-dockerclient"

// ContainerRuntimeInterface is the abstract interface for container runtime.
type ContainerRuntimeInterface interface {
	ListContainers(options ListContainersOptions) ([]Container, error)
	InspectContainer(id string) (*docker.Container, error)
	CreateContainer(docker.CreateContainerOptions) (*docker.Container, error)
	StartContainer(id string, hostConfig *docker.HostConfig) error
	StopContainer(id string, timeout uint) error
	RemoveContainer(opts docker.RemoveContainerOptions) error
	InspectImage(image string) (*docker.Image, error)
	ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error)
	PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error
	RemoveImage(image string) error
	Logs(opts docker.LogsOptions) error
	Version() (*docker.Env, error)
	CreateExec(docker.CreateExecOptions) (*docker.Exec, error)
	StartExec(string, docker.StartExecOptions) error
}
