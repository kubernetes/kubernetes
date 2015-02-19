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

// Runtime is the abstract interface for container runtime.
type Runtime interface {
	ListContainers(opts ListContainersOptions) ([]*Container, error)
	InspectContainer(id string) (*Container, error)
	CreateContainer(opts CreateContainerOptions) (*Container, error)
	StartContainer(id string, hostConfig *HostConfig) error
	StopContainer(id string, timeout uint) error
	RemoveContainer(opts RemoveContainerOptions) error
	InspectImage(image string) (*Image, error)
	ListImages(opts ListImagesOptions) ([]*Image, error)
	PullImage(opts PullImageOptions) error
	RemoveImage(image string) error
	Logs(opts LogsOptions) error
	Version() ([]string, error)
	CreateExec(opts CreateExecOptions) (*Exec, error)
	StartExec(string, StartExecOptions) error
}
