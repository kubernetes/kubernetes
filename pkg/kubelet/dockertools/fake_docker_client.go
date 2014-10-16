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

package dockertools

import (
	"fmt"
	"reflect"
	"sync"

	"github.com/fsouza/go-dockerclient"
)

// FakeDockerClient is a simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	sync.Mutex
	ContainerList []docker.APIContainers
	Container     *docker.Container
	Image         *docker.Image
	Err           error
	called        []string
	Stopped       []string
	pulled        []string
	Created       []string
}

func (f *FakeDockerClient) clearCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = []string{}
}

func (f *FakeDockerClient) AssertCalls(calls []string) (err error) {
	f.Lock()
	defer f.Unlock()

	if !reflect.DeepEqual(calls, f.called) {
		err = fmt.Errorf("expected %#v, got %#v", calls, f.called)
	}

	return
}

// ListContainers is a test-spy implementation of DockerInterface.ListContainers.
// It adds an entry "list" to the internal method call record.
func (f *FakeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "list")
	return f.ContainerList, f.Err
}

// InspectContainer is a test-spy implementation of DockerInterface.InspectContainer.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "inspect_container")
	return f.Container, f.Err
}

// InspectImage is a test-spy implementation of DockerInterface.InspectImage.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImage(name string) (*docker.Image, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "inspect_image")
	return f.Image, f.Err
}

// CreateContainer is a test-spy implementation of DockerInterface.CreateContainer.
// It adds an entry "create" to the internal method call record.
func (f *FakeDockerClient) CreateContainer(c docker.CreateContainerOptions) (*docker.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "create")
	f.Created = append(f.Created, c.Name)
	// This is not a very good fake. We'll just add this container's name to the list.
	// Docker likes to add a '/', so copy that behavior.
	name := "/" + c.Name
	f.ContainerList = append(f.ContainerList, docker.APIContainers{ID: name, Names: []string{name}, Image: c.Config.Image})
	return &docker.Container{ID: name}, nil
}

// StartContainer is a test-spy implementation of DockerInterface.StartContainer.
// It adds an entry "start" to the internal method call record.
func (f *FakeDockerClient) StartContainer(id string, hostConfig *docker.HostConfig) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "start")
	return f.Err
}

// StopContainer is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "stop" to the internal method call record.
func (f *FakeDockerClient) StopContainer(id string, timeout uint) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "stop")
	f.Stopped = append(f.Stopped, id)
	var newList []docker.APIContainers
	for _, container := range f.ContainerList {
		if container.ID != id {
			newList = append(newList, container)
		}
	}
	f.ContainerList = newList
	return f.Err
}

// Logs is a test-spy implementation of DockerInterface.Logs.
// It adds an entry "logs" to the internal method call record.
func (f *FakeDockerClient) Logs(opts docker.LogsOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "logs")
	return f.Err
}

// PullImage is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "pull" to the internal method call record.
func (f *FakeDockerClient) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "pull")
	f.pulled = append(f.pulled, fmt.Sprintf("%s/%s:%s", opts.Repository, opts.Registry, opts.Tag))
	return f.Err
}

// FakeDockerPuller is a stub implementation of DockerPuller.
type FakeDockerPuller struct {
	sync.Mutex

	HasImages    []string
	ImagesPulled []string

	// Every pull will return the first error here, and then reslice
	// to remove it. Will give nil errors if this slice is empty.
	ErrorsToInject []error
}

// Pull records the image pull attempt, and optionally injects an error.
func (f *FakeDockerPuller) Pull(image string) (err error) {
	f.Lock()
	defer f.Unlock()
	f.ImagesPulled = append(f.ImagesPulled, image)

	if len(f.ErrorsToInject) > 0 {
		err = f.ErrorsToInject[0]
		f.ErrorsToInject = f.ErrorsToInject[1:]
	}
	return err
}

func (f *FakeDockerPuller) IsImagePresent(name string) (bool, error) {
	f.Lock()
	defer f.Unlock()
	if f.HasImages == nil {
		return true, nil
	}
	for _, s := range f.HasImages {
		if s == name {
			return true, nil
		}
	}
	return false, nil
}
