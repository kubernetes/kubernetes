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

package kubelet

import (
	"fmt"
	"sync"

	"github.com/fsouza/go-dockerclient"
)

// FakeDockerClient is a simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	lock          sync.Mutex
	containerList []docker.APIContainers
	container     *docker.Container
	err           error
	called        []string
	stopped       []string
	pulled        []string
	Created       []string
}

func (f *FakeDockerClient) clearCalls() {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = []string{}
}

// ListContainers is a test-spy implementation of DockerInterface.ListContainers.
// It adds an entry "list" to the internal method call record.
func (f *FakeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "list")
	return f.containerList, f.err
}

// InspectContainer is a test-spy implementation of DockerInterface.InspectContainer.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "inspect")
	return f.container, f.err
}

// CreateContainer is a test-spy implementation of DockerInterface.CreateContainer.
// It adds an entry "create" to the internal method call record.
func (f *FakeDockerClient) CreateContainer(c docker.CreateContainerOptions) (*docker.Container, error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "create")
	f.Created = append(f.Created, c.Name)
	// This is not a very good fake. We'll just add this container's name to the list.
	// Docker likes to add a '/', so copy that behavior.
	name := "/" + c.Name
	f.containerList = append(f.containerList, docker.APIContainers{ID: name, Names: []string{name}})
	return &docker.Container{ID: name}, nil
}

// StartContainer is a test-spy implementation of DockerInterface.StartContainer.
// It adds an entry "start" to the internal method call record.
func (f *FakeDockerClient) StartContainer(id string, hostConfig *docker.HostConfig) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "start")
	return f.err
}

// StopContainer is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "stop" to the internal method call record.
func (f *FakeDockerClient) StopContainer(id string, timeout uint) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "stop")
	f.stopped = append(f.stopped, id)
	var newList []docker.APIContainers
	for _, container := range f.containerList {
		if container.ID != id {
			newList = append(newList, container)
		}
	}
	f.containerList = newList
	return f.err
}

// PullImage is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "pull" to the internal method call record.
func (f *FakeDockerClient) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.called = append(f.called, "pull")
	f.pulled = append(f.pulled, fmt.Sprintf("%s/%s:%s", opts.Repository, opts.Registry, opts.Tag))
	return f.err
}

// FakeDockerPuller is a stub implementation of DockerPuller.
type FakeDockerPuller struct {
	lock         sync.Mutex
	ImagesPulled []string

	// Every pull will return the first error here, and then reslice
	// to remove it. Will give nil errors if this slice is empty.
	ErrorsToInject []error
}

// Pull records the image pull attempt, and optionally injects an error.
func (f *FakeDockerPuller) Pull(image string) (err error) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.ImagesPulled = append(f.ImagesPulled, image)

	if len(f.ErrorsToInject) > 0 {
		err = f.ErrorsToInject[0]
		f.ErrorsToInject = f.ErrorsToInject[1:]
	}
	return err
}
