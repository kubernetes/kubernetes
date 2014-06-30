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
	"github.com/fsouza/go-dockerclient"
)

// A simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	containerList []docker.APIContainers
	container     *docker.Container
	err           error
	called        []string
	stopped       []string
	Created       []string
}

func (f *FakeDockerClient) clearCalls() {
	f.called = []string{}
}

func (f *FakeDockerClient) appendCall(call string) {
	f.called = append(f.called, call)
}

func (f *FakeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	f.appendCall("list")
	return f.containerList, f.err
}

func (f *FakeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	f.appendCall("inspect")
	return f.container, f.err
}

func (f *FakeDockerClient) CreateContainer(c docker.CreateContainerOptions) (*docker.Container, error) {
	f.appendCall("create")
	f.Created = append(f.Created, c.Name)
	// This is not a very good fake. We'll just add this container's name to the list.
	// Docker likes to add a '/', so copy that behavior.
	f.containerList = append(f.containerList, docker.APIContainers{ID: c.Name, Names: []string{"/" + c.Name}})
	return &docker.Container{ID: "/" + c.Name}, nil
}

func (f *FakeDockerClient) StartContainer(id string, hostConfig *docker.HostConfig) error {
	f.appendCall("start")
	return f.err
}

func (f *FakeDockerClient) StopContainer(id string, timeout uint) error {
	f.appendCall("stop")
	f.stopped = append(f.stopped, id)
	return f.err
}

type FakeDockerPuller struct {
	ImagesPulled []string

	// Every pull will return the first error here, and then reslice
	// to remove it. Will give nil errors if this slice is empty.
	ErrorsToInject []error
}

// Records the image pull attempt, and optionally injects an error.
func (f *FakeDockerPuller) Pull(image string) error {
	f.ImagesPulled = append(f.ImagesPulled, image)

	if n := len(f.ErrorsToInject); n > 0 {
		err := f.ErrorsToInject[0]
		f.ErrorsToInject = f.ErrorsToInject[:n-1]
		return err
	}
	return nil
}
