/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"sync"

	docker "github.com/fsouza/go-dockerclient"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// FakeDockerClient is a simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	sync.Mutex
	ContainerList       []docker.APIContainers
	ExitedContainerList []docker.APIContainers
	Container           *docker.Container
	ContainerMap        map[string]*docker.Container
	Image               *docker.Image
	Images              []docker.APIImages
	Errors              map[string]error
	called              []string
	Stopped             []string
	pulled              []string
	Created             []string
	Removed             []string
	RemovedImages       util.StringSet
	VersionInfo         docker.Env
	Information         docker.Env
	ExecInspect         *docker.ExecInspect
	execCmd             []string
}

func (f *FakeDockerClient) ClearCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = []string{}
	f.Stopped = []string{}
	f.pulled = []string{}
	f.Created = []string{}
	f.Removed = []string{}
}

func (f *FakeDockerClient) AssertCalls(calls []string) (err error) {
	f.Lock()
	defer f.Unlock()

	if !reflect.DeepEqual(calls, f.called) {
		err = fmt.Errorf("expected %#v, got %#v", calls, f.called)
	}

	return
}

func (f *FakeDockerClient) AssertCreated(created []string) error {
	f.Lock()
	defer f.Unlock()

	actualCreated := []string{}
	for _, c := range f.Created {
		dockerName, _, err := ParseDockerName(c)
		if err != nil {
			return fmt.Errorf("unexpected error: %v", err)
		}
		actualCreated = append(actualCreated, dockerName.ContainerName)
	}
	sort.StringSlice(created).Sort()
	sort.StringSlice(actualCreated).Sort()
	if !reflect.DeepEqual(created, actualCreated) {
		return fmt.Errorf("expected %#v, got %#v", created, actualCreated)
	}
	return nil
}

func (f *FakeDockerClient) AssertStopped(stopped []string) error {
	f.Lock()
	defer f.Unlock()
	sort.StringSlice(stopped).Sort()
	sort.StringSlice(f.Stopped).Sort()
	if !reflect.DeepEqual(stopped, f.Stopped) {
		return fmt.Errorf("expected %#v, got %#v", stopped, f.Stopped)
	}
	return nil
}

func (f *FakeDockerClient) AssertUnorderedCalls(calls []string) (err error) {
	f.Lock()
	defer f.Unlock()

	expected := make([]string, len(calls))
	actual := make([]string, len(f.called))
	copy(expected, calls)
	copy(actual, f.called)

	sort.StringSlice(expected).Sort()
	sort.StringSlice(actual).Sort()

	if !reflect.DeepEqual(actual, expected) {
		err = fmt.Errorf("expected(sorted) %#v, got(sorted) %#v", expected, actual)
	}
	return
}

func (f *FakeDockerClient) popError(op string) error {
	if f.Errors == nil {
		return nil
	}
	err, ok := f.Errors[op]
	if ok {
		delete(f.Errors, op)
		return err
	} else {
		return nil
	}
}

// ListContainers is a test-spy implementation of DockerInterface.ListContainers.
// It adds an entry "list" to the internal method call record.
func (f *FakeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "list")
	err := f.popError("list")
	if options.All {
		return append(f.ContainerList, f.ExitedContainerList...), err
	}
	return append([]docker.APIContainers{}, f.ContainerList...), err
}

// InspectContainer is a test-spy implementation of DockerInterface.InspectContainer.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "inspect_container")
	err := f.popError("inspect_container")
	if f.ContainerMap != nil {
		if container, ok := f.ContainerMap[id]; ok {
			return container, err
		}
	}
	return f.Container, err
}

// InspectImage is a test-spy implementation of DockerInterface.InspectImage.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImage(name string) (*docker.Image, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "inspect_image")
	err := f.popError("inspect_image")
	return f.Image, err
}

// CreateContainer is a test-spy implementation of DockerInterface.CreateContainer.
// It adds an entry "create" to the internal method call record.
func (f *FakeDockerClient) CreateContainer(c docker.CreateContainerOptions) (*docker.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "create")
	err := f.popError("create")
	if err == nil {
		f.Created = append(f.Created, c.Name)
		// This is not a very good fake. We'll just add this container's name to the list.
		// Docker likes to add a '/', so copy that behavior.
		name := "/" + c.Name
		f.ContainerList = append(f.ContainerList, docker.APIContainers{ID: name, Names: []string{name}, Image: c.Config.Image})
		return &docker.Container{ID: name}, nil
	}
	return nil, err
}

// StartContainer is a test-spy implementation of DockerInterface.StartContainer.
// It adds an entry "start" to the internal method call record.
func (f *FakeDockerClient) StartContainer(id string, hostConfig *docker.HostConfig) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "start")
	err := f.popError("start")
	if err == nil {

		f.Container = &docker.Container{
			ID:         id,
			Name:       id, // For testing purpose, we set name to id
			Config:     &docker.Config{Image: "testimage"},
			HostConfig: hostConfig,
			State: docker.State{
				Running: true,
				Pid:     os.Getpid(),
			},
			NetworkSettings: &docker.NetworkSettings{IPAddress: "1.2.3.4"},
		}
	}
	return err
}

// StopContainer is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "stop" to the internal method call record.
func (f *FakeDockerClient) StopContainer(id string, timeout uint) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "stop")
	err := f.popError("stop")
	if err == nil {
		f.Stopped = append(f.Stopped, id)
		var newList []docker.APIContainers
		for _, container := range f.ContainerList {
			if container.ID == id {
				f.ExitedContainerList = append(f.ExitedContainerList, container)
				continue
			}
			newList = append(newList, container)
		}
		f.ContainerList = newList
	}
	return err
}

func (f *FakeDockerClient) RemoveContainer(opts docker.RemoveContainerOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "remove")
	err := f.popError("remove")
	if err == nil {
		f.Removed = append(f.Removed, opts.ID)
	}
	return err
}

// Logs is a test-spy implementation of DockerInterface.Logs.
// It adds an entry "logs" to the internal method call record.
func (f *FakeDockerClient) Logs(opts docker.LogsOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "logs")
	return f.popError("logs")
}

// PullImage is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "pull" to the internal method call record.
func (f *FakeDockerClient) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "pull")
	err := f.popError("pull")
	if err == nil {
		registry := opts.Registry
		if len(registry) != 0 {
			registry = registry + "/"
		}
		authJson, _ := json.Marshal(auth)
		f.pulled = append(f.pulled, fmt.Sprintf("%s%s:%s using %s", registry, opts.Repository, opts.Tag, string(authJson)))
	}
	return err
}

func (f *FakeDockerClient) Version() (*docker.Env, error) {
	return &f.VersionInfo, nil
}

func (f *FakeDockerClient) Info() (*docker.Env, error) {
	return &f.Information, nil
}

func (f *FakeDockerClient) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	f.Lock()
	defer f.Unlock()
	f.execCmd = opts.Cmd
	f.called = append(f.called, "create_exec")
	return &docker.Exec{"12345678"}, nil
}

func (f *FakeDockerClient) StartExec(_ string, _ docker.StartExecOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, "start_exec")
	return nil
}

func (f *FakeDockerClient) InspectExec(id string) (*docker.ExecInspect, error) {
	return f.ExecInspect, f.popError("inspect_exec")
}

func (f *FakeDockerClient) ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error) {
	err := f.popError("list_images")
	return f.Images, err
}

func (f *FakeDockerClient) RemoveImage(image string) error {
	err := f.popError("remove_image")
	if err == nil {
		f.RemovedImages.Insert(image)
	}
	return err
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
func (f *FakeDockerPuller) Pull(image string, secrets []api.Secret) (err error) {
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
