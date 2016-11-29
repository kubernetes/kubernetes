/*
Copyright 2014 The Kubernetes Authors.

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
	"math/rand"
	"os"
	"reflect"
	"sort"
	"sync"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	"k8s.io/kubernetes/pkg/util/clock"

	"k8s.io/kubernetes/pkg/api/v1"
)

type calledDetail struct {
	name      string
	arguments []interface{}
}

// NewCalledDetail create a new call detail item.
func NewCalledDetail(name string, arguments []interface{}) calledDetail {
	return calledDetail{name: name, arguments: arguments}
}

// FakeDockerClient is a simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	sync.Mutex
	Clock                clock.Clock
	RunningContainerList []dockertypes.Container
	ExitedContainerList  []dockertypes.Container
	ContainerMap         map[string]*dockertypes.ContainerJSON
	Image                *dockertypes.ImageInspect
	Images               []dockertypes.Image
	Errors               map[string]error
	called               []calledDetail
	pulled               []string

	// Created, Stopped and Removed all container docker ID
	Created         []string
	Started         []string
	Stopped         []string
	Removed         []string
	VersionInfo     dockertypes.Version
	Information     dockertypes.Info
	ExecInspect     *dockertypes.ContainerExecInspect
	execCmd         []string
	EnableSleep     bool
	ImageHistoryMap map[string][]dockertypes.ImageHistory
}

// We don't check docker version now, just set the docker version of fake docker client to 1.8.1.
// Notice that if someday we also have minimum docker version requirement, this should also be updated.
const fakeDockerVersion = "1.8.1"

func NewFakeDockerClient() *FakeDockerClient {
	return NewFakeDockerClientWithVersion(fakeDockerVersion, minimumDockerAPIVersion)
}

func NewFakeDockerClientWithClock(c clock.Clock) *FakeDockerClient {
	return newClientWithVersionAndClock(fakeDockerVersion, minimumDockerAPIVersion, c)
}

func NewFakeDockerClientWithVersion(version, apiVersion string) *FakeDockerClient {
	return newClientWithVersionAndClock(version, apiVersion, clock.RealClock{})
}

func newClientWithVersionAndClock(version, apiVersion string, c clock.Clock) *FakeDockerClient {
	return &FakeDockerClient{
		VersionInfo:  dockertypes.Version{Version: version, APIVersion: apiVersion},
		Errors:       make(map[string]error),
		ContainerMap: make(map[string]*dockertypes.ContainerJSON),
		Clock:        c,
		// default this to an empty result, so that we never have a nil non-error response from InspectImage
		Image: &dockertypes.ImageInspect{},
	}
}

func (f *FakeDockerClient) InjectError(fn string, err error) {
	f.Lock()
	defer f.Unlock()
	f.Errors[fn] = err
}

func (f *FakeDockerClient) InjectErrors(errs map[string]error) {
	f.Lock()
	defer f.Unlock()
	for fn, err := range errs {
		f.Errors[fn] = err
	}
}

func (f *FakeDockerClient) ClearErrors() {
	f.Lock()
	defer f.Unlock()
	f.Errors = map[string]error{}
}

func (f *FakeDockerClient) ClearCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = []calledDetail{}
	f.Stopped = []string{}
	f.pulled = []string{}
	f.Created = []string{}
	f.Removed = []string{}
}

func (f *FakeDockerClient) getCalledNames() []string {
	names := []string{}
	for _, detail := range f.called {
		names = append(names, detail.name)
	}
	return names
}

// Because the new data type returned by engine-api is too complex to manually initialize, we need a
// fake container which is easier to initialize.
type FakeContainer struct {
	ID         string
	Name       string
	Running    bool
	ExitCode   int
	Pid        int
	CreatedAt  time.Time
	StartedAt  time.Time
	FinishedAt time.Time
	Config     *dockercontainer.Config
	HostConfig *dockercontainer.HostConfig
}

// convertFakeContainer converts the fake container to real container
func convertFakeContainer(f *FakeContainer) *dockertypes.ContainerJSON {
	if f.Config == nil {
		f.Config = &dockercontainer.Config{}
	}
	if f.HostConfig == nil {
		f.HostConfig = &dockercontainer.HostConfig{}
	}
	return &dockertypes.ContainerJSON{
		ContainerJSONBase: &dockertypes.ContainerJSONBase{
			ID:   f.ID,
			Name: f.Name,
			State: &dockertypes.ContainerState{
				Running:    f.Running,
				ExitCode:   f.ExitCode,
				Pid:        f.Pid,
				StartedAt:  dockerTimestampToString(f.StartedAt),
				FinishedAt: dockerTimestampToString(f.FinishedAt),
			},
			Created:    dockerTimestampToString(f.CreatedAt),
			HostConfig: f.HostConfig,
		},
		Config:          f.Config,
		NetworkSettings: &dockertypes.NetworkSettings{},
	}
}

func (f *FakeDockerClient) SetFakeContainers(containers []*FakeContainer) {
	f.Lock()
	defer f.Unlock()
	// Reset the lists and the map.
	f.ContainerMap = map[string]*dockertypes.ContainerJSON{}
	f.RunningContainerList = []dockertypes.Container{}
	f.ExitedContainerList = []dockertypes.Container{}

	for i := range containers {
		c := containers[i]
		f.ContainerMap[c.ID] = convertFakeContainer(c)
		container := dockertypes.Container{
			Names: []string{c.Name},
			ID:    c.ID,
		}
		if c.Running {
			f.RunningContainerList = append(f.RunningContainerList, container)
		} else {
			f.ExitedContainerList = append(f.ExitedContainerList, container)
		}
	}
}

func (f *FakeDockerClient) SetFakeRunningContainers(containers []*FakeContainer) {
	for _, c := range containers {
		c.Running = true
	}
	f.SetFakeContainers(containers)
}

func (f *FakeDockerClient) AssertCalls(calls []string) (err error) {
	f.Lock()
	defer f.Unlock()

	if !reflect.DeepEqual(calls, f.getCalledNames()) {
		err = fmt.Errorf("expected %#v, got %#v", calls, f.getCalledNames())
	}

	return
}

func (f *FakeDockerClient) AssertCallDetails(calls ...calledDetail) (err error) {
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

func (f *FakeDockerClient) AssertStarted(started []string) error {
	f.Lock()
	defer f.Unlock()
	sort.StringSlice(started).Sort()
	sort.StringSlice(f.Started).Sort()
	if !reflect.DeepEqual(started, f.Started) {
		return fmt.Errorf("expected %#v, got %#v", started, f.Started)
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
func (f *FakeDockerClient) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "list"})
	err := f.popError("list")
	containerList := append([]dockertypes.Container{}, f.RunningContainerList...)
	if options.All {
		// Although the container is not sorted, but the container with the same name should be in order,
		// that is enough for us now.
		// TODO(random-liu): Is a fully sorted array needed?
		containerList = append(containerList, f.ExitedContainerList...)
	}
	return containerList, err
}

// InspectContainer is a test-spy implementation of DockerInterface.InspectContainer.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "inspect_container"})
	err := f.popError("inspect_container")
	if container, ok := f.ContainerMap[id]; ok {
		return container, err
	}
	if err != nil {
		// Use the custom error if it exists.
		return nil, err
	}
	return nil, fmt.Errorf("container %q not found", id)
}

// InspectImageByRef is a test-spy implementation of DockerInterface.InspectImageByRef.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImageByRef(name string) (*dockertypes.ImageInspect, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "inspect_image"})
	err := f.popError("inspect_image")
	return f.Image, err
}

// InspectImageByID is a test-spy implementation of DockerInterface.InspectImageByID.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImageByID(name string) (*dockertypes.ImageInspect, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "inspect_image"})
	err := f.popError("inspect_image")
	return f.Image, err
}

// Sleeps random amount of time with the normal distribution with given mean and stddev
// (in milliseconds), we never sleep less than cutOffMillis
func (f *FakeDockerClient) normalSleep(mean, stdDev, cutOffMillis int) {
	if !f.EnableSleep {
		return
	}
	cutoff := (time.Duration)(cutOffMillis) * time.Millisecond
	delay := (time.Duration)(rand.NormFloat64()*float64(stdDev)+float64(mean)) * time.Millisecond
	if delay < cutoff {
		delay = cutoff
	}
	time.Sleep(delay)
}

// CreateContainer is a test-spy implementation of DockerInterface.CreateContainer.
// It adds an entry "create" to the internal method call record.
func (f *FakeDockerClient) CreateContainer(c dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "create"})
	if err := f.popError("create"); err != nil {
		return nil, err
	}
	// This is not a very good fake. We'll just add this container's name to the list.
	// Docker likes to add a '/', so copy that behavior.
	name := "/" + c.Name
	id := name
	f.Created = append(f.Created, name)
	// The newest container should be in front, because we assume so in GetPodStatus()
	f.RunningContainerList = append([]dockertypes.Container{
		{ID: name, Names: []string{name}, Image: c.Config.Image, Labels: c.Config.Labels},
	}, f.RunningContainerList...)
	f.ContainerMap[name] = convertFakeContainer(&FakeContainer{
		ID: id, Name: name, Config: c.Config, HostConfig: c.HostConfig, CreatedAt: f.Clock.Now()})
	f.normalSleep(100, 25, 25)
	return &dockertypes.ContainerCreateResponse{ID: id}, nil
}

// StartContainer is a test-spy implementation of DockerInterface.StartContainer.
// It adds an entry "start" to the internal method call record.
func (f *FakeDockerClient) StartContainer(id string) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "start"})
	if err := f.popError("start"); err != nil {
		return err
	}
	f.Started = append(f.Started, id)
	container, ok := f.ContainerMap[id]
	if !ok {
		container = convertFakeContainer(&FakeContainer{ID: id, Name: id})
	}
	container.State.Running = true
	container.State.Pid = os.Getpid()
	container.State.StartedAt = dockerTimestampToString(f.Clock.Now())
	container.NetworkSettings.IPAddress = "2.3.4.5"
	f.ContainerMap[id] = container
	f.updateContainerStatus(id, statusRunningPrefix)
	f.normalSleep(200, 50, 50)
	return nil
}

// StopContainer is a test-spy implementation of DockerInterface.StopContainer.
// It adds an entry "stop" to the internal method call record.
func (f *FakeDockerClient) StopContainer(id string, timeout int) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "stop"})
	if err := f.popError("stop"); err != nil {
		return err
	}
	f.Stopped = append(f.Stopped, id)
	// Container status should be Updated before container moved to ExitedContainerList
	f.updateContainerStatus(id, statusExitedPrefix)
	var newList []dockertypes.Container
	for _, container := range f.RunningContainerList {
		if container.ID == id {
			// The newest exited container should be in front. Because we assume so in GetPodStatus()
			f.ExitedContainerList = append([]dockertypes.Container{container}, f.ExitedContainerList...)
			continue
		}
		newList = append(newList, container)
	}
	f.RunningContainerList = newList
	container, ok := f.ContainerMap[id]
	if !ok {
		container = convertFakeContainer(&FakeContainer{
			ID:         id,
			Name:       id,
			Running:    false,
			StartedAt:  time.Now().Add(-time.Second),
			FinishedAt: time.Now(),
		})
	} else {
		container.State.FinishedAt = dockerTimestampToString(f.Clock.Now())
		container.State.Running = false
	}
	f.ContainerMap[id] = container
	f.normalSleep(200, 50, 50)
	return nil
}

func (f *FakeDockerClient) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "remove"})
	err := f.popError("remove")
	if err != nil {
		return err
	}
	for i := range f.ExitedContainerList {
		if f.ExitedContainerList[i].ID == id {
			delete(f.ContainerMap, id)
			f.ExitedContainerList = append(f.ExitedContainerList[:i], f.ExitedContainerList[i+1:]...)
			f.Removed = append(f.Removed, id)
			return nil
		}

	}
	// To be a good fake, report error if container is not stopped.
	return fmt.Errorf("container not stopped")
}

// Logs is a test-spy implementation of DockerInterface.Logs.
// It adds an entry "logs" to the internal method call record.
func (f *FakeDockerClient) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "logs"})
	return f.popError("logs")
}

// PullImage is a test-spy implementation of DockerInterface.PullImage.
// It adds an entry "pull" to the internal method call record.
func (f *FakeDockerClient) PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "pull"})
	err := f.popError("pull")
	if err == nil {
		authJson, _ := json.Marshal(auth)
		f.pulled = append(f.pulled, fmt.Sprintf("%s using %s", image, string(authJson)))
	}
	return err
}

func (f *FakeDockerClient) Version() (*dockertypes.Version, error) {
	f.Lock()
	defer f.Unlock()
	return &f.VersionInfo, f.popError("version")
}

func (f *FakeDockerClient) Info() (*dockertypes.Info, error) {
	return &f.Information, nil
}

func (f *FakeDockerClient) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error) {
	f.Lock()
	defer f.Unlock()
	f.execCmd = opts.Cmd
	f.called = append(f.called, calledDetail{name: "create_exec"})
	return &dockertypes.ContainerExecCreateResponse{ID: "12345678"}, nil
}

func (f *FakeDockerClient) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "start_exec"})
	return nil
}

func (f *FakeDockerClient) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "attach"})
	return nil
}

func (f *FakeDockerClient) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	return f.ExecInspect, f.popError("inspect_exec")
}

func (f *FakeDockerClient) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error) {
	f.called = append(f.called, calledDetail{name: "list_images"})
	err := f.popError("list_images")
	return f.Images, err
}

func (f *FakeDockerClient) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error) {
	f.called = append(f.called, calledDetail{name: "remove_image", arguments: []interface{}{image, opts}})
	err := f.popError("remove_image")
	if err == nil {
		for i := range f.Images {
			if f.Images[i].ID == image {
				f.Images = append(f.Images[:i], f.Images[i+1:]...)
				break
			}
		}
	}
	return []dockertypes.ImageDelete{{Deleted: image}}, err
}

func (f *FakeDockerClient) InjectImages(images []dockertypes.Image) {
	f.Lock()
	defer f.Unlock()
	f.Images = append(f.Images, images...)
}

func (f *FakeDockerClient) updateContainerStatus(id, status string) {
	for i := range f.RunningContainerList {
		if f.RunningContainerList[i].ID == id {
			f.RunningContainerList[i].Status = status
		}
	}
}

func (f *FakeDockerClient) ResizeExecTTY(id string, height, width int) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "resize_exec"})
	return nil
}

func (f *FakeDockerClient) ResizeContainerTTY(id string, height, width int) error {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "resize_container"})
	return nil
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
func (f *FakeDockerPuller) Pull(image string, secrets []v1.Secret) (err error) {
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
func (f *FakeDockerClient) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	f.Lock()
	defer f.Unlock()
	f.called = append(f.called, calledDetail{name: "image_history"})
	history := f.ImageHistoryMap[id]
	return history, nil
}

func (f *FakeDockerClient) InjectImageHistory(data map[string][]dockertypes.ImageHistory) {
	f.Lock()
	defer f.Unlock()
	f.ImageHistoryMap = data
}

// dockerTimestampToString converts the timestamp to string
func dockerTimestampToString(t time.Time) string {
	return t.Format(time.RFC3339Nano)
}
