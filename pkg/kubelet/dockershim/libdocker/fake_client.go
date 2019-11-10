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

package libdocker

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	dockerimagetypes "github.com/docker/docker/api/types/image"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/clock"
)

// CalledDetail is a struct that stores the information aobut a call that's been made to the fake client
type CalledDetail struct {
	name      string
	arguments []interface{}
}

// NewCalledDetail create a new call detail item.
func NewCalledDetail(name string, arguments []interface{}) CalledDetail {
	return CalledDetail{name: name, arguments: arguments}
}

// FakeDockerClient is a simple fake docker client, so that kubelet can be run for testing without requiring a real docker setup.
type FakeDockerClient struct {
	sync.Mutex
	Clock                clock.Clock
	RunningContainerList []dockertypes.Container
	ExitedContainerList  []dockertypes.Container
	ContainerMap         map[string]*dockertypes.ContainerJSON
	ImageInspects        map[string]*dockertypes.ImageInspect
	Images               []dockertypes.ImageSummary
	ImageIDsNeedingAuth  map[string]dockertypes.AuthConfig
	Errors               map[string]error
	called               []CalledDetail
	pulled               []string
	EnableTrace          bool
	RandGenerator        *rand.Rand

	// Created, Started, Stopped and Removed all contain container docker ID
	Created []string
	Started []string
	Stopped []string
	Removed []string
	// Images pulled by ref (name or ID).
	ImagesPulled []string

	VersionInfo       dockertypes.Version
	Information       dockertypes.Info
	ExecInspect       *dockertypes.ContainerExecInspect
	execCmd           []string
	EnableSleep       bool
	ImageHistoryMap   map[string][]dockerimagetypes.HistoryResponseItem
	ContainerStatsMap map[string]*dockertypes.StatsJSON
}

const (
	// Notice that if someday we also have minimum docker version requirement, this should also be updated.
	fakeDockerVersion = "1.13.1"

	fakeImageSize = 1024

	// Docker prepends '/' to the container name.
	dockerNamePrefix = "/"
)

// NewFakeDockerClient creates a new instance of the fake docker client
func NewFakeDockerClient() *FakeDockerClient {
	return &FakeDockerClient{
		// Docker's API version does not include the patch number.
		VersionInfo:  dockertypes.Version{Version: fakeDockerVersion, APIVersion: strings.TrimSuffix(MinimumDockerAPIVersion, ".0")},
		Errors:       make(map[string]error),
		ContainerMap: make(map[string]*dockertypes.ContainerJSON),
		Clock:        clock.RealClock{},
		// default this to true, so that we trace calls, image pulls and container lifecycle
		EnableTrace:         true,
		ExecInspect:         &dockertypes.ContainerExecInspect{},
		ImageInspects:       make(map[string]*dockertypes.ImageInspect),
		ImageIDsNeedingAuth: make(map[string]dockertypes.AuthConfig),
		RandGenerator:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// WithClock sets the clock that's being used by the fake
func (f *FakeDockerClient) WithClock(c clock.Clock) *FakeDockerClient {
	f.Lock()
	defer f.Unlock()
	f.Clock = c
	return f
}

// WithVersion is a fake implementation of WithVersion for a real docker client
func (f *FakeDockerClient) WithVersion(version, apiVersion string) *FakeDockerClient {
	f.Lock()
	defer f.Unlock()
	f.VersionInfo = dockertypes.Version{Version: version, APIVersion: apiVersion}
	return f
}

// WithTraceDisabled disables tracing for the fake
func (f *FakeDockerClient) WithTraceDisabled() *FakeDockerClient {
	f.Lock()
	defer f.Unlock()
	f.EnableTrace = false
	return f
}

// WithRandSource sets the random source that will be used by the fake
func (f *FakeDockerClient) WithRandSource(source rand.Source) *FakeDockerClient {
	f.Lock()
	defer f.Unlock()
	f.RandGenerator = rand.New(source)
	return f
}

func (f *FakeDockerClient) appendCalled(callDetail CalledDetail) {
	if f.EnableTrace {
		f.called = append(f.called, callDetail)
	}
}

func (f *FakeDockerClient) appendPulled(pull string) {
	if f.EnableTrace {
		f.pulled = append(f.pulled, pull)
	}
}

func (f *FakeDockerClient) appendContainerTrace(traceCategory string, containerName string) {
	if !f.EnableTrace {
		return
	}
	switch traceCategory {
	case "Created":
		f.Created = append(f.Created, containerName)
	case "Started":
		f.Started = append(f.Started, containerName)
	case "Stopped":
		f.Stopped = append(f.Stopped, containerName)
	case "Removed":
		f.Removed = append(f.Removed, containerName)
	}
}

// InjectError injects an error into the fake client to simulate an error in a real client
func (f *FakeDockerClient) InjectError(fn string, err error) {
	f.Lock()
	defer f.Unlock()
	f.Errors[fn] = err
}

// InjectErrors injects multiple errors into the fake client to simulate errors in a real client
func (f *FakeDockerClient) InjectErrors(errs map[string]error) {
	f.Lock()
	defer f.Unlock()
	for fn, err := range errs {
		f.Errors[fn] = err
	}
}

// ClearErrors clears any errors that have been injected
func (f *FakeDockerClient) ClearErrors() {
	f.Lock()
	defer f.Unlock()
	f.Errors = map[string]error{}
}

// ClearCalls clears the calls that have been made to the fake
func (f *FakeDockerClient) ClearCalls() {
	f.Lock()
	defer f.Unlock()
	f.called = []CalledDetail{}
	f.pulled = []string{}
	f.Created = []string{}
	f.Started = []string{}
	f.Stopped = []string{}
	f.Removed = []string{}
}

func (f *FakeDockerClient) getCalledNames() []string {
	names := []string{}
	for _, detail := range f.called {
		names = append(names, detail.name)
	}
	return names
}

// FakeContainer is a fake data type that represents a container
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
			ID:    f.ID,
			Name:  f.Name,
			Image: f.Config.Image,
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

// SetFakeContainers adds fake containers to the fake client
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
		if c.Config != nil {
			container.Labels = c.Config.Labels
		}
		if c.Running {
			f.RunningContainerList = append(f.RunningContainerList, container)
		} else {
			f.ExitedContainerList = append(f.ExitedContainerList, container)
		}
	}
}

// SetFakeRunningContainers adds fake containers to the client in the running state
func (f *FakeDockerClient) SetFakeRunningContainers(containers []*FakeContainer) {
	for _, c := range containers {
		c.Running = true
	}
	f.SetFakeContainers(containers)
}

// AssertCalls checks that a list of calls have been made to the fake client but does not check the arguments
func (f *FakeDockerClient) AssertCalls(calls []string) (err error) {
	f.Lock()
	defer f.Unlock()

	if !reflect.DeepEqual(calls, f.getCalledNames()) {
		err = fmt.Errorf("expected %#v, got %#v", calls, f.getCalledNames())
	}

	return
}

// AssertCallDetails checks that a list of calls have been made and makes sure that the arguments match as well
func (f *FakeDockerClient) AssertCallDetails(calls ...CalledDetail) (err error) {
	f.Lock()
	defer f.Unlock()

	if !reflect.DeepEqual(calls, f.called) {
		err = fmt.Errorf("expected %#v, got %#v", calls, f.called)
	}

	return
}

// idsToNames converts container ids into names. The caller must hold the lock.
func (f *FakeDockerClient) idsToNames(ids []string) ([]string, error) {
	names := []string{}
	for _, id := range ids {
		names = append(names, strings.TrimPrefix(f.ContainerMap[id].Name, dockerNamePrefix))
	}
	return names, nil
}

// AssertCreatedByNameWithOrder checks that containers were created in a certain order by their names
func (f *FakeDockerClient) AssertCreatedByNameWithOrder(created []string) error {
	f.Lock()
	defer f.Unlock()
	actualCreated, err := f.idsToNames(f.Created)
	if err != nil {
		return err
	}
	if !reflect.DeepEqual(created, actualCreated) {
		return fmt.Errorf("expected %#v, got %#v", created, actualCreated)
	}
	return nil
}

// AssertCreatedByName checks that a set of containers were created by their names
func (f *FakeDockerClient) AssertCreatedByName(created []string) error {
	f.Lock()
	defer f.Unlock()

	actualCreated, err := f.idsToNames(f.Created)
	if err != nil {
		return err
	}
	return sortedStringSlicesEqual(created, actualCreated)
}

// AssertStoppedByName checks that a list of container were stopped by their names
func (f *FakeDockerClient) AssertStoppedByName(stopped []string) error {
	f.Lock()
	defer f.Unlock()
	actualStopped, err := f.idsToNames(f.Stopped)
	if err != nil {
		return err
	}
	return sortedStringSlicesEqual(stopped, actualStopped)
}

// AssertStopped checks that a list of container were stopped by their ids
func (f *FakeDockerClient) AssertStopped(stopped []string) error {
	f.Lock()
	defer f.Unlock()
	// Copy stopped to avoid modifying it.
	actualStopped := append([]string{}, f.Stopped...)
	return sortedStringSlicesEqual(stopped, actualStopped)
}

// AssertImagesPulled checks that a list a set of images were pulled
func (f *FakeDockerClient) AssertImagesPulled(pulled []string) error {
	f.Lock()
	defer f.Unlock()
	// Copy pulled to avoid modifying it.
	actualPulled := append([]string{}, f.ImagesPulled...)
	return sortedStringSlicesEqual(pulled, actualPulled)
}

// AssertImagesPulledMsgs checks that the messages from pulling image match whats expected
func (f *FakeDockerClient) AssertImagesPulledMsgs(expected []string) error {
	f.Lock()
	defer f.Unlock()
	// Copy pulled to avoid modifying it.
	actual := append([]string{}, f.pulled...)
	return sortedStringSlicesEqual(expected, actual)
}

func sortedStringSlicesEqual(expected, actual []string) error {
	sort.StringSlice(expected).Sort()
	sort.StringSlice(actual).Sort()
	if !reflect.DeepEqual(expected, actual) {
		return fmt.Errorf("expected %#v, got %#v", expected, actual)
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
	}
	return nil
}

// ListContainers is a test-spy implementation of Interface.ListContainers.
// It adds an entry "list" to the internal method call record.
func (f *FakeDockerClient) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "list"})
	err := f.popError("list")
	containerList := append([]dockertypes.Container{}, f.RunningContainerList...)
	if options.All {
		// Although the container is not sorted, but the container with the same name should be in order,
		// that is enough for us now.
		// TODO(random-liu): Is a fully sorted array needed?
		containerList = append(containerList, f.ExitedContainerList...)
	}
	// Filters containers with id, only support 1 id.
	idFilters := options.Filters.Get("id")
	if len(idFilters) != 0 {
		var filtered []dockertypes.Container
		for _, container := range containerList {
			for _, idFilter := range idFilters {
				if container.ID == idFilter {
					filtered = append(filtered, container)
					break
				}
			}
		}
		containerList = filtered
	}
	// Filters containers with status, only support 1 status.
	statusFilters := options.Filters.Get("status")
	if len(statusFilters) == 1 {
		var filtered []dockertypes.Container
		for _, container := range containerList {
			for _, statusFilter := range statusFilters {
				if toDockerContainerStatus(container.Status) == statusFilter {
					filtered = append(filtered, container)
					break
				}
			}
		}
		containerList = filtered
	}
	// Filters containers with label filter.
	labelFilters := options.Filters.Get("label")
	if len(labelFilters) != 0 {
		var filtered []dockertypes.Container
		for _, container := range containerList {
			match := true
			for _, labelFilter := range labelFilters {
				kv := strings.Split(labelFilter, "=")
				if len(kv) != 2 {
					return nil, fmt.Errorf("invalid label filter %q", labelFilter)
				}
				if container.Labels[kv[0]] != kv[1] {
					match = false
					break
				}
			}
			if match {
				filtered = append(filtered, container)
			}
		}
		containerList = filtered
	}
	return containerList, err
}

func toDockerContainerStatus(state string) string {
	switch {
	case strings.HasPrefix(state, StatusCreatedPrefix):
		return "created"
	case strings.HasPrefix(state, StatusRunningPrefix):
		return "running"
	case strings.HasPrefix(state, StatusExitedPrefix):
		return "exited"
	default:
		return "unknown"
	}
}

// InspectContainer is a test-spy implementation of Interface.InspectContainer.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "inspect_container"})
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

// InspectContainerWithSize is a test-spy implementation of Interface.InspectContainerWithSize.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectContainerWithSize(id string) (*dockertypes.ContainerJSON, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "inspect_container_withsize"})
	err := f.popError("inspect_container_withsize")
	if container, ok := f.ContainerMap[id]; ok {
		return container, err
	}
	if err != nil {
		// Use the custom error if it exists.
		return nil, err
	}
	return nil, fmt.Errorf("container %q not found", id)
}

// InspectImageByRef is a test-spy implementation of Interface.InspectImageByRef.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImageByRef(name string) (*dockertypes.ImageInspect, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "inspect_image"})
	if err := f.popError("inspect_image"); err != nil {
		return nil, err
	}
	if result, ok := f.ImageInspects[name]; ok {
		return result, nil
	}
	return nil, ImageNotFoundError{name}
}

// InspectImageByID is a test-spy implementation of Interface.InspectImageByID.
// It adds an entry "inspect" to the internal method call record.
func (f *FakeDockerClient) InspectImageByID(name string) (*dockertypes.ImageInspect, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "inspect_image"})
	if err := f.popError("inspect_image"); err != nil {
		return nil, err
	}
	if result, ok := f.ImageInspects[name]; ok {
		return result, nil
	}
	return nil, ImageNotFoundError{name}
}

// Sleeps random amount of time with the normal distribution with given mean and stddev
// (in milliseconds), we never sleep less than cutOffMillis
func (f *FakeDockerClient) normalSleep(mean, stdDev, cutOffMillis int) {
	if !f.EnableSleep {
		return
	}
	cutoff := (time.Duration)(cutOffMillis) * time.Millisecond
	delay := (time.Duration)(f.RandGenerator.NormFloat64()*float64(stdDev)+float64(mean)) * time.Millisecond
	if delay < cutoff {
		delay = cutoff
	}
	time.Sleep(delay)
}

// GetFakeContainerID generates a fake container id from container name with a hash.
func GetFakeContainerID(name string) string {
	hash := fnv.New64a()
	hash.Write([]byte(name))
	return strconv.FormatUint(hash.Sum64(), 16)
}

// CreateContainer is a test-spy implementation of Interface.CreateContainer.
// It adds an entry "create" to the internal method call record.
func (f *FakeDockerClient) CreateContainer(c dockertypes.ContainerCreateConfig) (*dockercontainer.ContainerCreateCreatedBody, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "create"})
	if err := f.popError("create"); err != nil {
		return nil, err
	}
	// This is not a very good fake. We'll just add this container's name to the list.
	name := dockerNamePrefix + c.Name
	id := GetFakeContainerID(name)
	f.appendContainerTrace("Created", id)
	timestamp := f.Clock.Now()
	// The newest container should be in front, because we assume so in GetPodStatus()
	f.RunningContainerList = append([]dockertypes.Container{
		{ID: id, Names: []string{name}, Image: c.Config.Image, Created: timestamp.Unix(), State: StatusCreatedPrefix, Labels: c.Config.Labels},
	}, f.RunningContainerList...)
	f.ContainerMap[id] = convertFakeContainer(&FakeContainer{
		ID: id, Name: name, Config: c.Config, HostConfig: c.HostConfig, CreatedAt: timestamp})

	f.normalSleep(100, 25, 25)

	return &dockercontainer.ContainerCreateCreatedBody{ID: id}, nil
}

// StartContainer is a test-spy implementation of Interface.StartContainer.
// It adds an entry "start" to the internal method call record.
func (f *FakeDockerClient) StartContainer(id string) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "start"})
	if err := f.popError("start"); err != nil {
		return err
	}
	f.appendContainerTrace("Started", id)
	container, ok := f.ContainerMap[id]
	if container.HostConfig.NetworkMode.IsContainer() {
		hostContainerID := container.HostConfig.NetworkMode.ConnectedContainer()
		found := false
		for _, container := range f.RunningContainerList {
			if container.ID == hostContainerID {
				found = true
			}
		}
		if !found {
			return fmt.Errorf("failed to start container \"%s\": Error response from daemon: cannot join network of a non running container: %s", id, hostContainerID)
		}
	}
	timestamp := f.Clock.Now()
	if !ok {
		container = convertFakeContainer(&FakeContainer{ID: id, Name: id, CreatedAt: timestamp})
	}
	container.State.Running = true
	container.State.Pid = os.Getpid()
	container.State.StartedAt = dockerTimestampToString(timestamp)
	r := f.RandGenerator.Uint32()
	container.NetworkSettings.IPAddress = fmt.Sprintf("10.%d.%d.%d", byte(r>>16), byte(r>>8), byte(r))
	f.ContainerMap[id] = container
	f.updateContainerStatus(id, StatusRunningPrefix)
	f.normalSleep(200, 50, 50)
	return nil
}

// StopContainer is a test-spy implementation of Interface.StopContainer.
// It adds an entry "stop" to the internal method call record.
func (f *FakeDockerClient) StopContainer(id string, timeout time.Duration) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "stop"})
	if err := f.popError("stop"); err != nil {
		return err
	}
	f.appendContainerTrace("Stopped", id)
	// Container status should be Updated before container moved to ExitedContainerList
	f.updateContainerStatus(id, StatusExitedPrefix)
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

// RemoveContainer removes a container from the fake client
func (f *FakeDockerClient) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "remove"})
	err := f.popError("remove")
	if err != nil {
		return err
	}
	for i := range f.ExitedContainerList {
		if f.ExitedContainerList[i].ID == id {
			delete(f.ContainerMap, id)
			f.ExitedContainerList = append(f.ExitedContainerList[:i], f.ExitedContainerList[i+1:]...)
			f.appendContainerTrace("Removed", id)
			return nil
		}

	}
	for i := range f.RunningContainerList {
		// allow removal of running containers which are not running
		if f.RunningContainerList[i].ID == id && !f.ContainerMap[id].State.Running {
			delete(f.ContainerMap, id)
			f.RunningContainerList = append(f.RunningContainerList[:i], f.RunningContainerList[i+1:]...)
			f.appendContainerTrace("Removed", id)
			return nil
		}
	}
	// To be a good fake, report error if container is not stopped.
	return fmt.Errorf("container not stopped")
}

// UpdateContainerResources is the fake implementation of the same call in a real client
func (f *FakeDockerClient) UpdateContainerResources(id string, updateConfig dockercontainer.UpdateConfig) error {
	return nil
}

// Logs is a test-spy implementation of Interface.Logs.
// It adds an entry "logs" to the internal method call record.
func (f *FakeDockerClient) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "logs"})
	return f.popError("logs")
}

func (f *FakeDockerClient) isAuthorizedForImage(image string, auth dockertypes.AuthConfig) bool {
	reqd, exists := f.ImageIDsNeedingAuth[image]
	if !exists {
		return true // no auth needed
	}

	return auth.Username == reqd.Username && auth.Password == reqd.Password
}

// PullImage is a test-spy implementation of Interface.PullImage.
// It adds an entry "pull" to the internal method call record.
func (f *FakeDockerClient) PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "pull"})
	err := f.popError("pull")
	if err == nil {
		if !f.isAuthorizedForImage(image, auth) {
			return ImageNotFoundError{ID: image}
		}

		authJSON, _ := json.Marshal(auth)
		inspect := createImageInspectFromRef(image)
		f.ImageInspects[image] = inspect
		f.appendPulled(fmt.Sprintf("%s using %s", image, string(authJSON)))
		f.Images = append(f.Images, *createImageFromImageInspect(*inspect))
		f.ImagesPulled = append(f.ImagesPulled, image)
	}
	return err
}

// Version gets the version of the fake client
func (f *FakeDockerClient) Version() (*dockertypes.Version, error) {
	f.Lock()
	defer f.Unlock()
	v := f.VersionInfo
	return &v, f.popError("version")
}

// Info gets the client information stored in the the fake client
func (f *FakeDockerClient) Info() (*dockertypes.Info, error) {
	return &f.Information, nil
}

// CreateExec is a fake implentation of the same call in the real client, the call is stored in the fake with the name "create_exec"
func (f *FakeDockerClient) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.IDResponse, error) {
	f.Lock()
	defer f.Unlock()
	f.execCmd = opts.Cmd
	f.appendCalled(CalledDetail{name: "create_exec"})
	return &dockertypes.IDResponse{ID: "12345678"}, nil
}

// StartExec is a fake implementation of the same call in the real client, the call is stored in the fake with the name "start_exec"
func (f *FakeDockerClient) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "start_exec"})
	return nil
}

// AttachToContainer is a fake implemenattion of the call in the real client, the call is stored in the fake with the name "attach"
func (f *FakeDockerClient) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "attach"})
	return nil
}

// InspectExec is a fake implemenattion of the same call in the real client
func (f *FakeDockerClient) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	return f.ExecInspect, f.popError("inspect_exec")
}

// ListImages is a fake implemenattion of the call in the real client, the call is stored in the fake with the name "list_images"
func (f *FakeDockerClient) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.ImageSummary, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "list_images"})
	err := f.popError("list_images")
	return f.Images, err
}

// RemoveImage is a fake implemenattion of the call in the real client, the call is stored in the fake with the name "remove_image"
func (f *FakeDockerClient) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDeleteResponseItem, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "remove_image", arguments: []interface{}{image, opts}})
	err := f.popError("remove_image")
	if err == nil {
		for i := range f.Images {
			if f.Images[i].ID == image {
				f.Images = append(f.Images[:i], f.Images[i+1:]...)
				break
			}
		}
	}
	return []dockertypes.ImageDeleteResponseItem{{Deleted: image}}, err
}

// InjectImages injects a set images into the fake client
func (f *FakeDockerClient) InjectImages(images []dockertypes.ImageSummary) {
	f.Lock()
	defer f.Unlock()
	f.Images = append(f.Images, images...)
	for _, i := range images {
		f.ImageInspects[i.ID] = createImageInspectFromImage(i)
	}
}

// MakeImagesPrivate sets an given auth config for a list of docker images
func (f *FakeDockerClient) MakeImagesPrivate(images []dockertypes.ImageSummary, auth dockertypes.AuthConfig) {
	f.Lock()
	defer f.Unlock()
	for _, i := range images {
		f.ImageIDsNeedingAuth[i.ID] = auth
	}
}

// ResetImages clears the stored images in the fake client
func (f *FakeDockerClient) ResetImages() {
	f.Lock()
	defer f.Unlock()
	f.Images = []dockertypes.ImageSummary{}
	f.ImageInspects = make(map[string]*dockertypes.ImageInspect)
	f.ImageIDsNeedingAuth = make(map[string]dockertypes.AuthConfig)
}

// InjectImageInspects injects images into the client created from a image inspect
func (f *FakeDockerClient) InjectImageInspects(inspects []dockertypes.ImageInspect) {
	f.Lock()
	defer f.Unlock()
	for _, i := range inspects {
		f.Images = append(f.Images, *createImageFromImageInspect(i))
		f.ImageInspects[i.ID] = &i
	}
}

func (f *FakeDockerClient) updateContainerStatus(id, status string) {
	for i := range f.RunningContainerList {
		if f.RunningContainerList[i].ID == id {
			f.RunningContainerList[i].Status = status
		}
	}
}

// ResizeExecTTY is a fake implemenattion of the call in the real client, the call is stored in the fake with the name "resize_exec"
func (f *FakeDockerClient) ResizeExecTTY(id string, height, width uint) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "resize_exec"})
	return nil
}

// ResizeContainerTTY is a fake implemenattion of the call in the real client, the call is stored in the fake with the name "resize_container"
func (f *FakeDockerClient) ResizeContainerTTY(id string, height, width uint) error {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "resize_container"})
	return nil
}

func createImageInspectFromRef(ref string) *dockertypes.ImageInspect {
	return &dockertypes.ImageInspect{
		ID:       ref,
		RepoTags: []string{ref},
		// Image size is required to be non-zero for CRI integration.
		VirtualSize: fakeImageSize,
		Size:        fakeImageSize,
		Config:      &dockercontainer.Config{},
	}
}

func createImageInspectFromImage(image dockertypes.ImageSummary) *dockertypes.ImageInspect {
	return &dockertypes.ImageInspect{
		ID:       image.ID,
		RepoTags: image.RepoTags,
		// Image size is required to be non-zero for CRI integration.
		VirtualSize: fakeImageSize,
		Size:        fakeImageSize,
		Config:      &dockercontainer.Config{},
	}
}

func createImageFromImageInspect(inspect dockertypes.ImageInspect) *dockertypes.ImageSummary {
	return &dockertypes.ImageSummary{
		ID:       inspect.ID,
		RepoTags: inspect.RepoTags,
		// Image size is required to be non-zero for CRI integration.
		VirtualSize: fakeImageSize,
		Size:        fakeImageSize,
	}
}

// dockerTimestampToString converts the timestamp to string
func dockerTimestampToString(t time.Time) string {
	return t.Format(time.RFC3339Nano)
}

// ImageHistory gets the history from an image stored in the fake client, the call is logged in the fake client with the name "image_history"
func (f *FakeDockerClient) ImageHistory(id string) ([]dockerimagetypes.HistoryResponseItem, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "image_history"})
	history := f.ImageHistoryMap[id]
	return history, nil
}

// InjectImageHistory sets the history image map in the fake client
func (f *FakeDockerClient) InjectImageHistory(data map[string][]dockerimagetypes.HistoryResponseItem) {
	f.Lock()
	defer f.Unlock()
	f.ImageHistoryMap = data
}

// FakeDockerPuller is meant to be a simple wrapper around FakeDockerClient.
// Please do not add more functionalities to it.
type FakeDockerPuller struct {
	client Interface
}

// Pull pulls\ an image from the fake client that stored in the fake puller
func (f *FakeDockerPuller) Pull(image string, _ []v1.Secret) error {
	return f.client.PullImage(image, dockertypes.AuthConfig{}, dockertypes.ImagePullOptions{})
}

// GetImageRef gets an image that's stored in the fake client by it's ref
func (f *FakeDockerPuller) GetImageRef(image string) (string, error) {
	_, err := f.client.InspectImageByRef(image)
	if err != nil && IsImageNotFoundError(err) {
		return "", nil
	}
	return image, err
}

// InjectContainerStats sets the value for container stats that's stored inside the fake client
func (f *FakeDockerClient) InjectContainerStats(data map[string]*dockertypes.StatsJSON) {
	f.Lock()
	defer f.Unlock()
	f.ContainerStatsMap = data
}

// GetContainerStats returns the stored value for container stats and logs the call with the name "get_container_stats"
func (f *FakeDockerClient) GetContainerStats(id string) (*dockertypes.StatsJSON, error) {
	f.Lock()
	defer f.Unlock()
	f.appendCalled(CalledDetail{name: "get_container_stats"})
	stats, ok := f.ContainerStatsMap[id]
	if !ok {
		return nil, fmt.Errorf("container %q not found", id)
	}
	return stats, nil
}
