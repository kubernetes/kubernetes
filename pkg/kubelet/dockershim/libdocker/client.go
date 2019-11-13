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
	"time"

	dockertypes "github.com/docker/docker/api/types"
	dockercontainer "github.com/docker/docker/api/types/container"
	dockerimagetypes "github.com/docker/docker/api/types/image"
	dockerapi "github.com/docker/docker/client"
	"k8s.io/klog"
)

const (
	// MinimumDockerAPIVersion is the lower docker version acceptable
	// https://docs.docker.com/engine/reference/api/docker_remote_api/
	// Docker API version should be at least 1.26.0 (the API version for minimum docker version of 1.13.1)
	MinimumDockerAPIVersion = "1.26.0"

	// Constants for the prefix for a status of a container returned by ListContainers.

	// StatusRunningPrefix is the prefix that the status has when the a container is running
	StatusRunningPrefix = "Up"
	// StatusCreatedPrefix is the prefix that the status has when the a container is first created
	StatusCreatedPrefix = "Created"
	// StatusExitedPrefix is the prefix that the status has when the a container is exited
	StatusExitedPrefix = "Exited"

	// FakeDockerEndpoint is used to know when to return a fake docker client instead of a real one
	FakeDockerEndpoint = "fake://"
)

// Interface is an abstract interface for testability.  It abstracts the interface of docker client.
type Interface interface {
	ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error)
	InspectContainer(id string) (*dockertypes.ContainerJSON, error)
	InspectContainerWithSize(id string) (*dockertypes.ContainerJSON, error)
	CreateContainer(dockertypes.ContainerCreateConfig) (*dockercontainer.ContainerCreateCreatedBody, error)
	StartContainer(id string) error
	StopContainer(id string, timeout time.Duration) error
	UpdateContainerResources(id string, updateConfig dockercontainer.UpdateConfig) error
	RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error
	InspectImageByRef(imageRef string) (*dockertypes.ImageInspect, error)
	InspectImageByID(imageID string) (*dockertypes.ImageInspect, error)
	ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.ImageSummary, error)
	PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error
	RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDeleteResponseItem, error)
	ImageHistory(id string) ([]dockerimagetypes.HistoryResponseItem, error)
	Logs(string, dockertypes.ContainerLogsOptions, StreamOptions) error
	Version() (*dockertypes.Version, error)
	Info() (*dockertypes.Info, error)
	CreateExec(string, dockertypes.ExecConfig) (*dockertypes.IDResponse, error)
	StartExec(string, dockertypes.ExecStartCheck, StreamOptions) error
	InspectExec(id string) (*dockertypes.ContainerExecInspect, error)
	AttachToContainer(string, dockertypes.ContainerAttachOptions, StreamOptions) error
	ResizeContainerTTY(id string, height, width uint) error
	ResizeExecTTY(id string, height, width uint) error
	GetContainerStats(id string) (*dockertypes.StatsJSON, error)
}

// Get a *dockerapi.Client, either using the endpoint passed in, or using
// DOCKER_HOST, DOCKER_TLS_VERIFY, and DOCKER_CERT path per their spec
func getDockerClient(dockerEndpoint string) (*dockerapi.Client, error) {
	if len(dockerEndpoint) > 0 {
		klog.Infof("Connecting to docker on %s", dockerEndpoint)
		return dockerapi.NewClient(dockerEndpoint, "", nil, nil)
	}
	return dockerapi.NewClientWithOpts(dockerapi.FromEnv)
}

// ConnectToDockerOrDie creates docker client connecting to docker daemon.
// If the endpoint passed in is "fake://", a fake docker client
// will be returned. The program exits if error occurs. The requestTimeout
// is the timeout for docker requests. If timeout is exceeded, the request
// will be cancelled and throw out an error. If requestTimeout is 0, a default
// value will be applied.
func ConnectToDockerOrDie(dockerEndpoint string, requestTimeout, imagePullProgressDeadline time.Duration,
	withTraceDisabled bool, enableSleep bool) Interface {
	if dockerEndpoint == FakeDockerEndpoint {
		fakeClient := NewFakeDockerClient()
		if withTraceDisabled {
			fakeClient = fakeClient.WithTraceDisabled()
		}

		if enableSleep {
			fakeClient.EnableSleep = true
		}
		return fakeClient
	}
	client, err := getDockerClient(dockerEndpoint)
	if err != nil {
		klog.Fatalf("Couldn't connect to docker: %v", err)
	}
	klog.Infof("Start docker client with request timeout=%v", requestTimeout)
	return newKubeDockerClient(client, requestTimeout, imagePullProgressDeadline)
}
