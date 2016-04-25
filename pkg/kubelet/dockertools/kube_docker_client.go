/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"time"

	dockermessage "github.com/docker/docker/pkg/jsonmessage"
	dockerstdcopy "github.com/docker/docker/pkg/stdcopy"
	dockerapi "github.com/docker/engine-api/client"
	dockertypes "github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// kubeDockerClient is a wrapped layer of docker client for kubelet internal use. This layer is added to:
//	1) Redirect stream for exec and attach operations.
//	2) Wrap the context in this layer to make the DockerInterface cleaner.
//	3) Stabilize the DockerInterface. The engine-api is still under active development, the interface
//	is not stabilized yet. However, the DockerInterface is used in many files in Kubernetes, we may
//	not want to change the interface frequently. With this layer, we can port the engine api to the
//	DockerInterface to avoid changing DockerInterface as much as possible.
//	(See
//	  * https://github.com/docker/engine-api/issues/89
//	  * https://github.com/docker/engine-api/issues/137
//	  * https://github.com/docker/engine-api/pull/140)
// TODO(random-liu): Swith to new docker interface by refactoring the functions in the old DockerInterface
// one by one.
type kubeDockerClient struct {
	client *dockerapi.Client
}

// Make sure that kubeDockerClient implemented the DockerInterface.
var _ DockerInterface = &kubeDockerClient{}

// the default ShmSize to use (in bytes) if not specified.
const defaultShmSize = int64(1024 * 1024 * 64)

// newKubeDockerClient creates an kubeDockerClient from an existing docker client.
func newKubeDockerClient(dockerClient *dockerapi.Client) DockerInterface {
	return &kubeDockerClient{
		client: dockerClient,
	}
}

// getDefaultContext returns the default context, now the default context is
// context.Background()
// TODO(random-liu): Add timeout and timeout handling mechanism.
func getDefaultContext() context.Context {
	return context.Background()
}

func (k *kubeDockerClient) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	containers, err := k.client.ContainerList(getDefaultContext(), options)
	if err != nil {
		return nil, err
	}
	apiContainers := []dockertypes.Container{}
	for _, c := range containers {
		apiContainers = append(apiContainers, dockertypes.Container(c))
	}
	return apiContainers, nil
}

func (d *kubeDockerClient) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	containerJSON, err := d.client.ContainerInspect(getDefaultContext(), id)
	if err != nil {
		if dockerapi.IsErrContainerNotFound(err) {
			return nil, containerNotFoundError{ID: id}
		}
		return nil, err
	}
	return &containerJSON, nil
}

func (d *kubeDockerClient) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error) {
	// we provide an explicit default shm size as to not depend on docker daemon.
	// TODO: evaluate exposing this as a knob in the API
	if opts.HostConfig != nil && opts.HostConfig.ShmSize <= 0 {
		opts.HostConfig.ShmSize = defaultShmSize
	}
	createResp, err := d.client.ContainerCreate(getDefaultContext(), opts.Config, opts.HostConfig, opts.NetworkingConfig, opts.Name)
	if err != nil {
		return nil, err
	}
	return &createResp, nil
}

func (d *kubeDockerClient) StartContainer(id string) error {
	return d.client.ContainerStart(getDefaultContext(), id)
}

// Stopping an already stopped container will not cause an error in engine-api.
func (d *kubeDockerClient) StopContainer(id string, timeout int) error {
	return d.client.ContainerStop(getDefaultContext(), id, timeout)
}

func (d *kubeDockerClient) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	return d.client.ContainerRemove(getDefaultContext(), id, opts)
}

func (d *kubeDockerClient) InspectImage(image string) (*dockertypes.ImageInspect, error) {
	resp, _, err := d.client.ImageInspectWithRaw(getDefaultContext(), image, true)
	if err != nil {
		if dockerapi.IsErrImageNotFound(err) {
			err = imageNotFoundError{ID: image}
		}
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	return d.client.ImageHistory(getDefaultContext(), id)
}

func (d *kubeDockerClient) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error) {
	images, err := d.client.ImageList(getDefaultContext(), opts)
	if err != nil {
		return nil, err
	}
	return images, nil
}

func base64EncodeAuth(auth dockertypes.AuthConfig) (string, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(auth); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(buf.Bytes()), nil
}

func (d *kubeDockerClient) PullImage(image string, auth dockertypes.AuthConfig, opts dockertypes.ImagePullOptions) error {
	// RegistryAuth is the base64 encoded credentials for the registry
	base64Auth, err := base64EncodeAuth(auth)
	if err != nil {
		return err
	}
	opts.RegistryAuth = base64Auth
	resp, err := d.client.ImagePull(getDefaultContext(), image, opts)
	if err != nil {
		return err
	}
	defer resp.Close()
	// TODO(random-liu): Use the image pulling progress information.
	decoder := json.NewDecoder(resp)
	for {
		var msg dockermessage.JSONMessage
		err := decoder.Decode(&msg)
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if msg.Error != nil {
			return msg.Error
		}
	}
	return nil
}

func (d *kubeDockerClient) RemoveImage(image string, opts dockertypes.ImageRemoveOptions) ([]dockertypes.ImageDelete, error) {
	return d.client.ImageRemove(getDefaultContext(), image, opts)
}

func (d *kubeDockerClient) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	resp, err := d.client.ContainerLogs(getDefaultContext(), id, opts)
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.redirectResponseToOutputStream(sopts.RawTerminal, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) Version() (*dockertypes.Version, error) {
	resp, err := d.client.ServerVersion(getDefaultContext())
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) Info() (*dockertypes.Info, error) {
	resp, err := d.client.Info(getDefaultContext())
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// TODO(random-liu): Add unit test for exec and attach functions, just like what go-dockerclient did.
func (d *kubeDockerClient) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error) {
	resp, err := d.client.ContainerExecCreate(getDefaultContext(), id, opts)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	if opts.Detach {
		return d.client.ContainerExecStart(getDefaultContext(), startExec, opts)
	}
	resp, err := d.client.ContainerExecAttach(getDefaultContext(), startExec, dockertypes.ExecConfig{
		Detach: opts.Detach,
		Tty:    opts.Tty,
	})
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.holdHijackedConnection(sopts.RawTerminal || opts.Tty, sopts.InputStream, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	resp, err := d.client.ContainerExecInspect(getDefaultContext(), id)
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	resp, err := d.client.ContainerAttach(getDefaultContext(), id, opts)
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.holdHijackedConnection(sopts.RawTerminal, sopts.InputStream, sopts.OutputStream, sopts.ErrorStream, resp)
}

// redirectResponseToOutputStream redirect the response stream to stdout and stderr. When tty is true, all stream will
// only be redirected to stdout.
func (d *kubeDockerClient) redirectResponseToOutputStream(tty bool, outputStream, errorStream io.Writer, resp io.Reader) error {
	if outputStream == nil {
		outputStream = ioutil.Discard
	}
	if errorStream == nil {
		errorStream = ioutil.Discard
	}
	var err error
	if tty {
		_, err = io.Copy(outputStream, resp)
	} else {
		_, err = dockerstdcopy.StdCopy(outputStream, errorStream, resp)
	}
	return err
}

// holdHijackedConnection hold the HijackedResponse, redirect the inputStream to the connection, and redirect the response
// stream to stdout and stderr. NOTE: If needed, we could also add context in this function.
func (d *kubeDockerClient) holdHijackedConnection(tty bool, inputStream io.Reader, outputStream, errorStream io.Writer, resp dockertypes.HijackedResponse) error {
	receiveStdout := make(chan error)
	if outputStream != nil || errorStream != nil {
		go func() {
			receiveStdout <- d.redirectResponseToOutputStream(tty, outputStream, errorStream, resp.Reader)
		}()
	}

	stdinDone := make(chan struct{})
	go func() {
		if inputStream != nil {
			io.Copy(resp.Conn, inputStream)
		}
		resp.CloseWrite()
		close(stdinDone)
	}()

	select {
	case err := <-receiveStdout:
		return err
	case <-stdinDone:
		if outputStream != nil || errorStream != nil {
			return <-receiveStdout
		}
	}
	return nil
}

// parseDockerTimestamp parses the timestamp returned by DockerInterface from string to time.Time
func parseDockerTimestamp(s string) (time.Time, error) {
	// Timestamp returned by Docker is in time.RFC3339Nano format.
	return time.Parse(time.RFC3339Nano, s)
}

// StreamOptions are the options used to configure the stream redirection
type StreamOptions struct {
	RawTerminal  bool
	InputStream  io.Reader
	OutputStream io.Writer
	ErrorStream  io.Writer
}

// containerNotFoundError is the error returned by InspectContainer when container not found. We
// add this error type for testability. We don't use the original error returned by engine-api
// because dockertypes.containerNotFoundError is private, we can't create and inject it in our test.
type containerNotFoundError struct {
	ID string
}

func (e containerNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such container: %s", e.ID)
}

// imageNotFoundError is the error returned by InspectImage when image not found.
type imageNotFoundError struct {
	ID string
}

func (e imageNotFoundError) Error() string {
	return fmt.Sprintf("Error: No such image: %s", e.ID)
}
