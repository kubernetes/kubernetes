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

const (
	// defaultTimeout is the default timeout of all docker operations.
	defaultTimeout = 2 * time.Minute

	// defaultShmSize is the default ShmSize to use (in bytes) if not specified.
	defaultShmSize = int64(1024 * 1024 * 64)
)

// newKubeDockerClient creates an kubeDockerClient from an existing docker client.
func newKubeDockerClient(dockerClient *dockerapi.Client) DockerInterface {
	return &kubeDockerClient{
		client: dockerClient,
	}
}

func (k *kubeDockerClient) ListContainers(options dockertypes.ContainerListOptions) ([]dockertypes.Container, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	containers, err := k.client.ContainerList(ctx, options)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return containers, nil
}

func (d *kubeDockerClient) InspectContainer(id string) (*dockertypes.ContainerJSON, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	containerJSON, err := d.client.ContainerInspect(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		if dockerapi.IsErrContainerNotFound(err) {
			return nil, containerNotFoundError{ID: id}
		}
		return nil, err
	}
	return &containerJSON, nil
}

func (d *kubeDockerClient) CreateContainer(opts dockertypes.ContainerCreateConfig) (*dockertypes.ContainerCreateResponse, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	// we provide an explicit default shm size as to not depend on docker daemon.
	// TODO: evaluate exposing this as a knob in the API
	if opts.HostConfig != nil && opts.HostConfig.ShmSize <= 0 {
		opts.HostConfig.ShmSize = defaultShmSize
	}
	createResp, err := d.client.ContainerCreate(ctx, opts.Config, opts.HostConfig, opts.NetworkingConfig, opts.Name)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &createResp, nil
}

func (d *kubeDockerClient) StartContainer(id string) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	err := d.client.ContainerStart(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

// Stopping an already stopped container will not cause an error in engine-api.
func (d *kubeDockerClient) StopContainer(id string, timeout int) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	err := d.client.ContainerStop(ctx, id, timeout)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

func (d *kubeDockerClient) RemoveContainer(id string, opts dockertypes.ContainerRemoveOptions) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	err := d.client.ContainerRemove(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	return err
}

func (d *kubeDockerClient) InspectImage(image string) (*dockertypes.ImageInspect, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, _, err := d.client.ImageInspectWithRaw(ctx, image, true)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		if dockerapi.IsErrImageNotFound(err) {
			err = imageNotFoundError{ID: image}
		}
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) ImageHistory(id string) ([]dockertypes.ImageHistory, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ImageHistory(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	return resp, err
}

func (d *kubeDockerClient) ListImages(opts dockertypes.ImageListOptions) ([]dockertypes.Image, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	images, err := d.client.ImageList(ctx, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
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
	ctx, cancel := getDefaultContext()
	defer cancel()
	opts.RegistryAuth = base64Auth
	resp, err := d.client.ImagePull(ctx, image, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
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
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ImageRemove(ctx, image, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	return resp, err
}

func (d *kubeDockerClient) Logs(id string, opts dockertypes.ContainerLogsOptions, sopts StreamOptions) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ContainerLogs(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.redirectResponseToOutputStream(sopts.RawTerminal, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) Version() (*dockertypes.Version, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ServerVersion(ctx)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) Info() (*dockertypes.Info, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.Info(ctx)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

// TODO(random-liu): Add unit test for exec and attach functions, just like what go-dockerclient did.
func (d *kubeDockerClient) CreateExec(id string, opts dockertypes.ExecConfig) (*dockertypes.ContainerExecCreateResponse, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ContainerExecCreate(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) StartExec(startExec string, opts dockertypes.ExecStartCheck, sopts StreamOptions) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	if opts.Detach {
		err := d.client.ContainerExecStart(ctx, startExec, opts)
		if ctxErr := contextError(ctx); ctxErr != nil {
			return ctxErr
		}
		return err
	}
	resp, err := d.client.ContainerExecAttach(ctx, startExec, dockertypes.ExecConfig{
		Detach: opts.Detach,
		Tty:    opts.Tty,
	})
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.holdHijackedConnection(sopts.RawTerminal || opts.Tty, sopts.InputStream, sopts.OutputStream, sopts.ErrorStream, resp)
}

func (d *kubeDockerClient) InspectExec(id string) (*dockertypes.ContainerExecInspect, error) {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ContainerExecInspect(ctx, id)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return nil, ctxErr
	}
	if err != nil {
		return nil, err
	}
	return &resp, nil
}

func (d *kubeDockerClient) AttachToContainer(id string, opts dockertypes.ContainerAttachOptions, sopts StreamOptions) error {
	ctx, cancel := getDefaultContext()
	defer cancel()
	resp, err := d.client.ContainerAttach(ctx, id, opts)
	if ctxErr := contextError(ctx); ctxErr != nil {
		return ctxErr
	}
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

func getDefaultContext() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), defaultTimeout)
}

// contextError checks the context, and returns error if the context is timeout.
func contextError(ctx context.Context) error {
	if ctx.Err() == context.DeadlineExceeded {
		return operationTimeout{err: ctx.Err()}
	}
	return ctx.Err()
}

// StreamOptions are the options used to configure the stream redirection
type StreamOptions struct {
	RawTerminal  bool
	InputStream  io.Reader
	OutputStream io.Writer
	ErrorStream  io.Writer
}

// operationTimeout is the error returned when the docker operations are timeout.
type operationTimeout struct {
	err error
}

func (e operationTimeout) Error() string {
	return fmt.Sprintf("operation timeout: %v", e.err)
}

// containerNotFoundError is the error returned by InspectContainer when container not found. We
// add this error type for testability. We don't use the original error returned by engine-api
// because dockertypes.containerNotFoundError is private, we can't create and inject it in our test.
type containerNotFoundError struct {
	ID string
}

func (e containerNotFoundError) Error() string {
	return fmt.Sprintf("no such container: %q", e.ID)
}

// imageNotFoundError is the error returned by InspectImage when image not found.
type imageNotFoundError struct {
	ID string
}

func (e imageNotFoundError) Error() string {
	return fmt.Sprintf("no such image: %q", e.ID)
}
