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
	"io"
	"io/ioutil"
	"strconv"

	"github.com/docker/docker/pkg/stdcopy"
	dockerapi "github.com/docker/engine-api/client"
	dockertypes "github.com/docker/engine-api/types"
	dockercontainer "github.com/docker/engine-api/types/container"
	dockerfilters "github.com/docker/engine-api/types/filters"
	docker "github.com/fsouza/go-dockerclient"
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

// convertType converts between different types with the same json format.
func convertType(src interface{}, dst interface{}) error {
	data, err := json.Marshal(src)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, dst)
}

// convertFilters converts filters to the filter type in engine-api.
func convertFilters(filters map[string][]string) dockerfilters.Args {
	args := dockerfilters.NewArgs()
	for name, fields := range filters {
		for _, field := range fields {
			args.Add(name, field)
		}
	}
	return args
}

// convertEnv converts data to a go-dockerclient Env
func convertEnv(src interface{}) (*docker.Env, error) {
	m := make(map[string]interface{})
	if err := convertType(&src, &m); err != nil {
		return nil, err
	}
	env := &docker.Env{}
	for k, v := range m {
		env.SetAuto(k, v)
	}
	return env, nil
}

func (k *kubeDockerClient) ListContainers(options docker.ListContainersOptions) ([]docker.APIContainers, error) {
	containers, err := k.client.ContainerList(getDefaultContext(), dockertypes.ContainerListOptions{
		Size:   options.Size,
		All:    options.All,
		Limit:  options.Limit,
		Since:  options.Since,
		Before: options.Before,
		Filter: convertFilters(options.Filters),
	})
	if err != nil {
		return nil, err
	}
	apiContainers := []docker.APIContainers{}
	if err := convertType(&containers, &apiContainers); err != nil {
		return nil, err
	}
	return apiContainers, nil
}

func (d *kubeDockerClient) InspectContainer(id string) (*docker.Container, error) {
	containerJSON, err := d.client.ContainerInspect(getDefaultContext(), id)
	if err != nil {
		// TODO(random-liu): Use IsErrContainerNotFound instead of NoSuchContainer error
		if dockerapi.IsErrContainerNotFound(err) {
			err = &docker.NoSuchContainer{ID: id, Err: err}
		}
		return nil, err
	}
	container := &docker.Container{}
	if err := convertType(&containerJSON, container); err != nil {
		return nil, err
	}
	return container, nil
}

func (d *kubeDockerClient) CreateContainer(opts docker.CreateContainerOptions) (*docker.Container, error) {
	config := &dockercontainer.Config{}
	if err := convertType(opts.Config, config); err != nil {
		return nil, err
	}
	hostConfig := &dockercontainer.HostConfig{}
	if err := convertType(opts.HostConfig, hostConfig); err != nil {
		return nil, err
	}
	resp, err := d.client.ContainerCreate(getDefaultContext(), config, hostConfig, nil, opts.Name)
	if err != nil {
		return nil, err
	}
	container := &docker.Container{}
	if err := convertType(&resp, container); err != nil {
		return nil, err
	}
	return container, nil
}

// TODO(random-liu): The HostConfig at container start is deprecated, will remove this in the following refactoring.
func (d *kubeDockerClient) StartContainer(id string, _ *docker.HostConfig) error {
	return d.client.ContainerStart(getDefaultContext(), id)
}

// Stopping an already stopped container will not cause an error in engine-api.
func (d *kubeDockerClient) StopContainer(id string, timeout uint) error {
	return d.client.ContainerStop(getDefaultContext(), id, int(timeout))
}

func (d *kubeDockerClient) RemoveContainer(opts docker.RemoveContainerOptions) error {
	return d.client.ContainerRemove(getDefaultContext(), dockertypes.ContainerRemoveOptions{
		ContainerID:   opts.ID,
		RemoveVolumes: opts.RemoveVolumes,
		Force:         opts.Force,
	})
}

func (d *kubeDockerClient) InspectImage(image string) (*docker.Image, error) {
	resp, _, err := d.client.ImageInspectWithRaw(getDefaultContext(), image, true)
	if err != nil {
		// TODO(random-liu): Use IsErrImageNotFound instead of ErrNoSuchImage
		if dockerapi.IsErrImageNotFound(err) {
			err = docker.ErrNoSuchImage
		}
		return nil, err
	}
	imageInfo := &docker.Image{}
	if err := convertType(&resp, imageInfo); err != nil {
		return nil, err
	}
	return imageInfo, nil
}

func (d *kubeDockerClient) ListImages(opts docker.ListImagesOptions) ([]docker.APIImages, error) {
	resp, err := d.client.ImageList(getDefaultContext(), dockertypes.ImageListOptions{
		MatchName: opts.Filter,
		All:       opts.All,
		Filters:   convertFilters(opts.Filters),
	})
	if err != nil {
		return nil, err
	}
	images := []docker.APIImages{}
	if err = convertType(&resp, &images); err != nil {
		return nil, err
	}
	return images, nil
}

func base64EncodeAuth(auth docker.AuthConfiguration) (string, error) {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(auth); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(buf.Bytes()), nil
}

func (d *kubeDockerClient) PullImage(opts docker.PullImageOptions, auth docker.AuthConfiguration) error {
	base64Auth, err := base64EncodeAuth(auth)
	if err != nil {
		return err
	}
	resp, err := d.client.ImagePull(getDefaultContext(), dockertypes.ImagePullOptions{
		ImageID:      opts.Repository,
		Tag:          opts.Tag,
		RegistryAuth: base64Auth,
	}, nil)
	if err != nil {
		return err
	}
	defer resp.Close()
	// TODO(random-liu): Use the image pulling progress information.
	_, err = io.Copy(ioutil.Discard, resp)
	return err
}

func (d *kubeDockerClient) RemoveImage(image string) error {
	_, err := d.client.ImageRemove(getDefaultContext(), dockertypes.ImageRemoveOptions{ImageID: image})
	return err
}

func (d *kubeDockerClient) Logs(opts docker.LogsOptions) error {
	resp, err := d.client.ContainerLogs(getDefaultContext(), dockertypes.ContainerLogsOptions{
		ContainerID: opts.Container,
		ShowStdout:  opts.Stdout,
		ShowStderr:  opts.Stderr,
		Since:       strconv.FormatInt(opts.Since, 10),
		Timestamps:  opts.Timestamps,
		Follow:      opts.Follow,
		Tail:        opts.Tail,
	})
	if err != nil {
		return err
	}
	defer resp.Close()
	return d.redirectResponseToOutputStream(opts.RawTerminal, opts.OutputStream, opts.ErrorStream, resp)
}

func (d *kubeDockerClient) Version() (*docker.Env, error) {
	resp, err := d.client.ServerVersion(getDefaultContext())
	if err != nil {
		return nil, err
	}
	return convertEnv(resp)
}

func (d *kubeDockerClient) Info() (*docker.Env, error) {
	resp, err := d.client.Info(getDefaultContext())
	if err != nil {
		return nil, err
	}
	return convertEnv(resp)
}

func (d *kubeDockerClient) CreateExec(opts docker.CreateExecOptions) (*docker.Exec, error) {
	cfg := dockertypes.ExecConfig{}
	if err := convertType(&opts, &cfg); err != nil {
		return nil, err
	}
	resp, err := d.client.ContainerExecCreate(getDefaultContext(), cfg)
	if err != nil {
		return nil, err
	}
	exec := &docker.Exec{}
	if err := convertType(&resp, exec); err != nil {
		return nil, err
	}
	return exec, nil
}

func (d *kubeDockerClient) StartExec(startExec string, opts docker.StartExecOptions) error {
	if opts.Detach {
		return d.client.ContainerExecStart(getDefaultContext(), startExec, dockertypes.ExecStartCheck{
			Detach: opts.Detach,
			Tty:    opts.Tty,
		})
	}
	resp, err := d.client.ContainerExecAttach(getDefaultContext(), startExec, dockertypes.ExecConfig{
		Detach: opts.Detach,
		Tty:    opts.Tty,
	})
	if err != nil {
		return err
	}
	defer resp.Close()
	if opts.Success != nil {
		opts.Success <- struct{}{}
		<-opts.Success
	}
	return d.holdHijackedConnection(opts.RawTerminal || opts.Tty, opts.InputStream, opts.OutputStream, opts.ErrorStream, resp)
}

func (d *kubeDockerClient) InspectExec(id string) (*docker.ExecInspect, error) {
	resp, err := d.client.ContainerExecInspect(getDefaultContext(), id)
	if err != nil {
		return nil, err
	}
	exec := &docker.ExecInspect{}
	if err := convertType(&resp, exec); err != nil {
		return nil, err
	}
	return exec, nil
}

func (d *kubeDockerClient) AttachToContainer(opts docker.AttachToContainerOptions) error {
	resp, err := d.client.ContainerAttach(getDefaultContext(), dockertypes.ContainerAttachOptions{
		ContainerID: opts.Container,
		Stream:      opts.Stream,
		Stdin:       opts.Stdin,
		Stdout:      opts.Stdout,
		Stderr:      opts.Stderr,
		// TODO: How to deal with the *Logs* here? There is no *Logs* field in the engine-api.
	})
	if err != nil {
		return err
	}
	defer resp.Close()
	if opts.Success != nil {
		opts.Success <- struct{}{}
		<-opts.Success
	}
	return d.holdHijackedConnection(opts.RawTerminal, opts.InputStream, opts.OutputStream, opts.ErrorStream, resp)
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
		_, err = stdcopy.StdCopy(outputStream, errorStream, resp)
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
