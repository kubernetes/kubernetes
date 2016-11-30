/*
Copyright 2016 The Kubernetes Authors.

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

package rktshim

import (
	"bytes"
	"errors"
	"io"
	"math/rand"
	"time"

	kubeletapi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
)

const (
	FakeStreamingHost = "localhost"
	FakeStreamingPort = "12345"
)

func init() {
	// Don't randomize due to testing purposes
	// rand.Seed(time.Now().UnixNano())
}

func randString(n int) string {
	runeAlphabet := []rune("abcdefghijklmnopqrstuvwxyz")
	dictLen := len(runeAlphabet)

	buf := make([]rune, n)
	for i := range buf {
		buf[i] = runeAlphabet[rand.Intn(dictLen)]
	}

	return string(buf)
}

var (
	ErrContainerNotFound               = errors.New("rktshim: container not found")
	ErrInvalidContainerStateTransition = errors.New("rktshim: wrong container operation for current state")
)

type FakeRuntime struct {
	Containers containerRegistry
}

type FakeRuntimeConfig struct{}

func NewFakeRuntime() (kubeletapi.ContainerManager, error) {
	return &FakeRuntime{Containers: make(containerRegistry)}, nil
}

type characterStreams struct {
	In  io.Reader
	Out io.Writer
	Err io.Writer
}

func newCharacterStreams(in io.Reader, out io.Writer, err io.Writer) characterStreams {
	std := characterStreams{in, out, err}

	return std
}

type fakeContainer struct {
	Config *runtimeapi.ContainerConfig

	Status runtimeapi.ContainerStatus

	State runtimeapi.ContainerState

	Streams characterStreams
}

func (c *fakeContainer) Start() {
	c.State = runtimeapi.ContainerState_CONTAINER_RUNNING

	c.Status.State = &c.State
}

func (c *fakeContainer) Stop() {
	c.State = runtimeapi.ContainerState_CONTAINER_EXITED

	c.Status.State = &c.State

	exitSuccess := int32(0)
	c.Status.ExitCode = &exitSuccess

	// c.Status.Reason
}

func (c *fakeContainer) Exec(cmd []string, in io.Reader, out, err io.WriteCloser) error {
	// TODO(tmrts): incomplete command execution logic
	// c.StreamCompare(c.Streams.In, s.InputStream)
	// c.StreamFlush(c.Streams.Out, s.OutputStream)
	// c.StreamFlush(c.Streams.Err, s.ErrorStream)

	return nil
}

type containerRegistry map[string]*fakeContainer

func (r *FakeRuntime) CreateContainer(pid string, cfg *runtimeapi.ContainerConfig, sandboxCfg *runtimeapi.PodSandboxConfig) (string, error) {
	// TODO(tmrts): allow customization
	containerIDLength := 8

	cid := randString(containerIDLength)

	r.Containers[cid] = &fakeContainer{
		Config:  cfg,
		Streams: newCharacterStreams(nil, nil, nil),
	}

	return cid, nil
}

func (r *FakeRuntime) StartContainer(id string) error {
	c, ok := r.Containers[id]
	if !ok {
		return ErrContainerNotFound
	}
	switch c.State {
	case runtimeapi.ContainerState_CONTAINER_EXITED:
		fallthrough
	case runtimeapi.ContainerState_CONTAINER_CREATED:
		c.Start()
	case runtimeapi.ContainerState_CONTAINER_UNKNOWN:
		// TODO(tmrts): add timeout to Start API or generalize timeout somehow
		//<-time.After(time.Duration(timeout) * time.Second)
		fallthrough
	default:
		return ErrInvalidContainerStateTransition
	}

	return nil
}

func (r *FakeRuntime) StopContainer(id string, timeout int64) error {
	c, ok := r.Containers[id]
	if !ok {
		return ErrContainerNotFound
	}

	switch c.State {
	case runtimeapi.ContainerState_CONTAINER_RUNNING:
		c.State = runtimeapi.ContainerState_CONTAINER_EXITED // This state might not be the best one
	case runtimeapi.ContainerState_CONTAINER_UNKNOWN:
		<-time.After(time.Duration(timeout) * time.Second)
		fallthrough
	default:
		return ErrInvalidContainerStateTransition
	}

	return nil
}

func (r *FakeRuntime) RemoveContainer(id string) error {
	_, ok := r.Containers[id]
	if !ok {
		return ErrContainerNotFound
	}

	// Remove regardless of the container state
	delete(r.Containers, id)

	return nil
}

func (r *FakeRuntime) ListContainers(*runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	list := []*runtimeapi.Container{}

	// TODO(tmrts): apply the filter
	for _, c := range r.Containers {
		list = append(list, &runtimeapi.Container{
			Id:       c.Status.Id,
			Metadata: c.Config.Metadata,
			Labels:   c.Config.Labels,
			ImageRef: c.Status.ImageRef,
			State:    &c.State,
		})
	}

	return list, nil
}

func (r *FakeRuntime) ContainerStatus(id string) (*runtimeapi.ContainerStatus, error) {
	c, ok := r.Containers[id]
	if !ok {
		return &runtimeapi.ContainerStatus{}, ErrContainerNotFound
	}

	return &c.Status, nil
}

func (r *FakeRuntime) ExecSync(containerID string, cmd []string, timeout time.Duration) (stdout []byte, stderr []byte, err error) {
	c, ok := r.Containers[containerID]
	if !ok {
		return nil, nil, ErrContainerNotFound
	}

	// TODO(tmrts): Validate the assumption that container has to be running for exec to work.
	if c.State != runtimeapi.ContainerState_CONTAINER_RUNNING {
		return nil, nil, ErrInvalidContainerStateTransition
	}

	var stdoutBuffer, stderrBuffer bytes.Buffer
	err = c.Exec(cmd, nil,
		ioutils.WriteCloserWrapper(&stdoutBuffer),
		ioutils.WriteCloserWrapper(&stderrBuffer))
	return stdoutBuffer.Bytes(), stderrBuffer.Bytes(), err
}

func (r *FakeRuntime) Exec(req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	url := "http://" + FakeStreamingHost + ":" + FakeStreamingPort + "/exec/" + req.GetContainerId()
	return &runtimeapi.ExecResponse{
		Url: &url,
	}, nil
}

func (r *FakeRuntime) Attach(req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	url := "http://" + FakeStreamingHost + ":" + FakeStreamingPort + "/attach/" + req.GetContainerId()
	return &runtimeapi.AttachResponse{
		Url: &url,
	}, nil
}
