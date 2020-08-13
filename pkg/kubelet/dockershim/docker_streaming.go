// +build !dockerless

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

package dockershim

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"k8s.io/client-go/tools/remotecommand"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/cri/streaming"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
	utilexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

type streamingRuntime struct {
	client      libdocker.Interface
	execHandler ExecHandler
}

var _ streaming.Runtime = &streamingRuntime{}

const maxMsgSize = 1024 * 1024 * 16

func (r *streamingRuntime) Exec(containerID string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	return r.exec(containerID, cmd, in, out, err, tty, resize, 0)
}

// Internal version of Exec adds a timeout.
func (r *streamingRuntime) exec(containerID string, cmd []string, in io.Reader, out, errw io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	container, err := checkContainerStatus(r.client, containerID)
	if err != nil {
		return err
	}
	return r.execHandler.ExecInContainer(r.client, container, cmd, in, out, errw, tty, resize, timeout)
}

func (r *streamingRuntime) Attach(containerID string, in io.Reader, out, errw io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	_, err := checkContainerStatus(r.client, containerID)
	if err != nil {
		return err
	}

	return attachContainer(r.client, containerID, in, out, errw, tty, resize)
}

func (r *streamingRuntime) PortForward(podSandboxID string, port int32, stream io.ReadWriteCloser) error {
	if port < 0 || port > math.MaxUint16 {
		return fmt.Errorf("invalid port %d", port)
	}
	return r.portForward(podSandboxID, port, stream)
}

// ExecSync executes a command in the container, and returns the stdout output.
// If command exits with a non-zero exit code, an error is returned.
func (ds *dockerService) ExecSync(_ context.Context, req *runtimeapi.ExecSyncRequest) (*runtimeapi.ExecSyncResponse, error) {
	timeout := time.Duration(req.Timeout) * time.Second
	var stdoutBuffer, stderrBuffer bytes.Buffer
	err := ds.streamingRuntime.exec(req.ContainerId, req.Cmd,
		nil, // in
		ioutils.WriteCloserWrapper(ioutils.LimitWriter(&stdoutBuffer, maxMsgSize)),
		ioutils.WriteCloserWrapper(ioutils.LimitWriter(&stderrBuffer, maxMsgSize)),
		false, // tty
		nil,   // resize
		timeout)

	var exitCode int32
	if err != nil {
		exitError, ok := err.(utilexec.ExitError)
		if !ok {
			return nil, err
		}

		exitCode = int32(exitError.ExitStatus())
	}
	return &runtimeapi.ExecSyncResponse{
		Stdout:   stdoutBuffer.Bytes(),
		Stderr:   stderrBuffer.Bytes(),
		ExitCode: exitCode,
	}, nil
}

// Exec prepares a streaming endpoint to execute a command in the container, and returns the address.
func (ds *dockerService) Exec(_ context.Context, req *runtimeapi.ExecRequest) (*runtimeapi.ExecResponse, error) {
	if ds.streamingServer == nil {
		return nil, streaming.NewErrorStreamingDisabled("exec")
	}
	_, err := checkContainerStatus(ds.client, req.ContainerId)
	if err != nil {
		return nil, err
	}
	return ds.streamingServer.GetExec(req)
}

// Attach prepares a streaming endpoint to attach to a running container, and returns the address.
func (ds *dockerService) Attach(_ context.Context, req *runtimeapi.AttachRequest) (*runtimeapi.AttachResponse, error) {
	if ds.streamingServer == nil {
		return nil, streaming.NewErrorStreamingDisabled("attach")
	}
	_, err := checkContainerStatus(ds.client, req.ContainerId)
	if err != nil {
		return nil, err
	}
	return ds.streamingServer.GetAttach(req)
}

// PortForward prepares a streaming endpoint to forward ports from a PodSandbox, and returns the address.
func (ds *dockerService) PortForward(_ context.Context, req *runtimeapi.PortForwardRequest) (*runtimeapi.PortForwardResponse, error) {
	if ds.streamingServer == nil {
		return nil, streaming.NewErrorStreamingDisabled("port forward")
	}
	_, err := checkContainerStatus(ds.client, req.PodSandboxId)
	if err != nil {
		return nil, err
	}
	// TODO(tallclair): Verify that ports are exposed.
	return ds.streamingServer.GetPortForward(req)
}

func checkContainerStatus(client libdocker.Interface, containerID string) (*dockertypes.ContainerJSON, error) {
	container, err := client.InspectContainer(containerID)
	if err != nil {
		return nil, err
	}
	if !container.State.Running {
		return nil, fmt.Errorf("container not running (%s)", container.ID)
	}
	return container, nil
}

func attachContainer(client libdocker.Interface, containerID string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	// Have to start this before the call to client.AttachToContainer because client.AttachToContainer is a blocking
	// call :-( Otherwise, resize events don't get processed and the terminal never resizes.
	kubecontainer.HandleResizing(resize, func(size remotecommand.TerminalSize) {
		client.ResizeContainerTTY(containerID, uint(size.Height), uint(size.Width))
	})

	// TODO(random-liu): Do we really use the *Logs* field here?
	opts := dockertypes.ContainerAttachOptions{
		Stream: true,
		Stdin:  stdin != nil,
		Stdout: stdout != nil,
		Stderr: stderr != nil,
	}
	sopts := libdocker.StreamOptions{
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  tty,
	}
	return client.AttachToContainer(containerID, opts, sopts)
}
