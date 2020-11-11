// +build !dockerless

/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"io"
	"strings"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"k8s.io/klog/v2"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/probe/exec"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// ExecHandler knows how to execute a command in a running Docker container.
type ExecHandler interface {
	ExecInContainer(client libdocker.Interface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error
}

type dockerExitError struct {
	Inspect *dockertypes.ContainerExecInspect
}

func (d *dockerExitError) String() string {
	return d.Error()
}

func (d *dockerExitError) Error() string {
	return fmt.Sprintf("Error executing in Docker Container: %d", d.Inspect.ExitCode)
}

func (d *dockerExitError) Exited() bool {
	return !d.Inspect.Running
}

func (d *dockerExitError) ExitStatus() int {
	return d.Inspect.ExitCode
}

// NativeExecHandler executes commands in Docker containers using Docker's exec API.
type NativeExecHandler struct{}

// ExecInContainer executes the cmd in container using the Docker's exec API
func (*NativeExecHandler) ExecInContainer(client libdocker.Interface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	done := make(chan struct{})
	defer close(done)

	createOpts := dockertypes.ExecConfig{
		Cmd:          cmd,
		AttachStdin:  stdin != nil,
		AttachStdout: stdout != nil,
		AttachStderr: stderr != nil,
		Tty:          tty,
	}
	execObj, err := client.CreateExec(container.ID, createOpts)
	if err != nil {
		return fmt.Errorf("failed to exec in container - Exec setup failed - %v", err)
	}

	// Have to start this before the call to client.StartExec because client.StartExec is a blocking
	// call :-( Otherwise, resize events don't get processed and the terminal never resizes.
	//
	// We also have to delay attempting to send a terminal resize request to docker until after the
	// exec has started; otherwise, the initial resize request will fail.
	execStarted := make(chan struct{})
	go func() {
		select {
		case <-execStarted:
			// client.StartExec has started the exec, so we can start resizing
		case <-done:
			// ExecInContainer has returned, so short-circuit
			return
		}

		kubecontainer.HandleResizing(resize, func(size remotecommand.TerminalSize) {
			client.ResizeExecTTY(execObj.ID, uint(size.Height), uint(size.Width))
		})
	}()

	startOpts := dockertypes.ExecStartCheck{Detach: false, Tty: tty}
	streamOpts := libdocker.StreamOptions{
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  tty,
		ExecStarted:  execStarted,
	}
	err = client.StartExec(execObj.ID, startOpts, streamOpts)
	if err != nil {
		return err
	}

	// if ExecProbeTimeout feature gate is disabled, preserve existing behavior to ignore exec timeouts
	var execTimeout <-chan time.Time
	if timeout > 0 && utilfeature.DefaultFeatureGate.Enabled(features.ExecProbeTimeout) {
		execTimeout = time.After(timeout)
	} else {
		// skip exec timeout if provided timeout is 0
		execTimeout = nil
	}

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	count := 0
	for {
		select {
		case <-execTimeout:
			return exec.NewTimeoutError(fmt.Errorf("command %q timed out", strings.Join(cmd, " ")), timeout)
		// need to use "default" here instead of <-ticker.C, otherwise we delay the initial InspectExec by 2 seconds.
		default:
			inspect, inspectErr := client.InspectExec(execObj.ID)
			if inspectErr != nil {
				return inspectErr
			}

			if !inspect.Running {
				if inspect.ExitCode != 0 {
					return &dockerExitError{inspect}
				}

				return nil
			}

			// Only limit the amount of InspectExec calls if the exec timeout was not set.
			// When a timeout is not set, we stop polling the exec session after 5 attempts and allow the process to continue running.
			if execTimeout == nil {
				count++
				if count == 5 {
					klog.Errorf("Exec session %s in container %s terminated but process still running!", execObj.ID, container.ID)
					return nil
				}
			}

			<-ticker.C
		}
	}
}
