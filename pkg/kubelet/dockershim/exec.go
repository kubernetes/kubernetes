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
	"os"
	"os/exec"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	"github.com/golang/glog"

	"k8s.io/client-go/tools/remotecommand"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/term"
	utilexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
)

// ExecHandler knows how to execute a command in a running Docker container.
type ExecHandler interface {
	ExecInContainer(client libdocker.Interface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error
}

// NsenterExecHandler executes commands in Docker containers using nsenter.
type NsenterExecHandler struct{}

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

// TODO should we support nsenter in a container, running with elevated privs and --pid=host?
func (*NsenterExecHandler) ExecInContainer(client libdocker.Interface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	nsenter, err := exec.LookPath("nsenter")
	if err != nil {
		return fmt.Errorf("exec unavailable - unable to locate nsenter")
	}

	containerPid := container.State.Pid

	// TODO what if the container doesn't have `env`???
	args := []string{"-t", fmt.Sprintf("%d", containerPid), "-m", "-i", "-u", "-n", "-p", "--", "env", "-i"}
	args = append(args, fmt.Sprintf("HOSTNAME=%s", container.Config.Hostname))
	args = append(args, container.Config.Env...)
	args = append(args, cmd...)
	command := exec.Command(nsenter, args...)
	var cmdErr error
	if tty {
		p, err := kubecontainer.StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		kubecontainer.HandleResizing(resize, func(size remotecommand.TerminalSize) {
			term.SetSize(p.Fd(), size)
		})

		if stdin != nil {
			go io.Copy(p, stdin)
		}

		if stdout != nil {
			go io.Copy(stdout, p)
		}

		cmdErr = command.Wait()
	} else {
		if stdin != nil {
			// Use an os.Pipe here as it returns true *os.File objects.
			// This way, if you run 'kubectl exec <pod> -i bash' (no tty) and type 'exit',
			// the call below to command.Run() can unblock because its Stdin is the read half
			// of the pipe.
			r, w, err := os.Pipe()
			if err != nil {
				return err
			}
			go io.Copy(w, stdin)

			command.Stdin = r
		}
		if stdout != nil {
			command.Stdout = stdout
		}
		if stderr != nil {
			command.Stderr = stderr
		}

		cmdErr = command.Run()
	}

	if exitErr, ok := cmdErr.(*exec.ExitError); ok {
		return &utilexec.ExitErrorWrapper{ExitError: exitErr}
	}
	return cmdErr
}

// NativeExecHandler executes commands in Docker containers using Docker's exec API.
type NativeExecHandler struct{}

func (*NativeExecHandler) ExecInContainer(client libdocker.Interface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
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
	kubecontainer.HandleResizing(resize, func(size remotecommand.TerminalSize) {
		client.ResizeExecTTY(execObj.ID, uint(size.Height), uint(size.Width))
	})

	startOpts := dockertypes.ExecStartCheck{Detach: false, Tty: tty}
	streamOpts := libdocker.StreamOptions{
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		RawTerminal:  tty,
	}
	err = client.StartExec(execObj.ID, startOpts, streamOpts)
	if err != nil {
		return err
	}

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	count := 0
	for {
		inspect, err2 := client.InspectExec(execObj.ID)
		if err2 != nil {
			return err2
		}
		if !inspect.Running {
			if inspect.ExitCode != 0 {
				err = &dockerExitError{inspect}
			}
			break
		}

		count++
		if count == 5 {
			glog.Errorf("Exec session %s in container %s terminated but process still running!", execObj.ID, container.ID)
			break
		}

		<-ticker.C
	}

	return err
}
