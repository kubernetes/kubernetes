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

package dockertools

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/term"
)

// ExecHandler knows how to execute a command in a running Docker container.
type ExecHandler interface {
	ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size, timeout time.Duration) error
}

// NsenterExecHandler executes commands in Docker containers using nsenter.
type NsenterExecHandler struct{}

// TODO should we support nsenter in a container, running with elevated privs and --pid=host?
func (*NsenterExecHandler) ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size, timeout time.Duration) error {
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
	if tty {
		p, err := kubecontainer.StartPty(command)
		if err != nil {
			return err
		}
		defer p.Close()

		// make sure to close the stdout stream
		defer stdout.Close()

		kubecontainer.HandleResizing(resize, func(size term.Size) {
			term.SetSize(p.Fd(), size)
		})

		if stdin != nil {
			go io.Copy(p, stdin)
		}

		if stdout != nil {
			go io.Copy(stdout, p)
		}
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
		if err := command.Start(); err != nil {
			return err
		}
	}
	if timeout > 0 {
		t := time.AfterFunc(timeout, func() {
			command.Process.Kill()
		})
		defer t.Stop()
	}
	err = command.Wait()
	if exitErr, ok := err.(*exec.ExitError); ok {
		return &utilexec.ExitErrorWrapper{ExitError: exitErr}
	}
	return err
}

// NativeExecHandler executes commands in Docker containers using Docker's exec API.
type NativeExecHandler struct{}

// ExecInContainer executes a command in a Docker container. It may leave a
// goroutine running the process in the container after the function returns
// because of a timeout. However, the goroutine does not leak, it terminates
// when the process exits.
func (*NativeExecHandler) ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size, timeout time.Duration) error {
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

	startExec := func() error {
		// Have to start this before the call to client.StartExec because client.StartExec is a blocking
		// call :-( Otherwise, resize events don't get processed and the terminal never resizes.
		kubecontainer.HandleResizing(resize, func(size term.Size) {
			client.ResizeExecTTY(execObj.ID, int(size.Height), int(size.Width))
		})

		startOpts := dockertypes.ExecStartCheck{Detach: false, Tty: tty}
		streamOpts := StreamOptions{
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

	// No timeout, block until startExec is finished.
	if timeout <= 0 {
		return startExec()
	}
	// Otherwise, run startExec in a new goroutine and wait for completion
	// or timeout, whatever happens first.
	ch := make(chan error, 1)
	go func() {
		ch <- startExec()
	}()
	select {
	case err := <-ch:
		return err
	case <-time.After(timeout):
		// FIXME: we should kill the process in the container, but the
		// Docker API doesn't support it. See
		// https://github.com/docker/docker/issues/9098.
		// For liveness probes this is probably okay, since the
		// container will be restarted. For readiness probes it means
		// that probe processes could start piling up.
		glog.Errorf("Exec session %s in container %s timed out, but process is still running!", execObj.ID, container.ID)

		// Return an utilexec.ExitError with code != 0, so that the
		// probe result will be probe.Failure, not probe.Unknown as for
		// errors that don't implement that interface.
		return utilexec.CodeExitError{
			Err:  fmt.Errorf("exec session timed out"),
			Code: 1,
		}
	}
}
