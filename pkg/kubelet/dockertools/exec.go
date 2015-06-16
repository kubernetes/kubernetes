/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

// ExecHandler knows how to execute a command in a running Docker container.
type ExecHandler interface {
	ExecInContainer(client DockerInterface, container *docker.Container, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error
}

// NsenterExecHandler executes commands in Docker containers using nsenter.
type NsenterExecHandler struct{}

// TODO should we support nsenter in a container, running with elevated privs and --pid=host?
func (*NsenterExecHandler) ExecInContainer(client DockerInterface, container *docker.Container, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
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

		if stdin != nil {
			go io.Copy(p, stdin)
		}

		if stdout != nil {
			go io.Copy(stdout, p)
		}

		return command.Wait()
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

		return command.Run()
	}
}

// NativeExecHandler executes commands in Docker containers using Docker's exec API.
type NativeExecHandler struct{}

func (*NativeExecHandler) ExecInContainer(client DockerInterface, container *docker.Container, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	createOpts := docker.CreateExecOptions{
		Container:    container.ID,
		Cmd:          cmd,
		AttachStdin:  stdin != nil,
		AttachStdout: stdout != nil,
		AttachStderr: stderr != nil,
		Tty:          tty,
	}
	execObj, err := client.CreateExec(createOpts)
	if err != nil {
		return fmt.Errorf("failed to exec in container - Exec setup failed - %v", err)
	}
	startOpts := docker.StartExecOptions{
		Detach:       false,
		InputStream:  stdin,
		OutputStream: stdout,
		ErrorStream:  stderr,
		Tty:          tty,
		RawTerminal:  tty,
	}
	err = client.StartExec(execObj.ID, startOpts)
	if err != nil {
		return err
	}
	tick := time.Tick(2 * time.Second)
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

		<-tick
	}

	return err
}
