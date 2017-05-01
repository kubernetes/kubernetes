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
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
	"time"

	dockertypes "github.com/docker/engine-api/types"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/client/unversioned/remotecommand"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utilexec "k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/term"
)

// ExecHandler knows how to execute a command in a running Docker container.
type ExecHandler interface {
	ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error
}

// NsenterExecHandler executes commands in Docker containers using nsenter.
type NsenterExecHandler struct{}

// TODO should we support nsenter in a container, running with elevated privs and --pid=host?
func (*NsenterExecHandler) ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	nsenter, err := exec.LookPath("nsenter")
	if err != nil {
		return fmt.Errorf("exec unavailable - unable to locate nsenter")
	}

	cgexec, err := exec.LookPath("cgexec")
	if err != nil {
		return fmt.Errorf("exec unavailable - unable to locate cgexec")
	}

	containerPid := container.State.Pid

	args, _ := cgexecArgs(containerPid)
	// TODO what if the container doesn't have `env`???
	args = append(args, nsenter, "-t", fmt.Sprintf("%d", containerPid), "-m", "-i", "-u", "-n", "-p", "--", "env", "-i")
	args = append(args, fmt.Sprintf("HOSTNAME=%s", container.Config.Hostname))
	args = append(args, container.Config.Env...)
	args = append(args, cmd...)
	//cgexec -g memory:/system.slice/docker-bbe0d2d2a404c8650472eb7a57975d25178fa4a49c4d6b176f9e25527805b17a.scope -g ... nsenter -t 17153 -m -i -u -n -p -- env -i HOSTNAME=test1-3020125803-sqqvl -i ... /bin/bash
	command := exec.Command(cgexec, args...)
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

func (*NativeExecHandler) ExecInContainer(client DockerInterface, container *dockertypes.ContainerJSON, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
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

func cgexecArgs(pid int) ([]string, error) {
	//$ cat /proc/17153/cgroup
	//10:cpuset:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//9:perf_event:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//8:freezer:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//7:hugetlb:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//6:devices:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//5:memory:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//4:blkio:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//3:net_cls:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//2:cpuacct,cpu:/docker/2469451a734cd4699f15b090ce9d36554e89bcc913989907e8961d66a9aebc17
	//1:name=systemd:/system.slice/docker.service
	data, err := ioutil.ReadFile(fmt.Sprintf("/proc/%d/cgroup", pid))
	if err != nil {
		return nil, err
	}
	var (
		id         int
		cgroupPath string
		args       []string
	)
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		line := sc.Text()
		if n, err := fmt.Sscanf(line, "%d:%s", &id, &cgroupPath); n == 2 && err == nil {
			//skip name=systemd:/system.slice/docker.service
			if !strings.HasPrefix(cgroupPath, "name") {
				args = append(args, "-g", cgroupPath)
			}
		}
	}
	return args, nil
}
