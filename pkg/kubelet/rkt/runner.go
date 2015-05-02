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

package rkt

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"strings"

	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/coreos/go-systemd/unit"
	"github.com/golang/glog"
	"github.com/kr/pty"
)

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *Runtime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	glog.V(4).Infof("Rkt running in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return nil, err
	}
	// TODO(yifan): Use appName instead of imageID.
	// see https://github.com/coreos/rkt/pull/640
	args := append([]string{}, "enter", "--imageid", id.imageID, id.uuid)
	args = append(args, cmd...)

	result, err := r.runCommand(args...)
	return []byte(strings.Join(result, "\n")), err
}

// Note: In rkt, the container ID is in the form of "UUID:appName:ImageID", where
// appName is the container name.
func (r *Runtime) ExecInContainer(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	glog.V(4).Infof("Rkt execing in container.")

	id, err := parseContainerID(containerID)
	if err != nil {
		return err
	}
	// TODO(yifan): Use appName instead of imageID.
	// see https://github.com/coreos/rkt/pull/640
	args := append([]string{}, "enter", "--imageid", id.imageID, id.uuid)
	args = append(args, cmd...)
	command := r.buildCommand(args...)

	if tty {
		// TODO(yifan): Merge with dockertools.StartPty().
		p, err := pty.Start(command)
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
	}
	if stdin != nil {
		// Use an os.Pipe here as it returns true *os.File objects.
		// This way, if you run 'kubectl exec -p <pod> -i bash' (no tty) and type 'exit',
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

// findRktID returns the rkt uuid for the pod.
// TODO(yifan): This is unefficient which require us to list
// all the unit files.
func (r *Runtime) findRktID(pod kubecontainer.Pod) (string, error) {
	units, err := r.systemd.ListUnits()
	if err != nil {
		return "", err
	}

	unitName := makePodServiceFileName(pod.ID)
	for _, u := range units {
		// u.Name contains file name ext such as .service, .socket, etc.
		if u.Name != unitName {
			continue
		}

		f, err := os.Open(path.Join(systemdServiceDir, u.Name))
		if err != nil {
			return "", err
		}
		defer f.Close()

		opts, err := unit.Deserialize(f)
		if err != nil {
			return "", err
		}

		for _, opt := range opts {
			if opt.Section == unitKubernetesSection && opt.Name == unitRktID {
				return opt.Value, nil
			}
		}
	}
	return "", fmt.Errorf("rkt uuid not found for pod %v", pod)
}

// PortForward executes socat in the pod's network namespace and copies
// data between stream (representing the user's local connection on their
// computer) and the specified port in the container.
//
// TODO:
//  - match cgroups of container
//  - should we support nsenter + socat on the host? (current impl)
//  - should we support nsenter + socat in a container, running with elevated privs and --pid=host?
//
// TODO(yifan): Merge with the same function in dockertools.
func (r *Runtime) PortForward(pod kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	glog.V(4).Infof("Rkt port forwarding in container.")

	podInfos, err := r.getPodInfos()
	if err != nil {
		return err
	}

	rktID, err := r.findRktID(pod)
	if err != nil {
		return err
	}

	info, ok := podInfos[rktID]
	if !ok {
		return fmt.Errorf("cannot find the pod info for pod %v", pod)
	}
	if info.pid < 0 {
		return fmt.Errorf("cannot get the pid for pod %v", pod)
	}

	_, lookupErr := exec.LookPath("socat")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: socat not found.")
	}
	args := []string{"-t", fmt.Sprintf("%d", info.pid), "-n", "socat", "-", fmt.Sprintf("TCP4:localhost:%d", port)}

	_, lookupErr = exec.LookPath("nsenter")
	if lookupErr != nil {
		return fmt.Errorf("unable to do port forwarding: nsenter not found.")
	}
	command := exec.Command("nsenter", args...)
	command.Stdin = stream
	command.Stdout = stream
	return command.Run()
}
