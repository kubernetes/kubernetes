/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"path/filepath"
	goruntime "runtime"
	"strings"

	pkgerrors "github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/errors"
	kubeadmapiv1beta1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta1"
	utilsexec "k8s.io/utils/exec"
)

// ContainerRuntime is an interface for working with container runtimes
type ContainerRuntime interface {
	IsDocker() bool
	IsRunning() error
	ListKubeContainers() ([]string, error)
	RemoveContainers(containers []string) error
	PullImage(image string) error
	ImageExists(image string) (bool, error)
}

// CRIRuntime is a struct that interfaces with the CRI
type CRIRuntime struct {
	exec      utilsexec.Interface
	criSocket string
}

// DockerRuntime is a struct that interfaces with the Docker daemon
type DockerRuntime struct {
	exec utilsexec.Interface
}

// NewContainerRuntime sets up and returns a ContainerRuntime struct
func NewContainerRuntime(execer utilsexec.Interface, criSocket string) (ContainerRuntime, error) {
	var toolName string
	var runtime ContainerRuntime

	if criSocket != kubeadmapiv1beta1.DefaultCRISocket {
		toolName = "crictl"
		// !!! temporary work around crictl warning:
		// Using "/var/run/crio/crio.sock" as endpoint is deprecated,
		// please consider using full url format "unix:///var/run/crio/crio.sock"
		if filepath.IsAbs(criSocket) && goruntime.GOOS != "windows" {
			criSocket = "unix://" + criSocket
		}
		runtime = &CRIRuntime{execer, criSocket}
	} else {
		toolName = "docker"
		runtime = &DockerRuntime{execer}
	}

	if _, err := execer.LookPath(toolName); err != nil {
		return nil, pkgerrors.Wrapf(err, "%s is required for container runtime", toolName)
	}

	return runtime, nil
}

// IsDocker returns true if the runtime is docker
func (runtime *CRIRuntime) IsDocker() bool {
	return false
}

// IsDocker returns true if the runtime is docker
func (runtime *DockerRuntime) IsDocker() bool {
	return true
}

// IsRunning checks if runtime is running
func (runtime *CRIRuntime) IsRunning() error {
	if out, err := runtime.exec.Command("crictl", "-r", runtime.criSocket, "info").CombinedOutput(); err != nil {
		return pkgerrors.Wrapf(err, "container runtime is not running: output: %s, error", string(out))
	}
	return nil
}

// IsRunning checks if runtime is running
func (runtime *DockerRuntime) IsRunning() error {
	if out, err := runtime.exec.Command("docker", "info").CombinedOutput(); err != nil {
		return pkgerrors.Wrapf(err, "container runtime is not running: output: %s, error", string(out))
	}
	return nil
}

// ListKubeContainers lists running k8s CRI pods
func (runtime *CRIRuntime) ListKubeContainers() ([]string, error) {
	out, err := runtime.exec.Command("crictl", "-r", runtime.criSocket, "pods", "-q").CombinedOutput()
	if err != nil {
		return nil, pkgerrors.Wrapf(err, "output: %s, error", string(out))
	}
	pods := []string{}
	for _, pod := range strings.Fields(string(out)) {
		pods = append(pods, pod)
	}
	return pods, nil
}

// ListKubeContainers lists running k8s containers
func (runtime *DockerRuntime) ListKubeContainers() ([]string, error) {
	output, err := runtime.exec.Command("docker", "ps", "-a", "--filter", "name=k8s_", "-q").CombinedOutput()
	return strings.Fields(string(output)), err
}

// RemoveContainers removes running k8s pods
func (runtime *CRIRuntime) RemoveContainers(containers []string) error {
	errs := []error{}
	for _, container := range containers {
		out, err := runtime.exec.Command("crictl", "-r", runtime.criSocket, "stopp", container).CombinedOutput()
		if err != nil {
			// don't stop on errors, try to remove as many containers as possible
			errs = append(errs, pkgerrors.Wrapf(err, "failed to stop running pod %s: output: %s, error", container, string(out)))
		} else {
			out, err = runtime.exec.Command("crictl", "-r", runtime.criSocket, "rmp", container).CombinedOutput()
			if err != nil {
				errs = append(errs, pkgerrors.Wrapf(err, "failed to remove running container %s: output: %s, error", container, string(out)))
			}
		}
	}
	return errors.NewAggregate(errs)
}

// RemoveContainers removes running containers
func (runtime *DockerRuntime) RemoveContainers(containers []string) error {
	errs := []error{}
	for _, container := range containers {
		out, err := runtime.exec.Command("docker", "rm", "--force", "--volumes", container).CombinedOutput()
		if err != nil {
			// don't stop on errors, try to remove as many containers as possible
			errs = append(errs, pkgerrors.Wrapf(err, "failed to remove running container %s: output: %s, error", container, string(out)))
		}
	}
	return errors.NewAggregate(errs)
}

// PullImage pulls the image
func (runtime *CRIRuntime) PullImage(image string) error {
	out, err := runtime.exec.Command("crictl", "-r", runtime.criSocket, "pull", image).CombinedOutput()
	if err != nil {
		return pkgerrors.Wrapf(err, "output: %s, error", string(out))
	}
	return nil
}

// PullImage pulls the image
func (runtime *DockerRuntime) PullImage(image string) error {
	out, err := runtime.exec.Command("docker", "pull", image).CombinedOutput()
	if err != nil {
		return pkgerrors.Wrapf(err, "output: %s, error", string(out))
	}
	return nil
}

// ImageExists checks to see if the image exists on the system
func (runtime *CRIRuntime) ImageExists(image string) (bool, error) {
	err := runtime.exec.Command("crictl", "-r", runtime.criSocket, "inspecti", image).Run()
	return err == nil, nil
}

// ImageExists checks to see if the image exists on the system
func (runtime *DockerRuntime) ImageExists(image string) (bool, error) {
	err := runtime.exec.Command("docker", "inspect", image).Run()
	return err == nil, nil
}
