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
	"fmt"
	"io"

	"k8s.io/kubernetes/pkg/api"
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util/term"
)

const (
	dockerRuntimeName = "docker"
	kubeAPIVersion    = "0.1.0"

	// String used to detect docker host mode for various namespaces (e.g.
	// networking). Must match the value returned by docker inspect -f
	// '{{.HostConfig.NetworkMode}}'.
	namespaceModeHost = "host"

	dockerNetNSFmt = "/proc/%v/ns/net"

	defaultSeccompProfile = "unconfined"

	// Internal docker labels used to identify whether a container is a sandbox
	// or a regular container.
	// TODO: This is not backward compatible with older containers. We will
	// need to add filtering based on names.
	containerTypeLabelKey       = "io.kubernetes.docker.type"
	containerTypeLabelSandbox   = "podsandbox"
	containerTypeLabelContainer = "container"
	containerLogPathLabelKey    = "io.kubernetes.container.logpath"
	sandboxIDLabelKey           = "io.kubernetes.sandbox.id"
)

var internalLabelKeys []string = []string{containerTypeLabelKey, containerLogPathLabelKey, sandboxIDLabelKey}

// NOTE: Anything passed to DockerService should be eventually handled in another way when we switch to running the shim as a different process.
func NewDockerService(client dockertools.DockerInterface, seccompProfileRoot string, podSandboxImage string) DockerService {
	return &dockerService{
		seccompProfileRoot: seccompProfileRoot,
		client:             dockertools.NewInstrumentedDockerInterface(client),
		os:                 kubecontainer.RealOS{},
		podSandboxImage:    podSandboxImage,
	}
}

// DockerService is an interface that embeds both the new RuntimeService and
// ImageService interfaces, while including DockerLegacyService for backward
// compatibility.
type DockerService interface {
	internalApi.RuntimeService
	internalApi.ImageManagerService
	DockerLegacyService
}

// DockerLegacyService is an interface that embeds all legacy methods for
// backward compatibility.
type DockerLegacyService interface {
	// Supporting legacy methods for docker.
	GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error)
	kubecontainer.ContainerAttacher
	PortForward(sandboxID string, port uint16, stream io.ReadWriteCloser) error

	// TODO: Remove this once exec is properly defined in CRI.
	ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error
}

type dockerService struct {
	seccompProfileRoot string
	client             dockertools.DockerInterface
	os                 kubecontainer.OSInterface
	podSandboxImage    string
}

// Version returns the runtime name, runtime version and runtime API version
func (ds *dockerService) Version(_ string) (*runtimeApi.VersionResponse, error) {
	v, err := ds.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}
	runtimeAPIVersion := kubeAPIVersion
	name := dockerRuntimeName
	// Docker API version (e.g., 1.23) is not semver compatible. Add a ".0"
	// suffix to remedy this.
	apiVersion := fmt.Sprintf("%s.0", v.APIVersion)
	return &runtimeApi.VersionResponse{
		Version:           &runtimeAPIVersion,
		RuntimeName:       &name,
		RuntimeVersion:    &v.Version,
		RuntimeApiVersion: &apiVersion,
	}, nil
}

func (ds *dockerService) UpdateRuntimeConfig(runtimeConfig *runtimeApi.RuntimeConfig) error {
	return nil
}
