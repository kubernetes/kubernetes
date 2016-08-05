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
)

func NewDockerSevice(client dockertools.DockerInterface) DockerLegacyService {
	return &dockerService{
		client: dockertools.NewInstrumentedDockerInterface(client),
	}
}

// DockerLegacyService is an interface that embeds both the new
// RuntimeService and ImageService interfaces, while including legacy methods
// for backward compatibility.
type DockerLegacyService interface {
	internalApi.RuntimeService
	internalApi.ImageManagerService

	// Supporting legacy methods for docker.
	GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error)
	kubecontainer.ContainerAttacher
	PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error
}

type dockerService struct {
	client dockertools.DockerInterface
}

// Version returns the runtime name, runtime version and runtime API version
func (ds *dockerService) Version(apiVersion string) (*runtimeApi.VersionResponse, error) {
	v, err := ds.client.Version()
	if err != nil {
		return nil, fmt.Errorf("docker: failed to get docker version: %v", err)
	}
	runtimeAPIVersion := kubeAPIVersion
	name := dockerRuntimeName
	return &runtimeApi.VersionResponse{
		Version:           &runtimeAPIVersion,
		RuntimeName:       &name,
		RuntimeVersion:    &v.Version,
		RuntimeApiVersion: &v.APIVersion,
	}, nil
}
