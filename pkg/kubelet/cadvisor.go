/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubelet

import (
	"errors"
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	cadvisorApi "github.com/google/cadvisor/info/v1"
)

var (
	// ErrNoKubeletContainers returned when there are not containers managed by the kubelet (ie: either no containers on the node, or none that the kubelet cares about).
	ErrNoKubeletContainers = errors.New("no containers managed by kubelet")

	// ErrContainerNotFound returned when a container in the given pod with the given container name was not found, amongst those managed by the kubelet.
	ErrContainerNotFound = errors.New("no matching container")

	// ErrCadvisorApiFailure returned when cadvisor couldn't retrieve stats for the given container, either because it isn't running or it was confused by the request
	ErrCadvisorApiFailure = errors.New("failed to retrieve cadvisor stats")
)

// cadvisorInterface is an abstract interface for testability.  It abstracts the interface of "github.com/google/cadvisor/client".Client.
type cadvisorInterface interface {
	DockerContainer(name string, req *cadvisorApi.ContainerInfoRequest) (cadvisorApi.ContainerInfo, error)
	ContainerInfo(name string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error)
	MachineInfo() (*cadvisorApi.MachineInfo, error)
}

// statsFromContainerPath takes a container's absolute path and returns the stats for the
// container. The container's absolute path refers to its hierarchy in the
// cgroup file system. e.g. The root container, which represents the whole
// machine, has path "/"; all docker containers have path "/docker/<docker id>"
func statsFromContainerPath(cc cadvisorInterface, containerPath string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	cinfo, err := cc.ContainerInfo(containerPath, req)
	if err != nil {
		return nil, err
	}
	return cinfo, nil
}

// statsFromDockerContainer takes a Docker container's ID and returns the stats for the
// container.
func statsFromDockerContainer(cc cadvisorInterface, containerId string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	cinfo, err := cc.DockerContainer(containerId, req)
	if err != nil {
		return nil, err
	}
	return &cinfo, nil
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, fmt.Errorf("no cadvisor connection")
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return nil, err
	}
	if len(dockerContainers) == 0 {
		return nil, ErrNoKubeletContainers
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uid, containerName)
	if !found {
		return nil, ErrContainerNotFound
	}

	ci, err := statsFromDockerContainer(cc, dockerContainer.ID, req)
	if err != nil {
		return nil, ErrCadvisorApiFailure
	}
	return ci, nil
}

// GetRootInfo returns stats (from Cadvisor) of current machine (root container).
func (kl *Kubelet) GetRootInfo(req *cadvisorApi.ContainerInfoRequest) (*cadvisorApi.ContainerInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, fmt.Errorf("no cadvisor connection")
	}
	return statsFromContainerPath(cc, "/", req)
}

func (kl *Kubelet) GetMachineInfo() (*cadvisorApi.MachineInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, fmt.Errorf("no cadvisor connection")
	}
	return cc.MachineInfo()
}
