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
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	cadvisor "github.com/google/cadvisor/info"
)

// cadvisorInterface is an abstract interface for testability.  It abstracts the interface of "github.com/google/cadvisor/client".Client.
type cadvisorInterface interface {
	ContainerInfo(name string, req *cadvisor.ContainerInfoRequest) (*cadvisor.ContainerInfo, error)
	MachineInfo() (*cadvisor.MachineInfo, error)
}

// This method takes a container's absolute path and returns the stats for the
// container.  The container's absolute path refers to its hierarchy in the
// cgroup file system. e.g. The root container, which represents the whole
// machine, has path "/"; all docker containers have path "/docker/<docker id>"
func (kl *Kubelet) statsFromContainerPath(cc cadvisorInterface, containerPath string, req *cadvisor.ContainerInfoRequest) (*cadvisor.ContainerInfo, error) {
	cinfo, err := cc.ContainerInfo(containerPath, req)
	if err != nil {
		return nil, err
	}
	return cinfo, nil
}

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName, uuid, containerName string, req *cadvisor.ContainerInfoRequest) (*cadvisor.ContainerInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, nil
	}
	dockerContainers, err := dockertools.GetKubeletDockerContainers(kl.dockerClient, false)
	if err != nil {
		return nil, err
	}
	dockerContainer, found, _ := dockerContainers.FindPodContainer(podFullName, uuid, containerName)
	if !found {
		return nil, fmt.Errorf("couldn't find container")
	}
	return kl.statsFromContainerPath(cc, fmt.Sprintf("/docker/%s", dockerContainer.ID), req)
}

// GetRootInfo returns stats (from Cadvisor) of current machine (root container).
func (kl *Kubelet) GetRootInfo(req *cadvisor.ContainerInfoRequest) (*cadvisor.ContainerInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, fmt.Errorf("no cadvisor connection")
	}
	return kl.statsFromContainerPath(cc, "/", req)
}

func (kl *Kubelet) GetMachineInfo() (*cadvisor.MachineInfo, error) {
	cc := kl.GetCadvisorClient()
	if cc == nil {
		return nil, fmt.Errorf("no cadvisor connection")
	}
	return cc.MachineInfo()
}
