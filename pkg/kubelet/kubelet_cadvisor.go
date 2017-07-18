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

package kubelet

import (
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// GetContainerInfo returns stats (from Cadvisor) for a container.
func (kl *Kubelet) GetContainerInfo(podFullName string, podUID types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error) {

	podUID = kl.podManager.TranslatePodUID(podUID)

	pods, err := kl.runtimeCache.GetPods()
	if err != nil {
		return nil, err
	}
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	container := pod.FindContainerByName(containerName)
	if container == nil {
		return nil, kubecontainer.ErrContainerNotFound
	}

	ci, err := kl.cadvisor.DockerContainer(container.ID.ID, req)
	if err != nil {
		return nil, err
	}
	return &ci, nil
}

// GetContainerInfoV2 returns stats (from Cadvisor) for containers.
func (kl *Kubelet) GetContainerInfoV2(name string, options cadvisorapiv2.RequestOptions) (map[string]cadvisorapiv2.ContainerInfo, error) {
	return kl.cadvisor.ContainerInfoV2(name, options)
}

// ImagesFsInfo returns information about docker image fs usage from
// cadvisor.
func (kl *Kubelet) ImagesFsInfo() (cadvisorapiv2.FsInfo, error) {
	return kl.cadvisor.ImagesFsInfo()
}

// RootFsInfo returns info about the root fs from cadvisor.
func (kl *Kubelet) RootFsInfo() (cadvisorapiv2.FsInfo, error) {
	return kl.cadvisor.RootFsInfo()
}

// Returns stats (from Cadvisor) for a non-Kubernetes container.
func (kl *Kubelet) GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error) {
	if subcontainers {
		return kl.cadvisor.SubcontainerInfo(containerName, req)
	} else {
		containerInfo, err := kl.cadvisor.ContainerInfo(containerName, req)
		if err != nil {
			return nil, err
		}
		return map[string]*cadvisorapi.ContainerInfo{
			containerInfo.Name: containerInfo,
		}, nil
	}
}

// GetVersionInfo returns information about the version of cAdvisor in use.
func (kl *Kubelet) GetVersionInfo() (*cadvisorapi.VersionInfo, error) {
	return kl.cadvisor.VersionInfo()
}

// GetCachedMachineInfo assumes that the machine info can't change without a reboot
func (kl *Kubelet) GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error) {
	if kl.machineInfo == nil {
		info, err := kl.cadvisor.MachineInfo()
		if err != nil {
			return nil, err
		}
		kl.machineInfo = info
	}
	return kl.machineInfo, nil
}

// GetCachedRootFsInfo assumes that the rootfs info can't change without a reboot
func (kl *Kubelet) GetCachedRootFsInfo() (cadvisorapiv2.FsInfo, error) {
	if kl.rootfsInfo == nil {
		info, err := kl.cadvisor.RootFsInfo()
		if err != nil {
			return cadvisorapiv2.FsInfo{}, err
		}
		kl.rootfsInfo = &info
	}
	return *kl.rootfsInfo, nil
}
