// +build !linux,!windows

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

package cm

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/status"
	"k8s.io/kubernetes/pkg/util/mount"
)

type containerManagerImpl struct {
	devicePluginHandler DevicePluginHandler
}

var _ ContainerManager = &containerManagerImpl{}

func (containerManagerImpl) Start(_ *v1.Node, _ ActivePodsFunc, _ status.PodStatusProvider, _ internalapi.RuntimeService) error {
	return fmt.Errorf("Container Manager is unsupported in this build")
}

func (containerManagerImpl) SystemCgroupsLimit() v1.ResourceList {
	return v1.ResourceList{}
}

func (containerManagerImpl) GetNodeConfig() NodeConfig {
	return NodeConfig{}
}

func (containerManagerImpl) GetMountedSubsystems() *CgroupSubsystems {
	return &CgroupSubsystems{}
}

func (containerManagerImpl) GetQOSContainersInfo() QOSContainersInfo {
	return QOSContainersInfo{}
}

func (containerManagerImpl) UpdateQOSCgroups() error {
	return nil
}

func (cm *containerManagerImpl) Status() Status {
	return Status{}
}

func (cm *containerManagerImpl) GetNodeAllocatableReservation() v1.ResourceList {
	return nil
}

func (cm *containerManagerImpl) GetCapacity() v1.ResourceList {
	return nil
}

func (cm *containerManagerImpl) NewPodContainerManager() PodContainerManager {
	return &unsupportedPodContainerManager{}
}

func (cm *containerManagerImpl) GetResources(pod *v1.Pod, container *v1.Container, activePods []*v1.Pod) (*kubecontainer.RunContainerOptions, error) {
	return &kubecontainer.RunContainerOptions{}, nil
}

func (cm *containerManagerImpl) InternalContainerLifecycle() InternalContainerLifecycle {
	return &internalContainerLifecycleImpl{cpumanager.NewFakeManager()}
}

func NewContainerManager(_ mount.Interface, _ cadvisor.Interface, _ NodeConfig, failSwapOn bool, devicePluginEnabled bool, recorder record.EventRecorder) (ContainerManager, error) {
	return &containerManagerImpl{}, nil
}
