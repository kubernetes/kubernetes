/*
Copyright 2017 The Kubernetes Authors.

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
	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

type InternalContainerLifecycle interface {
	PreCreateContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerConfig *runtimeapi.ContainerConfig) error
	PreStartContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerID string) error
	PostStopContainer(logger klog.Logger, containerID string) error
}

// Implements InternalContainerLifecycle interface.
type internalContainerLifecycleImpl struct {
	cpuManager      cpumanager.Manager
	memoryManager   memorymanager.Manager
	topologyManager topologymanager.Manager
}

func (i *internalContainerLifecycleImpl) PreStartContainer(logger klog.Logger, pod *v1.Pod, container *v1.Container, containerID string) error {
	if i.cpuManager != nil {
		i.cpuManager.AddContainer(logger, pod, container, containerID)
	}

	if i.memoryManager != nil {
		i.memoryManager.AddContainer(logger, pod, container, containerID)
	}

	i.topologyManager.AddContainer(pod, container, containerID)

	return nil
}

func (i *internalContainerLifecycleImpl) PostStopContainer(logger klog.Logger, containerID string) error {
	return i.topologyManager.RemoveContainer(containerID)
}
