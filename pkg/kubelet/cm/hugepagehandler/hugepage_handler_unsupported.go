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

package hugepagehandler

import (
	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog"
)

type runtimeService interface {
	UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error
}

// Handler interface provides methods for Kubelet to set container hugetlb cgroup.
type Handler interface {
	// Start is called during Kubelet initialization.
	Start(containerRuntime runtimeService)

	// AddContainer is called between container create and container start
	// so that initial hugepage limit settings can be written through to the
	// container runtime before the first process begins to execute.
	AddContainer(p *v1.Pod, c *v1.Container, containerID string) error
}

type unsupportedHugepageHandler struct{}

func (m *unsupportedHugepageHandler) Start(_ runtimeService) {
	klog.Info("[fake hugepagehandler] Start()")
}

func (m *unsupportedHugepageHandler) AddContainer(pod *v1.Pod, container *v1.Container, containerID string) error {
	klog.Infof("[fake hugepagehandler] AddContainer (pod: %s, container: %s, container id: %s)", pod.Name, container.Name, containerID)
	return nil
}

// NewUnsupportedHugepageHandler create unsupoorted hugepage handler
func NewUnsupportedHugepageHandler() Handler {
	return &unsupportedHugepageHandler{}
}
