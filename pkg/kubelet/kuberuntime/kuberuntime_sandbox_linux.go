//go:build linux
// +build linux

/*
Copyright 2021 The Kubernetes Authors.

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

package kuberuntime

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"

	resourcehelper "k8s.io/kubernetes/pkg/api/v1/resource"
)

func (m *kubeGenericRuntimeManager) convertOverheadToLinuxResources(pod *v1.Pod) *runtimeapi.LinuxContainerResources {
	resources := &runtimeapi.LinuxContainerResources{}
	if pod.Spec.Overhead != nil {
		cpu := pod.Spec.Overhead.Cpu()
		memory := pod.Spec.Overhead.Memory()

		// For overhead, we do not differentiate between requests and limits. Treat this overhead
		// as "guaranteed", with requests == limits
		resources = m.calculateLinuxResources(cpu, cpu, memory)
	}

	return resources
}

func (m *kubeGenericRuntimeManager) calculateSandboxResources(pod *v1.Pod) *runtimeapi.LinuxContainerResources {
	opts := resourcehelper.PodResourcesOptions{
		ExcludeOverhead: true,
	}
	req := resourcehelper.PodRequests(pod, opts)
	lim := resourcehelper.PodLimits(pod, opts)
	var cpuRequest *resource.Quantity
	if _, cpuRequestExists := req[v1.ResourceCPU]; cpuRequestExists {
		cpuRequest = req.Cpu()
	}
	return m.calculateLinuxResources(cpuRequest, lim.Cpu(), lim.Memory())
}

func (m *kubeGenericRuntimeManager) applySandboxResources(pod *v1.Pod, config *runtimeapi.PodSandboxConfig) error {

	if config.Linux == nil {
		return nil
	}
	config.Linux.Resources = m.calculateSandboxResources(pod)
	config.Linux.Overhead = m.convertOverheadToLinuxResources(pod)

	return nil
}
