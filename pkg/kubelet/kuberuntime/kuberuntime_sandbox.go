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

package kuberuntime

import (
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// generatePodSandboxConfig generates pod sandbox config from api.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(pod *api.Pod, podIP string) (*runtimeApi.PodSandboxConfig, error) {
	sandboxName := buildSandboxName(pod)
	podSandboxConfig := &runtimeApi.PodSandboxConfig{
		Name:        &sandboxName,
		Labels:      newPodLabels(pod),
		Annotations: newPodAnnotations(pod),
	}

	if !kubecontainer.IsHostNetworkPod(pod) {
		dnsServers, dnsSearches, err := m.runtimeHelper.GetClusterDNS(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.DnsOptions = &runtimeApi.DNSOption{
			Servers:  dnsServers,
			Searches: dnsSearches,
		}
		// TODO: Add domain support in new runtime interface
		hostname, _, err := m.runtimeHelper.GeneratePodHostNameAndDomain(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Hostname = &hostname
	}

	cgroupParent := ""
	portMappings := []*runtimeApi.PortMapping{}
	for _, c := range pod.Spec.Containers {
		opts, err := m.runtimeHelper.GenerateRunContainerOptions(pod, &c, podIP)
		if err != nil {
			return nil, err
		}

		for _, port := range opts.PortMappings {
			hostPort := int32(port.HostPort)
			containerPort := int32(port.ContainerPort)
			protocol := toRuntimeProtocol(port.Protocol)
			portMappings = append(portMappings, &runtimeApi.PortMapping{
				HostIp:        &port.HostIP,
				HostPort:      &hostPort,
				ContainerPort: &containerPort,
				Protocol:      &protocol,
				Name:          &port.Name,
			})
		}

		cgroupParent = opts.CgroupParent
	}

	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	podSandboxConfig.Resources = generateSandboxResources(pod)
	podSandboxConfig.Linux = generatePodSandboxLinuxConfig(pod, cgroupParent)

	return podSandboxConfig, nil
}

// generateSandboxResources generates runtimeApi.PodSandboxResources from api.Pod
func generateSandboxResources(pod *api.Pod) *runtimeApi.PodSandboxResources {
	podResourceLimits := make(map[api.ResourceName]int64)
	podResourceRequests := make(map[api.ResourceName]int64)
	for _, c := range pod.Spec.Containers {
		for name, limit := range c.Resources.Limits {
			if l, ok := podResourceLimits[name]; ok {
				podResourceLimits[name] = l + limit.MilliValue()
			} else {
				podResourceLimits[name] = limit.MilliValue()
			}
		}
		for name, req := range c.Resources.Requests {
			if l, ok := podResourceRequests[name]; ok {
				podResourceRequests[name] = l + req.MilliValue()
			} else {
				podResourceRequests[name] = req.MilliValue()
			}
		}
	}

	var podCPULimit, podMemoryLimit, podCPURequest, podMemoryRequest int64
	for k, v := range podResourceLimits {
		switch k {
		case api.ResourceCPU:
			podCPULimit = v
		case api.ResourceMemory:
			podMemoryLimit = v
		}
	}
	for k, v := range podResourceRequests {
		switch k {
		case api.ResourceCPU:
			podCPURequest = v
		case api.ResourceMemory:
			podMemoryRequest = v
		}
	}

	cpuResource := &runtimeApi.ResourceRequirements{}
	cpulimit := float64(podCPULimit) / 1000
	cpuResource.Limits = &cpulimit

	cpuRequest := float64(podCPURequest) / 1000
	cpuResource.Requests = &cpuRequest

	memoryResource := &runtimeApi.ResourceRequirements{}
	memLimit := float64(podMemoryLimit) / 1000
	memoryResource.Limits = &memLimit

	memRequest := float64(podMemoryRequest) / 1000
	memoryResource.Requests = &memRequest

	return &runtimeApi.PodSandboxResources{
		Cpu:    cpuResource,
		Memory: memoryResource,
	}
}

// generatePodSandboxLinuxConfig generates LinuxPodSandboxConfig from api.Pod.
func generatePodSandboxLinuxConfig(pod *api.Pod, cgroupParent string) *runtimeApi.LinuxPodSandboxConfig {
	var linuxPodSandboxConfig *runtimeApi.LinuxPodSandboxConfig

	if pod.Spec.SecurityContext != nil {
		securityContext := pod.Spec.SecurityContext
		linuxPodSandboxConfig = &runtimeApi.LinuxPodSandboxConfig{
			NamespaceOptions: &runtimeApi.NamespaceOption{
				HostNetwork: &securityContext.HostNetwork,
				HostIpc:     &securityContext.HostIPC,
				HostPid:     &securityContext.HostPID,
			},
		}
	}

	if cgroupParent != "" {
		if linuxPodSandboxConfig != nil {
			linuxPodSandboxConfig.CgroupParent = &cgroupParent
		} else {
			linuxPodSandboxConfig = &runtimeApi.LinuxPodSandboxConfig{
				CgroupParent: &cgroupParent,
			}
		}
	}

	return linuxPodSandboxConfig
}
