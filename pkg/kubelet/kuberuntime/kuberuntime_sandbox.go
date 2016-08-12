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
	// TODO: deprecating podsandbox resource requirements in favor of the pod level cgroup
	// Refer https://github.com/kubernetes/kubernetes/issues/29871
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

		for idx := range opts.PortMappings {
			port := opts.PortMappings[idx]
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

		// TODO: refactor kubelet to get cgroup parent for pod instead of containers
		cgroupParent = opts.CgroupParent
	}
	podSandboxConfig.Linux = generatePodSandboxLinuxConfig(pod, cgroupParent)
	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	return podSandboxConfig, nil
}

// generatePodSandboxLinuxConfig generates LinuxPodSandboxConfig from api.Pod.
func generatePodSandboxLinuxConfig(pod *api.Pod, cgroupParent string) *runtimeApi.LinuxPodSandboxConfig {
	if pod.Spec.SecurityContext == nil && cgroupParent == "" {
		return nil
	}

	linuxPodSandboxConfig := &runtimeApi.LinuxPodSandboxConfig{}
	if pod.Spec.SecurityContext != nil {
		securityContext := pod.Spec.SecurityContext
		linuxPodSandboxConfig.NamespaceOptions = &runtimeApi.NamespaceOption{
			HostNetwork: &securityContext.HostNetwork,
			HostIpc:     &securityContext.HostIPC,
			HostPid:     &securityContext.HostPID,
		}
	}

	if cgroupParent != "" {
		linuxPodSandboxConfig.CgroupParent = &cgroupParent
	}

	return linuxPodSandboxConfig
}
