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
	"fmt"
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// getKubeletPodSandboxes lists all (or just the running) pod sandboxes managed by kubelet.
func (m *kubeGenericRuntimeManager) getKubeletPodSandboxes(all bool) ([]*runtimeApi.PodSandbox, error) {
	var filter *runtimeApi.PodSandboxFilter
	if !all {
		readyState := runtimeApi.PodSandBoxState_READY
		filter = &runtimeApi.PodSandboxFilter{
			State: &readyState,
		}
	}

	resp, err := m.runtimeService.ListPodSandbox(filter)
	if err != nil {
		glog.Errorf("ListPodSandbox failed: %v", err)
		return nil, err
	}

	result := []*runtimeApi.PodSandbox{}
	for _, s := range resp {
		if len(s.GetName()) == 0 {
			continue
		}

		if !isPodSandBoxManagedByKubelet(s.GetName()) {
			glog.V(3).Infof("%s sandbox %s is not managed by kubelet", m.runtimeName, s.GetName())
			continue
		}

		result = append(result, s)
	}

	return result, nil
}

// generatePodSandboxConfig generates pod sandbox config from api.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(pod *api.Pod, podIP string) (*runtimeApi.PodSandboxConfig, error) {
	_, podName, _ := buildPodName(pod.Name, pod.Namespace, string(pod.UID))
	podSandboxConfig := &runtimeApi.PodSandboxConfig{
		Name:        &podName,
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
	podResourceLimits := make(map[api.ResourceName]int64)
	podResourceRequests := make(map[api.ResourceName]int64)
	portMappings := []*runtimeApi.PortMapping{}
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

	podSandboxConfig.Linux = generatePodSandboxLinuxConfig(pod, cgroupParent)

	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	if len(podResourceLimits) > 0 || len(podResourceRequests) > 0 {
		podSandboxConfig.Resources = &runtimeApi.PodSandboxResources{}
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
		podSandboxConfig.Resources.Cpu = cpuResource

		memoryResource := &runtimeApi.ResourceRequirements{}
		memLimit := float64(podMemoryLimit) / 1000
		memoryResource.Limits = &memLimit

		memRequest := float64(podMemoryRequest) / 1000
		memoryResource.Requests = &memRequest
		podSandboxConfig.Resources.Memory = memoryResource

	}

	return podSandboxConfig, nil
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

// createPodSandbox creates a pod sandbox and returns (podSandBoxID, message, error)
func (m *kubeGenericRuntimeManager) createPodSandbox(pod *api.Pod, podIP string) (string, string, error) {
	podSandboxConfig, err := m.generatePodSandboxConfig(pod, podIP)
	if err != nil {
		message := fmt.Sprintf("GeneratePodSandboxConfig for pod %q failed: %v", format.Pod(pod), err)
		glog.Error(message)
		return "", message, err
	}

	podSandBoxID, err := m.runtimeService.CreatePodSandbox(podSandboxConfig)
	if err != nil {
		message := fmt.Sprintf("CreatePodSandbox for pod %q failed: %v", format.Pod(pod), err)
		glog.Error(message)
		return "", message, err
	}

	return podSandBoxID, "", nil
}

// getPodSandboxID gets the sandbox id by podUID and returns (found, []sandboxID, error).
func (m *kubeGenericRuntimeManager) getPodSandboxID(podUID string, state *runtimeApi.PodSandBoxState) (bool, []string, error) {
	filter := &runtimeApi.PodSandboxFilter{
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: podUID},
	}
	if state != nil {
		filter.State = state
	}
	sandboxes, err := m.runtimeService.ListPodSandbox(filter)
	if err != nil {
		glog.Infof("ListPodSandbox failed: %v", err)
		return false, nil, err
	}
	if len(sandboxes) == 0 {
		// Not found
		return false, nil, nil
	}

	// Newest first.
	sandboxIDs := make([]string, len(sandboxes))
	sort.Sort(podSandboxByCreated(sandboxes))
	for idx, s := range sandboxes {
		sandboxIDs[idx] = s.GetId()
	}

	return true, sandboxIDs, nil
}

// isHostNetwork checks whether the pod is running in host-network mode.
func (m *kubeGenericRuntimeManager) isHostNetwork(podSandBoxID string, pod *api.Pod) (bool, error) {
	if pod != nil {
		return kubecontainer.IsHostNetworkPod(pod), nil
	}

	podStatus, err := m.runtimeService.PodSandboxStatus(podSandBoxID)
	if err != nil {
		return false, err
	}

	if podStatus.Linux != nil && podStatus.Linux.Namespaces != nil && podStatus.Linux.Namespaces.Options != nil {
		if podStatus.Linux.Namespaces.Options.HostNetwork != nil {
			return podStatus.Linux.Namespaces.Options.GetHostNetwork(), nil
		}
	}

	return false, nil
}
