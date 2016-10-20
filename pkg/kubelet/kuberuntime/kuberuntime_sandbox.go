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
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// createPodSandbox creates a pod sandbox and returns (podSandBoxID, message, error).
func (m *kubeGenericRuntimeManager) createPodSandbox(pod *api.Pod, attempt uint32) (string, string, error) {
	podSandboxConfig, err := m.generatePodSandboxConfig(pod, attempt)
	if err != nil {
		message := fmt.Sprintf("GeneratePodSandboxConfig for pod %q failed: %v", format.Pod(pod), err)
		glog.Error(message)
		return "", message, err
	}

	// Create pod logs directory
	err = m.osInterface.MkdirAll(podSandboxConfig.GetLogDirectory(), 0755)
	if err != nil {
		message := fmt.Sprintf("Create pod log directory for pod %q failed: %v", format.Pod(pod), err)
		glog.Errorf(message)
		return "", message, err
	}

	podSandBoxID, err := m.runtimeService.RunPodSandbox(podSandboxConfig)
	if err != nil {
		message := fmt.Sprintf("CreatePodSandbox for pod %q failed: %v", format.Pod(pod), err)
		glog.Error(message)
		return "", message, err
	}

	return podSandBoxID, "", nil
}

// generatePodSandboxConfig generates pod sandbox config from api.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(pod *api.Pod, attempt uint32) (*runtimeApi.PodSandboxConfig, error) {
	// TODO: deprecating podsandbox resource requirements in favor of the pod level cgroup
	// Refer https://github.com/kubernetes/kubernetes/issues/29871
	podUID := string(pod.UID)
	podSandboxConfig := &runtimeApi.PodSandboxConfig{
		Metadata: &runtimeApi.PodSandboxMetadata{
			Name:      &pod.Name,
			Namespace: &pod.Namespace,
			Uid:       &podUID,
			Attempt:   &attempt,
		},
		Labels:      newPodLabels(pod),
		Annotations: newPodAnnotations(pod),
	}

	if !kubecontainer.IsHostNetworkPod(pod) {
		dnsServers, dnsSearches, err := m.runtimeHelper.GetClusterDNS(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.DnsConfig = &runtimeApi.DNSConfig{
			Servers:  dnsServers,
			Searches: dnsSearches,
			Options:  defaultDNSOptions,
		}
		// TODO: Add domain support in new runtime interface
		hostname, _, err := m.runtimeHelper.GeneratePodHostNameAndDomain(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Hostname = &hostname
	}

	logDir := buildPodLogsDirectory(pod.UID)
	podSandboxConfig.LogDirectory = &logDir

	cgroupParent := ""
	portMappings := []*runtimeApi.PortMapping{}
	for _, c := range pod.Spec.Containers {
		// TODO: use a separate interface to only generate portmappings
		opts, err := m.runtimeHelper.GenerateRunContainerOptions(pod, &c, "")
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

// getKubeletSandboxes lists all (or just the running) sandboxes managed by kubelet.
func (m *kubeGenericRuntimeManager) getKubeletSandboxes(all bool) ([]*runtimeApi.PodSandbox, error) {
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
		if !isManagedByKubelet(s.Labels) {
			glog.V(5).Infof("Sandbox %s is not managed by kubelet", kubecontainer.BuildPodFullName(
				s.Metadata.GetName(), s.Metadata.GetNamespace()))
			continue
		}

		result = append(result, s)
	}

	return result, nil
}

// determinePodSandboxIP determines the IP address of the given pod sandbox.
// TODO: remove determinePodSandboxIP after networking is delegated to the container runtime.
func (m *kubeGenericRuntimeManager) determinePodSandboxIP(podNamespace, podName string, podSandbox *runtimeApi.PodSandboxStatus) string {
	ip := ""

	if podSandbox.Network != nil {
		ip = podSandbox.Network.GetIp()
	}

	if m.networkPlugin.Name() != network.DefaultPluginName {
		// TODO: podInfraContainerID in GetPodNetworkStatus() interface should be renamed to sandboxID
		netStatus, err := m.networkPlugin.GetPodNetworkStatus(podNamespace, podName, kubecontainer.ContainerID{
			Type: m.runtimeName,
			ID:   podSandbox.GetId(),
		})
		if err != nil {
			glog.Errorf("NetworkPlugin %s failed on the status hook for pod '%s' - %v", m.networkPlugin.Name(), kubecontainer.BuildPodFullName(podName, podNamespace), err)
		} else if netStatus != nil {
			ip = netStatus.IP.String()
		}
	}

	return ip
}

// getPodSandboxID gets the sandbox id by podUID and returns ([]sandboxID, error).
// Param state could be nil in order to get all sandboxes belonging to same pod.
func (m *kubeGenericRuntimeManager) getSandboxIDByPodUID(podUID string, state *runtimeApi.PodSandBoxState) ([]string, error) {
	filter := &runtimeApi.PodSandboxFilter{
		State:         state,
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: podUID},
	}
	sandboxes, err := m.runtimeService.ListPodSandbox(filter)
	if err != nil {
		glog.Errorf("ListPodSandbox with pod UID %q failed: %v", podUID, err)
		return nil, err
	}

	if len(sandboxes) == 0 {
		return nil, nil
	}

	// Sort with newest first.
	sandboxIDs := make([]string, len(sandboxes))
	sort.Sort(podSandboxByCreated(sandboxes))
	for i, s := range sandboxes {
		sandboxIDs[i] = s.GetId()
	}

	return sandboxIDs, nil
}
