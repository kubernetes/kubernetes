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
	"net"
	"net/url"
	"sort"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	kubetypes "k8s.io/kubernetes/pkg/types"
)

// createPodSandbox creates a pod sandbox and returns (podSandBoxID, message, error).
func (m *kubeGenericRuntimeManager) createPodSandbox(pod *v1.Pod, attempt uint32) (string, string, error) {
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

// generatePodSandboxConfig generates pod sandbox config from v1.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(pod *v1.Pod, attempt uint32) (*runtimeapi.PodSandboxConfig, error) {
	// TODO: deprecating podsandbox resource requirements in favor of the pod level cgroup
	// Refer https://github.com/kubernetes/kubernetes/issues/29871
	podUID := string(pod.UID)
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
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
		podSandboxConfig.DnsConfig = &runtimeapi.DNSConfig{
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
	portMappings := []*runtimeapi.PortMapping{}
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
			portMappings = append(portMappings, &runtimeapi.PortMapping{
				HostIp:        &port.HostIP,
				HostPort:      &hostPort,
				ContainerPort: &containerPort,
				Protocol:      &protocol,
			})
		}

		// TODO: refactor kubelet to get cgroup parent for pod instead of containers
		cgroupParent = opts.CgroupParent
	}
	podSandboxConfig.Linux = m.generatePodSandboxLinuxConfig(pod, cgroupParent)
	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	return podSandboxConfig, nil
}

// generatePodSandboxLinuxConfig generates LinuxPodSandboxConfig from v1.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxLinuxConfig(pod *v1.Pod, cgroupParent string) *runtimeapi.LinuxPodSandboxConfig {
	if pod.Spec.SecurityContext == nil && cgroupParent == "" {
		return nil
	}

	lc := &runtimeapi.LinuxPodSandboxConfig{}
	if cgroupParent != "" {
		lc.CgroupParent = &cgroupParent
	}
	if pod.Spec.SecurityContext != nil {
		sc := pod.Spec.SecurityContext
		lc.SecurityContext = &runtimeapi.LinuxSandboxSecurityContext{
			NamespaceOptions: &runtimeapi.NamespaceOption{
				HostNetwork: &pod.Spec.HostNetwork,
				HostIpc:     &pod.Spec.HostIPC,
				HostPid:     &pod.Spec.HostPID,
			},
			RunAsUser: sc.RunAsUser,
		}

		if sc.FSGroup != nil {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, *sc.FSGroup)
		}
		if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, groups...)
		}
		if sc.SupplementalGroups != nil {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, sc.SupplementalGroups...)
		}
		if sc.SELinuxOptions != nil {
			lc.SecurityContext.SelinuxOptions = &runtimeapi.SELinuxOption{
				User:  &sc.SELinuxOptions.User,
				Role:  &sc.SELinuxOptions.Role,
				Type:  &sc.SELinuxOptions.Type,
				Level: &sc.SELinuxOptions.Level,
			}
		}
	}

	return lc
}

// getKubeletSandboxes lists all (or just the running) sandboxes managed by kubelet.
func (m *kubeGenericRuntimeManager) getKubeletSandboxes(all bool) ([]*runtimeapi.PodSandbox, error) {
	var filter *runtimeapi.PodSandboxFilter
	if !all {
		readyState := runtimeapi.PodSandboxState_SANDBOX_READY
		filter = &runtimeapi.PodSandboxFilter{
			State: &readyState,
		}
	}

	resp, err := m.runtimeService.ListPodSandbox(filter)
	if err != nil {
		glog.Errorf("ListPodSandbox failed: %v", err)
		return nil, err
	}

	result := []*runtimeapi.PodSandbox{}
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
func (m *kubeGenericRuntimeManager) determinePodSandboxIP(podNamespace, podName string, podSandbox *runtimeapi.PodSandboxStatus) string {
	if podSandbox.Network == nil {
		glog.Warningf("Pod Sandbox status doesn't have network information, cannot report IP")
		return ""
	}
	ip := podSandbox.Network.GetIp()
	if net.ParseIP(ip) == nil {
		glog.Warningf("Pod Sandbox reported an unparseable IP %v", ip)
		return ""
	}
	return ip
}

// getPodSandboxID gets the sandbox id by podUID and returns ([]sandboxID, error).
// Param state could be nil in order to get all sandboxes belonging to same pod.
func (m *kubeGenericRuntimeManager) getSandboxIDByPodUID(podUID kubetypes.UID, state *runtimeapi.PodSandboxState) ([]string, error) {
	filter := &runtimeapi.PodSandboxFilter{
		State:         state,
		LabelSelector: map[string]string{types.KubernetesPodUIDLabel: string(podUID)},
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

// GetPortForward gets the endpoint the runtime will serve the port-forward request from.
func (m *kubeGenericRuntimeManager) GetPortForward(podName, podNamespace string, podUID kubetypes.UID) (*url.URL, error) {
	sandboxIDs, err := m.getSandboxIDByPodUID(podUID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s: %v", format.PodDesc(podName, podNamespace, podUID), err)
	}
	if len(sandboxIDs) == 0 {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s", format.PodDesc(podName, podNamespace, podUID))
	}
	// TODO: Port is unused for now, but we may need it in the future.
	req := &runtimeapi.PortForwardRequest{
		PodSandboxId: &sandboxIDs[0],
	}
	resp, err := m.runtimeService.PortForward(req)
	if err != nil {
		return nil, err
	}
	return url.Parse(resp.GetUrl())
}
