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
	"context"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"runtime"
	"sort"

	v1 "k8s.io/api/core/v1"
	kubetypes "k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	netutils "k8s.io/utils/net"
)

// createPodSandbox creates a pod sandbox and returns (podSandBoxID, message, error).
func (m *kubeGenericRuntimeManager) createPodSandbox(ctx context.Context, pod *v1.Pod, attempt uint32) (string, string, error) {
	logger := klog.FromContext(ctx)
	podSandboxConfig, err := m.generatePodSandboxConfig(ctx, pod, attempt)
	if err != nil {
		message := fmt.Sprintf("Failed to generate sandbox config for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to generate sandbox config for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	// Create pod logs directory
	err = m.osInterface.MkdirAll(podSandboxConfig.LogDirectory, 0755)
	if err != nil {
		message := fmt.Sprintf("Failed to create log directory for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to create log directory for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	runtimeHandler := ""
	if m.runtimeClassManager != nil {
		runtimeHandler, err = m.runtimeClassManager.LookupRuntimeHandler(pod.Spec.RuntimeClassName)
		if err != nil {
			message := fmt.Sprintf("Failed to create sandbox for pod %q: %v", format.Pod(pod), err)
			return "", message, err
		}
		if runtimeHandler != "" {
			logger.V(2).Info("Running pod with runtime handler", "pod", klog.KObj(pod), "runtimeHandler", runtimeHandler)
		}
	}

	podSandBoxID, err := m.runtimeService.RunPodSandbox(ctx, podSandboxConfig, runtimeHandler)
	if err != nil {
		message := fmt.Sprintf("Failed to create sandbox for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to create sandbox for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	return podSandBoxID, "", nil
}

// acquireRestore records that a sandbox restore is starting for the given pod
// (keyed by namespace/name). It returns false if a restore for that pod is
// already in flight, so the caller can reject the second attempt instead of
// racing the first. It is keyed by namespace/name rather than UID because each
// restore attempt is admitted under a fresh pod UID, so a UID key would never
// collide.
func (m *kubeGenericRuntimeManager) acquireRestore(key string) bool {
	m.restoresInFlightLock.Lock()
	defer m.restoresInFlightLock.Unlock()
	if _, inFlight := m.restoresInFlight[key]; inFlight {
		return false
	}
	m.restoresInFlight[key] = struct{}{}
	return true
}

// releaseRestore clears the in-flight restore marker for the given pod key.
func (m *kubeGenericRuntimeManager) releaseRestore(key string) {
	m.restoresInFlightLock.Lock()
	defer m.restoresInFlightLock.Unlock()
	delete(m.restoresInFlight, key)
}

// restorePodSandbox restores a pod sandbox from a checkpoint and returns (podSandBoxID, message, error).
//
// IMPORTANT HARDCODED BEHAVIOR:
// Currently this function hardcodes pod-level PID namespace sharing (requires infra container).
// This is a temporary workaround required by CRI-O for pod restore functionality.
// TODO: The checkpoint archive should include complete pod metadata (ShareProcessNamespace,
// NamespaceOptions, SecurityContext, etc.) so these settings can be restored from the
// checkpoint instead of being hardcoded here.
func (m *kubeGenericRuntimeManager) restorePodSandbox(ctx context.Context, pod *v1.Pod, attempt uint32) (string, string, error) {
	logger := klog.FromContext(ctx)

	if pod.Spec.RestoreFrom == nil || *pod.Spec.RestoreFrom == "" {
		message := "RestoreFrom is not specified in pod spec"
		logger.Error(nil, message, "pod", klog.KObj(pod))
		return "", message, errors.New(message)
	}

	// Gate concurrent restores of the same pod. The CRI RestorePod call below
	// can be long-running, so reject a second restore for this pod while one is
	// already in flight rather than racing two restores into the same pod
	// sandbox. Keyed by namespace/name because each restore attempt is admitted
	// under a fresh pod UID.
	restoreKey := pod.Namespace + "/" + pod.Name
	if !m.acquireRestore(restoreKey) {
		// Another restore for the same (namespace, name) holds the lock. Record
		// the blocked state so the kubelet surfaces Restoring=False/RestoreInProgress
		// on this pod while it waits and retries (KEP-5823).
		m.runtimeHelper.SetPodRestoreBlocked(pod.UID, true)
		message := fmt.Sprintf("A restore is already in progress for pod %q", format.Pod(pod))
		logger.Info(message, "pod", klog.KObj(pod), "podUID", pod.UID)
		return "", message, &kubecontainer.RestoreError{Reason: events.RestoreInProgress, Err: fmt.Errorf("restore already in progress for pod %s", format.Pod(pod))}
	}
	defer m.releaseRestore(restoreKey)
	// This pod now holds the restore lock, so it is no longer blocked: its
	// Restoring condition flips from False/RestoreInProgress to True.
	m.runtimeHelper.SetPodRestoreBlocked(pod.UID, false)

	// Resolve spec.restoreFrom (the name of a PodCheckpoint in the pod's
	// namespace) to the on-node checkpoint archive path.
	archivePath, err := m.runtimeHelper.GetPodCheckpointArchivePath(ctx, pod)
	if err != nil {
		message := fmt.Sprintf("Failed to resolve checkpoint for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to resolve checkpoint archive for restore", "pod", klog.KObj(pod), "restoreFrom", *pod.Spec.RestoreFrom)
		return "", message, err
	}

	podSandboxConfig, err := m.generatePodSandboxConfig(ctx, pod, attempt)
	if err != nil {
		message := fmt.Sprintf("Failed to generate sandbox config for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to generate sandbox config for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	// Create pod logs directory
	err = m.osInterface.MkdirAll(podSandboxConfig.LogDirectory, 0755)
	if err != nil {
		message := fmt.Sprintf("Failed to create log directory for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to create log directory for pod", "pod", klog.KObj(pod))
		return "", message, err
	}

	runtimeHandler := ""
	if m.runtimeClassManager != nil {
		runtimeHandler, err = m.runtimeClassManager.LookupRuntimeHandler(pod.Spec.RuntimeClassName)
		if err != nil {
			message := fmt.Sprintf("Failed to restore sandbox for pod %q: %v", format.Pod(pod), err)
			return "", message, err
		}
		if runtimeHandler != "" {
			logger.V(2).Info("Restoring pod with runtime handler", "pod", klog.KObj(pod), "runtimeHandler", runtimeHandler)
		}
	}

	logger.V(2).Info("Restoring pod sandbox from checkpoint", "pod", klog.KObj(pod), "checkpointName", *pod.Spec.RestoreFrom, "checkpointPath", archivePath, "runtimeHandler", runtimeHandler)

	// The sandbox config (including namespace options and security context) is
	// derived from the pod spec by generatePodSandboxConfig, exactly as for
	// createPodSandbox. Because the restoring pod's spec is validated to match
	// the checkpoint, its ShareProcessNamespace/HostPID settings reproduce the
	// process-namespace topology that was captured. We deliberately do not
	// override the PID namespace here.

	// Generate container configs with mount information for all containers
	// This tells the CRI runtime where to mount host paths into the restored containers
	containerConfigs, err := m.generateContainerConfigsForRestore(ctx, pod, podSandboxConfig)
	if err != nil {
		message := fmt.Sprintf("Failed to generate container configs for restore for pod %q: %v", format.Pod(pod), err)
		logger.Error(err, "Failed to generate container configs for restore", "pod", klog.KObj(pod))
		return "", message, err
	}

	logger.V(1).Info("Generated container configs for restore", "pod", klog.KObj(pod), "containerCount", len(containerConfigs))

	// Create the RestorePodRequest
	// Note: RuntimeHandler is not part of RestorePodRequest; it's stored in the checkpoint
	restoreRequest := &runtimeapi.RestorePodRequest{
		Path:             archivePath,
		Config:           podSandboxConfig,
		ContainerConfigs: containerConfigs,
	}

	// Call the CRI RestorePod RPC
	podSandBoxID, err := m.runtimeService.RestorePod(ctx, restoreRequest)
	if err != nil {
		message := fmt.Sprintf("Failed to restore pod sandbox from checkpoint %q for pod %q: %v", archivePath, format.Pod(pod), err)
		logger.Error(err, "Failed to restore pod sandbox from checkpoint", "pod", klog.KObj(pod), "checkpointPath", archivePath)
		return "", message, err
	}

	if podSandBoxID == "" {
		message := fmt.Sprintf("RestorePod returned empty pod sandbox ID for pod %q", format.Pod(pod))
		logger.Error(nil, message, "pod", klog.KObj(pod))
		return "", message, errors.New(message)
	}

	logger.V(2).Info("Successfully restored pod sandbox from checkpoint", "pod", klog.KObj(pod), "podSandboxID", podSandBoxID, "checkpointName", *pod.Spec.RestoreFrom, "checkpointPath", archivePath)

	return podSandBoxID, "", nil
}

// generateContainerConfigsForRestore generates minimal container configurations for pod restore.
// These configs contain mount information that tells the CRI runtime where to mount host paths
// (e.g., /etc/hosts, termination logs, volumes) into the restored containers.
// The CRI runtime matches these configs with containers from the checkpoint by container name.
func (m *kubeGenericRuntimeManager) generateContainerConfigsForRestore(ctx context.Context, pod *v1.Pod, podSandboxConfig *runtimeapi.PodSandboxConfig) ([]*runtimeapi.ContainerConfig, error) {
	logger := klog.FromContext(ctx)

	// Collect all containers to restore (init + regular). Ephemeral containers
	// are debugging containers that are not part of the checkpointed application
	// and are out of scope for restore, so they are excluded.
	allContainers := append([]v1.Container{}, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)

	containerConfigs := make([]*runtimeapi.ContainerConfig, 0, len(allContainers))

	// Get pod IPs from the sandbox config for /etc/hosts generation
	podIP := ""
	podIPs := []string{}
	if podSandboxConfig.GetLabels() != nil {
		if ip, ok := podSandboxConfig.GetLabels()["io.kubernetes.pod.ip"]; ok && ip != "" {
			podIP = ip
			podIPs = append(podIPs, ip)
		}
	}
	// Also check pod status for IPs
	if pod.Status.PodIP != "" {
		podIP = pod.Status.PodIP
		podIPs = []string{pod.Status.PodIP}
	}
	for _, ip := range pod.Status.PodIPs {
		if ip.IP != "" && ip.IP != podIP {
			podIPs = append(podIPs, ip.IP)
		}
	}

	logger.V(1).Info("Pod IPs for restore", "pod", klog.KObj(pod), "podIP", podIP, "podIPs", podIPs)

	// Generate container config with mounts for each container
	for _, container := range allContainers {
		// Generate run container options which includes all mount information
		// Pass pod IPs so that /etc/hosts mount can be generated
		opts, _, err := m.runtimeHelper.GenerateRunContainerOptions(ctx, pod, &container, podIP, podIPs, nil)
		if err != nil {
			return nil, fmt.Errorf("failed to generate container options for %s: %w", container.Name, err)
		}

		logger.V(1).Info("Generated RunContainerOptions for restore", "pod", klog.KObj(pod), "containerName", container.Name, "kubeletMountCount", len(opts.Mounts))
		for i, mount := range opts.Mounts {
			logger.V(1).Info("Kubelet mount before CRI conversion", "pod", klog.KObj(pod), "containerName", container.Name, "mountIndex", i, "name", mount.Name, "containerPath", mount.ContainerPath, "hostPath", mount.HostPath)
		}

		// Convert kubelet mounts to CRI mounts
		mounts := m.makeMounts(opts, &container)

		// Manually add /etc/hosts mount since GenerateRunContainerOptions won't include it
		// without pod IPs (which aren't available yet during restore)
		podDir := m.runtimeHelper.GetPodDir(pod.UID)
		etcHostsPath := filepath.Join(podDir, "etc-hosts")
		hostsMount := &runtimeapi.Mount{
			HostPath:       etcHostsPath,
			ContainerPath:  "/etc/hosts",
			Readonly:       false,
			SelinuxRelabel: true,
		}
		mounts = append(mounts, hostsMount)
		logger.V(1).Info("Added /etc/hosts mount for restore", "pod", klog.KObj(pod), "containerName", container.Name, "hostPath", etcHostsPath)

		// Create minimal container config with just the information needed for restore
		// Include labels and annotations so kubelet can identify the containers
		labels := newContainerLabels(&container, pod)
		annotations := newContainerAnnotations(ctx, &container, pod, 0, opts)
		config := &runtimeapi.ContainerConfig{
			Metadata: &runtimeapi.ContainerMetadata{
				Name:    container.Name,
				Attempt: 0, // Restored containers start at attempt 0
			},
			Labels:      labels,
			Annotations: annotations,
			Mounts:      mounts,
		}

		logger.V(1).Info("Generated container config for restore", "pod", klog.KObj(pod), "containerName", container.Name, "mountCount", len(mounts), "labelCount", len(labels), "annotationCount", len(annotations))
		for i, mount := range mounts {
			logger.V(1).Info("Container mount for restore", "pod", klog.KObj(pod), "containerName", container.Name, "mountIndex", i, "containerPath", mount.ContainerPath, "hostPath", mount.HostPath, "readonly", mount.Readonly)
		}

		containerConfigs = append(containerConfigs, config)
	}

	return containerConfigs, nil
}

// generatePodSandboxConfig generates pod sandbox config from v1.Pod.
func (m *kubeGenericRuntimeManager) generatePodSandboxConfig(ctx context.Context, pod *v1.Pod, attempt uint32) (*runtimeapi.PodSandboxConfig, error) {
	// TODO: deprecating podsandbox resource requirements in favor of the pod level cgroup
	// Refer https://github.com/kubernetes/kubernetes/issues/29871
	logger := klog.FromContext(ctx)
	podUID := string(pod.UID)
	podSandboxConfig := &runtimeapi.PodSandboxConfig{
		Metadata: &runtimeapi.PodSandboxMetadata{
			Name:      pod.Name,
			Namespace: pod.Namespace,
			Uid:       podUID,
			Attempt:   attempt,
		},
		Labels:      newPodLabels(pod),
		Annotations: newPodAnnotations(pod),
	}

	dnsConfig, err := m.runtimeHelper.GetPodDNS(ctx, pod)
	if err != nil {
		return nil, err
	}
	podSandboxConfig.DnsConfig = dnsConfig

	if !kubecontainer.IsHostNetworkPod(pod) {
		// TODO: Add domain support in new runtime interface
		podHostname, podDomain, err := m.runtimeHelper.GeneratePodHostNameAndDomain(logger, pod)
		if err != nil {
			return nil, err
		}
		podHostname, err = util.GetNodenameForKernel(podHostname, podDomain, pod.Spec.SetHostnameAsFQDN)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Hostname = podHostname
	}

	logDir := BuildPodLogsDirectory(m.podLogsDirectory, pod.Namespace, pod.Name, pod.UID)
	podSandboxConfig.LogDirectory = logDir

	portMappings := []*runtimeapi.PortMapping{}
	for _, c := range pod.Spec.Containers {
		containerPortMappings := kubecontainer.MakePortMappings(logger, &c)

		for idx := range containerPortMappings {
			port := containerPortMappings[idx]
			hostPort := int32(port.HostPort)
			containerPort := int32(port.ContainerPort)
			protocol := toRuntimeProtocol(logger, port.Protocol)
			portMappings = append(portMappings, &runtimeapi.PortMapping{
				HostIp:        port.HostIP,
				HostPort:      hostPort,
				ContainerPort: containerPort,
				Protocol:      protocol,
			})
		}

	}
	if len(portMappings) > 0 {
		podSandboxConfig.PortMappings = portMappings
	}

	lc, err := m.generatePodSandboxLinuxConfig(ctx, pod)
	if err != nil {
		return nil, err
	}
	podSandboxConfig.Linux = lc

	if runtime.GOOS == "windows" {
		wc, err := m.generatePodSandboxWindowsConfig(pod)
		if err != nil {
			return nil, err
		}
		podSandboxConfig.Windows = wc
	}

	// Update config to include overhead, sandbox level resources
	if err := m.applySandboxResources(ctx, pod, podSandboxConfig); err != nil {
		return nil, err
	}
	return podSandboxConfig, nil
}

// generatePodSandboxLinuxConfig generates LinuxPodSandboxConfig from v1.Pod.
// We've to call PodSandboxLinuxConfig always irrespective of the underlying OS as securityContext is not part of
// podSandboxConfig. It is currently part of LinuxPodSandboxConfig. In future, if we have securityContext pulled out
// in podSandboxConfig we should be able to use it.
func (m *kubeGenericRuntimeManager) generatePodSandboxLinuxConfig(ctx context.Context, pod *v1.Pod) (*runtimeapi.LinuxPodSandboxConfig, error) {
	cgroupParent := m.runtimeHelper.GetPodCgroupParent(pod)
	lc := &runtimeapi.LinuxPodSandboxConfig{
		CgroupParent: cgroupParent,
		SecurityContext: &runtimeapi.LinuxSandboxSecurityContext{
			Privileged: kubecontainer.HasPrivilegedContainer(pod),

			// Forcing sandbox to run as `runtime/default` allow users to
			// use least privileged seccomp profiles at pod level. Issue #84623
			Seccomp: &runtimeapi.SecurityProfile{
				ProfileType: runtimeapi.SecurityProfile_RuntimeDefault,
			},
		},
	}

	sysctls := make(map[string]string)
	if pod.Spec.SecurityContext != nil {
		for _, c := range pod.Spec.SecurityContext.Sysctls {
			sysctls[c.Name] = c.Value
		}
	}

	lc.Sysctls = sysctls

	if pod.Spec.SecurityContext != nil {
		sc := pod.Spec.SecurityContext
		if sc.RunAsUser != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.RunAsUser = &runtimeapi.Int64Value{Value: int64(*sc.RunAsUser)}
		}
		if sc.RunAsGroup != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.RunAsGroup = &runtimeapi.Int64Value{Value: int64(*sc.RunAsGroup)}
		}
		namespaceOptions, err := runtimeutil.NamespacesForPod(ctx, pod, m.runtimeHelper, m.runtimeClassManager)
		if err != nil {
			return nil, err
		}
		lc.SecurityContext.NamespaceOptions = namespaceOptions

		if sc.FSGroup != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, int64(*sc.FSGroup))
		}
		if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
			lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, groups...)
		}
		if sc.SupplementalGroups != nil {
			for _, sg := range sc.SupplementalGroups {
				lc.SecurityContext.SupplementalGroups = append(lc.SecurityContext.SupplementalGroups, int64(sg))
			}
		}
		if sc.SupplementalGroupsPolicy != nil {
			policyValue, ok := runtimeapi.SupplementalGroupsPolicy_value[string(*sc.SupplementalGroupsPolicy)]
			if !ok {
				return nil, fmt.Errorf("unsupported supplementalGroupsPolicy: %s", string(*sc.SupplementalGroupsPolicy))
			}
			lc.SecurityContext.SupplementalGroupsPolicy = runtimeapi.SupplementalGroupsPolicy(policyValue)
		}

		if sc.SELinuxOptions != nil && runtime.GOOS != "windows" {
			lc.SecurityContext.SelinuxOptions = &runtimeapi.SELinuxOption{
				User:  sc.SELinuxOptions.User,
				Role:  sc.SELinuxOptions.Role,
				Type:  sc.SELinuxOptions.Type,
				Level: sc.SELinuxOptions.Level,
			}
		}
	}

	return lc, nil
}

// generatePodSandboxWindowsConfig generates WindowsPodSandboxConfig from v1.Pod.
// On Windows this will get called in addition to LinuxPodSandboxConfig because not all relevant fields have been added to
// WindowsPodSandboxConfig at this time.
func (m *kubeGenericRuntimeManager) generatePodSandboxWindowsConfig(pod *v1.Pod) (*runtimeapi.WindowsPodSandboxConfig, error) {
	wc := &runtimeapi.WindowsPodSandboxConfig{
		SecurityContext: &runtimeapi.WindowsSandboxSecurityContext{},
	}

	// If all of the containers in a pod are HostProcess containers, set the pod's HostProcess field
	// explicitly because the container runtime requires this information at sandbox creation time.
	if kubecontainer.HasWindowsHostProcessContainer(pod) {
		// At present Windows all containers in a Windows pod must be HostProcess containers
		// and HostNetwork is required to be set.
		if !kubecontainer.AllContainersAreWindowsHostProcess(pod) {
			return nil, fmt.Errorf("pod must not contain both HostProcess and non-HostProcess containers")
		}

		if !kubecontainer.IsHostNetworkPod(pod) {
			return nil, fmt.Errorf("hostNetwork is required if Pod contains HostProcess containers")
		}

		wc.SecurityContext.HostProcess = true
	}

	sc := pod.Spec.SecurityContext
	if sc == nil || sc.WindowsOptions == nil {
		return wc, nil
	}

	wo := sc.WindowsOptions
	if wo.GMSACredentialSpec != nil {
		wc.SecurityContext.CredentialSpec = *wo.GMSACredentialSpec
	}

	if wo.RunAsUserName != nil {
		wc.SecurityContext.RunAsUsername = *wo.RunAsUserName
	}

	if kubecontainer.HasWindowsHostProcessContainer(pod) {

		if wo.HostProcess != nil && !*wo.HostProcess {
			return nil, fmt.Errorf("pod must not contain any HostProcess containers if Pod's WindowsOptions.HostProcess is set to false")
		}
	}

	return wc, nil
}

// determinePodSandboxIP determines the IP addresses of the given pod sandbox.
func (m *kubeGenericRuntimeManager) determinePodSandboxIPs(ctx context.Context, podNamespace, podName string, podSandbox *runtimeapi.PodSandboxStatus) []string {
	logger := klog.FromContext(ctx)
	podIPs := make([]string, 0)
	if podSandbox.Network == nil {
		logger.Info("Pod Sandbox status doesn't have network information, cannot report IPs", "pod", klog.KRef(podNamespace, podName))
		return podIPs
	}

	// ip could be an empty string if runtime is not responsible for the
	// IP (e.g., host networking).

	// pick primary IP
	if len(podSandbox.Network.Ip) != 0 {
		if netutils.ParseIPSloppy(podSandbox.Network.Ip) == nil {
			logger.Info("Pod Sandbox reported an unparseable primary IP", "pod", klog.KRef(podNamespace, podName), "IP", podSandbox.Network.Ip)
			return nil
		}
		podIPs = append(podIPs, podSandbox.Network.Ip)
	}

	// pick additional ips, if cri reported them
	for _, podIP := range podSandbox.Network.AdditionalIps {
		if nil == netutils.ParseIPSloppy(podIP.Ip) {
			logger.Info("Pod Sandbox reported an unparseable additional IP", "pod", klog.KRef(podNamespace, podName), "IP", podIP.Ip)
			return nil
		}
		podIPs = append(podIPs, podIP.Ip)
	}

	return podIPs
}

// getPodSandboxIDByPodUID gets the sandbox id by podUID and returns ([]sandboxID, error).
// Param state could be nil in order to get all sandboxes belonging to same pod.
func (m *kubeGenericRuntimeManager) getSandboxIDByPodUID(ctx context.Context, podUID kubetypes.UID) ([]string, error) {
	logger := klog.FromContext(ctx)
	sandboxes, err := m.getSandboxes(ctx, listOptions{podUID: podUID})
	if err != nil {
		logger.Error(err, "Failed to list sandboxes for pod", "podUID", podUID)
		return nil, err
	}

	if len(sandboxes) == 0 {
		return nil, nil
	}

	// Sort with newest first.
	sandboxIDs := make([]string, len(sandboxes))
	sort.Sort(podSandboxByCreatedThenID(sandboxes))
	for i, s := range sandboxes {
		sandboxIDs[i] = s.Id
	}

	return sandboxIDs, nil
}

// GetPortForward gets the endpoint the runtime will serve the port-forward request from.
func (m *kubeGenericRuntimeManager) GetPortForward(ctx context.Context, podName, podNamespace string, podUID kubetypes.UID, ports []int32) (*url.URL, error) {
	sandboxIDs, err := m.getSandboxIDByPodUID(ctx, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s: %v", format.PodDesc(podName, podNamespace, podUID), err)
	}
	if len(sandboxIDs) == 0 {
		return nil, fmt.Errorf("failed to find sandboxID for pod %s", format.PodDesc(podName, podNamespace, podUID))
	}
	req := &runtimeapi.PortForwardRequest{
		PodSandboxId: sandboxIDs[0],
		Port:         ports,
	}
	resp, err := m.runtimeService.PortForward(ctx, req)
	if err != nil {
		return nil, err
	}
	return url.Parse(resp.Url)
}
