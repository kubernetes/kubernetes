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

package dockershim

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/blang/semver"
	dockercontainer "github.com/docker/docker/api/types/container"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	knetwork "k8s.io/kubernetes/pkg/kubelet/network"
)

// applySandboxSecurityContext updates docker sandbox options according to security context.
func applySandboxSecurityContext(lc *runtimeapi.LinuxPodSandboxConfig, config *dockercontainer.Config, hc *dockercontainer.HostConfig, network *knetwork.PluginManager, separator rune) error {
	if lc == nil {
		return nil
	}

	var sc *runtimeapi.LinuxContainerSecurityContext
	if lc.SecurityContext != nil {
		sc = &runtimeapi.LinuxContainerSecurityContext{
			SupplementalGroups: lc.SecurityContext.SupplementalGroups,
			RunAsUser:          lc.SecurityContext.RunAsUser,
			ReadonlyRootfs:     lc.SecurityContext.ReadonlyRootfs,
			SelinuxOptions:     lc.SecurityContext.SelinuxOptions,
			NamespaceOptions:   lc.SecurityContext.NamespaceOptions,
		}
	}

	modifyContainerConfig(sc, config)
	if err := modifyHostConfig(sc, hc, separator); err != nil {
		return err
	}
	modifySandboxNamespaceOptions(sc.GetNamespaceOptions(), hc, network)
	return nil
}

// applyContainerSecurityContext updates docker container options according to security context.
func applyContainerSecurityContext(lc *runtimeapi.LinuxContainerConfig, podSandboxID string, config *dockercontainer.Config, hc *dockercontainer.HostConfig, separator rune) error {
	if lc == nil {
		return nil
	}

	modifyContainerConfig(lc.SecurityContext, config)
	if err := modifyHostConfig(lc.SecurityContext, hc, separator); err != nil {
		return err
	}
	modifyContainerNamespaceOptions(lc.SecurityContext.GetNamespaceOptions(), podSandboxID, hc)
	return nil
}

// modifyContainerConfig applies container security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeapi.LinuxContainerSecurityContext, config *dockercontainer.Config) {
	if sc == nil {
		return
	}
	if sc.RunAsUser != nil {
		config.User = strconv.FormatInt(sc.GetRunAsUser().Value, 10)
	}
	if sc.RunAsUsername != "" {
		config.User = sc.RunAsUsername
	}
}

// modifyHostConfig applies security context config to dockercontainer.HostConfig.
func modifyHostConfig(sc *runtimeapi.LinuxContainerSecurityContext, hostConfig *dockercontainer.HostConfig, separator rune) error {
	if sc == nil {
		return nil
	}

	// Apply supplemental groups.
	for _, group := range sc.SupplementalGroups {
		hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(group, 10))
	}

	// Apply security context for the container.
	hostConfig.Privileged = sc.Privileged
	hostConfig.ReadonlyRootfs = sc.ReadonlyRootfs
	if sc.Capabilities != nil {
		hostConfig.CapAdd = sc.GetCapabilities().AddCapabilities
		hostConfig.CapDrop = sc.GetCapabilities().DropCapabilities
	}
	if sc.SelinuxOptions != nil {
		hostConfig.SecurityOpt = addSELinuxOptions(
			hostConfig.SecurityOpt,
			sc.SelinuxOptions,
			separator,
		)
	}

	// Apply apparmor options.
	apparmorSecurityOpts, err := getApparmorSecurityOpts(sc, separator)
	if err != nil {
		return fmt.Errorf("failed to generate apparmor security options: %v", err)
	}
	hostConfig.SecurityOpt = append(hostConfig.SecurityOpt, apparmorSecurityOpts...)

	if sc.NoNewPrivs {
		hostConfig.SecurityOpt = append(hostConfig.SecurityOpt, "no-new-privileges")
	}

	return nil
}

// modifySandboxNamespaceOptions apply namespace options for sandbox
func modifySandboxNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, hostConfig *dockercontainer.HostConfig, network *knetwork.PluginManager) {
	hostNetwork := false
	if nsOpts != nil {
		hostNetwork = nsOpts.HostNetwork
	}
	modifyCommonNamespaceOptions(nsOpts, hostConfig)
	modifyHostNetworkOptionForSandbox(hostNetwork, network, hostConfig)
}

// modifyContainerNamespaceOptions apply namespace options for container
func modifyContainerNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, podSandboxID string, hostConfig *dockercontainer.HostConfig) {
	hostNetwork := false
	if nsOpts != nil {
		hostNetwork = nsOpts.HostNetwork
	}
	hostConfig.PidMode = dockercontainer.PidMode(fmt.Sprintf("container:%v", podSandboxID))
	modifyCommonNamespaceOptions(nsOpts, hostConfig)
	modifyHostNetworkOptionForContainer(hostNetwork, podSandboxID, hostConfig)
}

// modifyCommonNamespaceOptions apply common namespace options for sandbox and container
func modifyCommonNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, hostConfig *dockercontainer.HostConfig) {
	if nsOpts != nil {
		if nsOpts.HostPid {
			hostConfig.PidMode = namespaceModeHost
		}
		if nsOpts.HostIpc {
			hostConfig.IpcMode = namespaceModeHost
		}
	}
}

// modifyHostNetworkOptionForSandbox applies NetworkMode/UTSMode to sandbox's dockercontainer.HostConfig.
func modifyHostNetworkOptionForSandbox(hostNetwork bool, network *knetwork.PluginManager, hc *dockercontainer.HostConfig) {
	if hostNetwork {
		hc.NetworkMode = namespaceModeHost
		return
	}

	if network == nil {
		hc.NetworkMode = "default"
		return
	}

	switch network.PluginName() {
	case "cni":
		fallthrough
	case "kubenet":
		hc.NetworkMode = "none"
	default:
		hc.NetworkMode = "default"
	}
}

// modifyHostNetworkOptionForContainer applies NetworkMode/UTSMode to container's dockercontainer.HostConfig.
func modifyHostNetworkOptionForContainer(hostNetwork bool, podSandboxID string, hc *dockercontainer.HostConfig) {
	sandboxNSMode := fmt.Sprintf("container:%v", podSandboxID)
	hc.NetworkMode = dockercontainer.NetworkMode(sandboxNSMode)
	hc.IpcMode = dockercontainer.IpcMode(sandboxNSMode)
	hc.UTSMode = ""

	if hostNetwork {
		hc.UTSMode = namespaceModeHost
	}
}

// modifyPIDNamespaceOverrides implements two temporary overrides for the default PID namespace sharing for Docker:
//     1. Docker engine prior to API Version 1.24 doesn't support attaching to another container's
//        PID namespace, and it didn't stabilize until 1.26. This check can be removed when Kubernetes'
//        minimum Docker version is at least 1.13.1 (API version 1.26).
//     2. The administrator has overridden the default behavior by means of a kubelet flag. This is an
//        "escape hatch" to return to previous behavior of isolated namespaces and should be removed once
//        no longer needed.
func modifyPIDNamespaceOverrides(disableSharedPID bool, version *semver.Version, hc *dockercontainer.HostConfig) {
	if !strings.HasPrefix(string(hc.PidMode), "container:") {
		return
	}
	if disableSharedPID || version.LT(semver.Version{Major: 1, Minor: 26}) {
		hc.PidMode = ""
	}
}
