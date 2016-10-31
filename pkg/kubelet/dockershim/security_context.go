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

	dockercontainer "github.com/docker/engine-api/types/container"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

const (
	dockerLabelUser  string = "label:user"
	dockerLabelRole  string = "label:role"
	dockerLabelType  string = "label:type"
	dockerLabelLevel string = "label:level"
)

// applySandboxSecurityContext updates docker sandbox options according to security context.
func applySandboxSecurityContext(lc *runtimeapi.LinuxPodSandboxConfig, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	if lc == nil {
		return
	}

	modifySandboxContainerConfig(lc.SecurityContext, config)
	modifySandboxHostConfig(lc.SecurityContext, hc)
}

// applyContainerSecurityContext updates docker container options according to security context.
func applyContainerSecurityContext(lc *runtimeapi.LinuxContainerConfig, sandboxID string, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	if lc == nil {
		return
	}

	modifyContainerConfig(lc.SecurityContext, config)
	modifyHostConfig(lc.SecurityContext, sandboxID, hc)
	return
}

// modifySandboxContainerConfig applies sandbox security context config to dockercontainer.Config.
func modifySandboxContainerConfig(podSc *runtimeapi.SandboxSecurityContext, config *dockercontainer.Config) {
	if podSc != nil && podSc.RunAsUser != nil {
		modifyContainerConfig(&runtimeapi.SecurityContext{
			RunAsUser: podSc.RunAsUser,
		}, config)
	}
}

// modifyContainerConfig applies container security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeapi.SecurityContext, config *dockercontainer.Config) {
	if sc != nil && sc.RunAsUser != nil {
		config.User = strconv.Itoa(int(*sc.RunAsUser))
	}
}

// modifySandboxHostConfig applies sandbox security context config to dockercontainer.HostConfig.
func modifySandboxHostConfig(podSc *runtimeapi.SandboxSecurityContext, hc *dockercontainer.HostConfig) {
	var sc *runtimeapi.SecurityContext
	if podSc != nil {
		sc = &runtimeapi.SecurityContext{
			// TODO: We skip application of supplemental groups to the
			// sandbox container to work around a runc issue which
			// requires containers to have the '/etc/group'. For more
			// information see: https://github.com/opencontainers/runc/pull/313.
			// This can be removed once the fix makes it into the required
			// version of docker.
			ReadonlyRootfs:   podSc.ReadonlyRootfs,
			SelinuxOptions:   podSc.SelinuxOptions,
			NamespaceOptions: podSc.NamespaceOptions,
		}
	}

	modifyHostConfig(sc, "", hc)
}

// modifyHostConfig applies security context config to dockercontainer.HostConfig.
func modifyHostConfig(sc *runtimeapi.SecurityContext, sandboxID string, hostConfig *dockercontainer.HostConfig) {
	// Apply namespace options.
	modifyNamespaceOptions(sc, sandboxID, hostConfig)

	if sc == nil {
		return
	}

	// Apply supplemental groups.
	for _, group := range sc.SupplementalGroups {
		hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(group, 10))
	}
	if sc.FsGroup != nil {
		hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(*sc.FsGroup, 10))
	}

	// Apply security context for the container.
	if sc.Privileged != nil {
		hostConfig.Privileged = sc.GetPrivileged()
	}
	if sc.ReadonlyRootfs != nil {
		hostConfig.ReadonlyRootfs = sc.GetReadonlyRootfs()
	}
	if sc.Capabilities != nil {
		add, drop := MakeCapabilities(sc.Capabilities.AddCapabilities, sc.Capabilities.DropCapabilities)
		hostConfig.CapAdd = add
		hostConfig.CapDrop = drop
	}
	if sc.SelinuxOptions != nil {
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelUser, sc.SelinuxOptions.GetUser())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelRole, sc.SelinuxOptions.GetRole())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelType, sc.SelinuxOptions.GetType())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelLevel, sc.SelinuxOptions.GetLevel())
	}
}

// modifyNamespaceOptions applies namespaceoptions to dockercontainer.HostConfig.
func modifyNamespaceOptions(sc *runtimeapi.SecurityContext, sandboxID string, hostConfig *dockercontainer.HostConfig) {
	hostNetwork := false
	if sc != nil && sc.NamespaceOptions != nil {
		nsOpts := sc.GetNamespaceOptions()
		if nsOpts.HostNetwork != nil {
			hostNetwork = nsOpts.GetHostNetwork()
		}
		if nsOpts.GetHostPid() {
			hostConfig.PidMode = namespaceModeHost
		}
		if nsOpts.GetHostIpc() {
			hostConfig.IpcMode = namespaceModeHost
		}
	}

	// Set for sandbox if sandboxID is not provided.
	if sandboxID == "" {
		modifyHostNetworkOptionForSandbox(hostNetwork, hostConfig)
	} else {
		// Set for container is sandboxID is provided.
		modifyHostNetworkOptionForContainer(hostNetwork, sandboxID, hostConfig)
	}
}

// modifyHostNetworkOptionForSandbox applies NetworkMode/UTSMode to sandbox's dockercontainer.HostConfig.
func modifyHostNetworkOptionForSandbox(hostNetwork bool, hc *dockercontainer.HostConfig) {
	if hostNetwork {
		hc.NetworkMode = namespaceModeHost
	} else {
		// Assume kubelet uses either the cni or the kubenet plugin.
		// TODO: support docker networking.
		hc.NetworkMode = "none"
	}
}

// modifyHostNetworkOptionForContainer applies NetworkMode/UTSMode to container's dockercontainer.HostConfig.
func modifyHostNetworkOptionForContainer(hostNetwork bool, sandboxID string, hc *dockercontainer.HostConfig) {
	sandboxNSMode := fmt.Sprintf("container:%v", sandboxID)
	hc.NetworkMode = dockercontainer.NetworkMode(sandboxNSMode)
	hc.IpcMode = dockercontainer.IpcMode(sandboxNSMode)
	hc.UTSMode = ""
	hc.PidMode = ""

	if hostNetwork {
		hc.UTSMode = namespaceModeHost
	}
}

// modifySecurityOption adds the security option of name to the config array with value in the form
// of name:value
func modifySecurityOption(config []string, name, value string) []string {
	if len(value) > 0 {
		config = append(config, fmt.Sprintf("%s:%s", name, value))
	}
	return config
}

// MakeCapabilities creates string slices from Capability slices
func MakeCapabilities(capAdd []string, capDrop []string) ([]string, []string) {
	var (
		addCaps  []string
		dropCaps []string
	)
	for _, cap := range capAdd {
		addCaps = append(addCaps, cap)
	}
	for _, cap := range capDrop {
		dropCaps = append(dropCaps, cap)
	}
	return addCaps, dropCaps
}
