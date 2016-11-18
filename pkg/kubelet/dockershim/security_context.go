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

	"k8s.io/kubernetes/pkg/api/v1"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// applySandboxSecurityContext updates docker sandbox options according to security context.
func applySandboxSecurityContext(lc *runtimeapi.LinuxPodSandboxConfig, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	if lc == nil {
		return
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
	modifyHostConfig(sc, "", hc)
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

// modifyContainerConfig applies container security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeapi.LinuxContainerSecurityContext, config *dockercontainer.Config) {
	if sc == nil {
		return
	}
	if sc.RunAsUser != nil {
		config.User = strconv.FormatInt(sc.GetRunAsUser(), 10)
	}
	if sc.RunAsUsername != nil {
		config.User = sc.GetRunAsUsername()
	}
}

// modifyHostConfig applies security context config to dockercontainer.HostConfig.
func modifyHostConfig(sc *runtimeapi.LinuxContainerSecurityContext, sandboxID string, hostConfig *dockercontainer.HostConfig) {
	// Apply namespace options.
	modifyNamespaceOptions(sc.GetNamespaceOptions(), sandboxID, hostConfig)

	if sc == nil {
		return
	}

	// Apply supplemental groups.
	for _, group := range sc.SupplementalGroups {
		hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(group, 10))
	}

	// Apply security context for the container.
	if sc.Privileged != nil {
		hostConfig.Privileged = sc.GetPrivileged()
	}
	if sc.ReadonlyRootfs != nil {
		hostConfig.ReadonlyRootfs = sc.GetReadonlyRootfs()
	}
	if sc.Capabilities != nil {
		hostConfig.CapAdd = sc.GetCapabilities().GetAddCapabilities()
		hostConfig.CapDrop = sc.GetCapabilities().GetDropCapabilities()
	}
	if sc.SelinuxOptions != nil {
		hostConfig.SecurityOpt = securitycontext.ModifySecurityOptions(
			hostConfig.SecurityOpt,
			&v1.SELinuxOptions{
				User:  sc.SelinuxOptions.GetUser(),
				Role:  sc.SelinuxOptions.GetRole(),
				Type:  sc.SelinuxOptions.GetType(),
				Level: sc.SelinuxOptions.GetLevel(),
			},
		)
	}
}

// modifyNamespaceOptions applies namespaceoptions to dockercontainer.HostConfig.
func modifyNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, sandboxID string, hostConfig *dockercontainer.HostConfig) {
	hostNetwork := false
	if nsOpts != nil {
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
		// Set for container if sandboxID is provided.
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
