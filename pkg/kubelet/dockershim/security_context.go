// +build !dockerless

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

	dockercontainer "github.com/docker/docker/api/types/container"

	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	knetwork "k8s.io/kubernetes/pkg/kubelet/dockershim/network"
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
			RunAsGroup:         lc.SecurityContext.RunAsGroup,
			ReadonlyRootfs:     lc.SecurityContext.ReadonlyRootfs,
			SelinuxOptions:     lc.SecurityContext.SelinuxOptions,
			NamespaceOptions:   lc.SecurityContext.NamespaceOptions,
		}
	}

	err := modifyContainerConfig(sc, config)
	if err != nil {
		return err
	}

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

	err := modifyContainerConfig(lc.SecurityContext, config)
	if err != nil {
		return err
	}
	if err := modifyHostConfig(lc.SecurityContext, hc, separator); err != nil {
		return err
	}
	modifyContainerNamespaceOptions(lc.SecurityContext.GetNamespaceOptions(), podSandboxID, hc)
	return nil
}

// modifyContainerConfig applies container security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeapi.LinuxContainerSecurityContext, config *dockercontainer.Config) error {
	if sc == nil {
		return nil
	}
	if sc.RunAsUser != nil {
		config.User = strconv.FormatInt(sc.GetRunAsUser().Value, 10)
	}
	if sc.RunAsUsername != "" {
		config.User = sc.RunAsUsername
	}

	user := config.User
	if sc.RunAsGroup != nil {
		if user == "" {
			return fmt.Errorf("runAsGroup is specified without a runAsUser")
		}
		user = fmt.Sprintf("%s:%d", config.User, sc.GetRunAsGroup().Value)
	}

	config.User = user

	return nil
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

	if !hostConfig.Privileged {
		hostConfig.MaskedPaths = sc.MaskedPaths
		hostConfig.ReadonlyPaths = sc.ReadonlyPaths
	}

	return nil
}

// modifySandboxNamespaceOptions apply namespace options for sandbox
func modifySandboxNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, hostConfig *dockercontainer.HostConfig, network *knetwork.PluginManager) {
	// The sandbox's PID namespace is the one that's shared, so CONTAINER and POD are equivalent for it
	if nsOpts.GetPid() == runtimeapi.NamespaceMode_NODE {
		hostConfig.PidMode = namespaceModeHost
	}
	modifyHostOptionsForSandbox(nsOpts, network, hostConfig)
}

// modifyContainerNamespaceOptions apply namespace options for container
func modifyContainerNamespaceOptions(nsOpts *runtimeapi.NamespaceOption, podSandboxID string, hostConfig *dockercontainer.HostConfig) {
	switch nsOpts.GetPid() {
	case runtimeapi.NamespaceMode_NODE:
		hostConfig.PidMode = namespaceModeHost
	case runtimeapi.NamespaceMode_POD:
		hostConfig.PidMode = dockercontainer.PidMode(fmt.Sprintf("container:%v", podSandboxID))
	case runtimeapi.NamespaceMode_TARGET:
		hostConfig.PidMode = dockercontainer.PidMode(fmt.Sprintf("container:%v", nsOpts.GetTargetId()))
	}
	modifyHostOptionsForContainer(nsOpts, podSandboxID, hostConfig)
}

// modifyHostOptionsForSandbox applies NetworkMode/UTSMode to sandbox's dockercontainer.HostConfig.
func modifyHostOptionsForSandbox(nsOpts *runtimeapi.NamespaceOption, network *knetwork.PluginManager, hc *dockercontainer.HostConfig) {
	if nsOpts.GetIpc() == runtimeapi.NamespaceMode_NODE {
		hc.IpcMode = namespaceModeHost
	}
	if nsOpts.GetNetwork() == runtimeapi.NamespaceMode_NODE {
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

// modifyHostOptionsForContainer applies NetworkMode/UTSMode to container's dockercontainer.HostConfig.
func modifyHostOptionsForContainer(nsOpts *runtimeapi.NamespaceOption, podSandboxID string, hc *dockercontainer.HostConfig) {
	sandboxNSMode := fmt.Sprintf("container:%v", podSandboxID)
	hc.NetworkMode = dockercontainer.NetworkMode(sandboxNSMode)
	hc.IpcMode = dockercontainer.IpcMode(sandboxNSMode)
	hc.UTSMode = ""

	if nsOpts.GetNetwork() == runtimeapi.NamespaceMode_NODE {
		hc.UTSMode = namespaceModeHost
	}
}
