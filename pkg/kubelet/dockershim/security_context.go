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
	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

const (
	dockerLabelUser  string = "label:user"
	dockerLabelRole  string = "label:role"
	dockerLabelType  string = "label:type"
	dockerLabelLevel string = "label:level"
)

// applySandboxSecurityContext updates docker sandbox options according to security context.
func applySandboxSecurityContext(podSc *runtimeApi.PodSecurityContext, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	modifyContainerConfig(nil, podSc, config)
	modifyHostConfig(nil, podSc, hc, true)
}

// applyContainerSecurityContext updates docker container options according to security context.
func applyContainerSecurityContext(podSc *runtimeApi.LinuxPodSandboxConfig, lc *runtimeApi.LinuxContainerConfig, config *dockercontainer.Config, hc *dockercontainer.HostConfig) {
	if podSc == nil && lc == nil {
		return
	}

	if podSc == nil && lc != nil {
		modifyContainerConfig(lc.SecurityContext, nil, config)
		modifyHostConfig(lc.SecurityContext, nil, hc, false)
		return
	}

	if podSc != nil && lc == nil {
		modifyContainerConfig(nil, podSc.SecurityContext, config)
		modifyHostConfig(nil, podSc.SecurityContext, hc, false)
		return
	}

	modifyContainerConfig(lc.SecurityContext, podSc.SecurityContext, config)
	modifyHostConfig(lc.SecurityContext, podSc.SecurityContext, hc, false)
}

// modifyContainerConfig applies security context config to dockercontainer.Config.
func modifyContainerConfig(sc *runtimeApi.SecurityContext, podSc *runtimeApi.PodSecurityContext, config *dockercontainer.Config) {
	effectiveSC := determineEffectiveSecurityContext(podSc, sc)
	if effectiveSC == nil {
		return
	}
	if effectiveSC.RunAsUser != nil {
		config.User = strconv.Itoa(int(*effectiveSC.RunAsUser))
	}
}

// modifyHostConfig applies security context config to dockercontainer.HostConfig.
func modifyHostConfig(sc *runtimeApi.SecurityContext, podSc *runtimeApi.PodSecurityContext, hostConfig *dockercontainer.HostConfig, isSandbox bool) {
	// Apply supplemental groups
	// TODO: We skip application of supplemental groups to the
	// sandbox container to work around a runc issue which
	// requires containers to have the '/etc/group'. For
	// more information see:
	// https://github.com/opencontainers/runc/pull/313
	// This can be removed once the fix makes it into the
	// required version of docker.
	if !isSandbox && podSc != nil {
		for _, group := range podSc.SupplementGroups {
			hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(group, 10))
		}
		if podSc.FsGroup != nil {
			hostConfig.GroupAdd = append(hostConfig.GroupAdd, strconv.FormatInt(*podSc.FsGroup, 10))
		}
	}

	// Apply effective security context for container
	effectiveSC := determineEffectiveSecurityContext(podSc, sc)
	if effectiveSC == nil {
		return
	}

	if effectiveSC.Privileged != nil {
		hostConfig.Privileged = effectiveSC.GetPrivileged()
	}

	if effectiveSC.ReadonlyRootfs != nil {
		hostConfig.ReadonlyRootfs = effectiveSC.GetReadonlyRootfs()
	}

	if effectiveSC.Capabilities != nil {
		add, drop := MakeCapabilities(effectiveSC.Capabilities.AddCapabilities, effectiveSC.Capabilities.DropCapabilities)
		hostConfig.CapAdd = add
		hostConfig.CapDrop = drop
	}

	if effectiveSC.SelinuxOptions != nil {
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelUser, effectiveSC.SelinuxOptions.GetUser())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelRole, effectiveSC.SelinuxOptions.GetRole())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelType, effectiveSC.SelinuxOptions.GetType())
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelLevel, effectiveSC.SelinuxOptions.GetLevel())
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

func determineEffectiveSecurityContext(podSc *runtimeApi.PodSecurityContext, sc *runtimeApi.SecurityContext) *runtimeApi.SecurityContext {
	effectiveSc := securityContextFromPodSecurityContext(podSc)
	if effectiveSc == nil && sc == nil {
		return nil
	}
	if effectiveSc != nil && sc == nil {
		return effectiveSc
	}
	if effectiveSc == nil && sc != nil {
		return sc
	}

	if sc.SelinuxOptions != nil {
		effectiveSc.SelinuxOptions = sc.SelinuxOptions
	}

	if sc.Capabilities != nil {
		effectiveSc.Capabilities = sc.Capabilities
	}

	if sc.Privileged != nil {
		effectiveSc.Privileged = sc.Privileged
	}

	if sc.RunAsUser != nil {
		effectiveSc.RunAsUser = sc.RunAsUser
	}

	if sc.RunAsNonRoot != nil {
		effectiveSc.RunAsNonRoot = sc.RunAsNonRoot
	}

	if sc.ReadonlyRootfs != nil {
		effectiveSc.ReadonlyRootfs = sc.ReadonlyRootfs
	}

	return effectiveSc
}

func securityContextFromPodSecurityContext(podSc *runtimeApi.PodSecurityContext) *runtimeApi.SecurityContext {
	if podSc == nil {
		return nil
	}

	synthesized := &runtimeApi.SecurityContext{}
	if podSc.SelinuxOptions != nil {
		synthesized.SelinuxOptions = podSc.SelinuxOptions
	}
	if podSc.RunAsUser != nil {
		synthesized.RunAsUser = podSc.RunAsUser
	}
	if podSc.RunAsNonRoot != nil {
		synthesized.RunAsNonRoot = podSc.RunAsNonRoot
	}

	return synthesized
}
