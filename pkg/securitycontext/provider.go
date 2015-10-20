/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package securitycontext

import (
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/leaky"

	docker "github.com/fsouza/go-dockerclient"
)

// NewSimpleSecurityContextProvider creates a new SimpleSecurityContextProvider.
func NewSimpleSecurityContextProvider() SecurityContextProvider {
	return SimpleSecurityContextProvider{}
}

// SimpleSecurityContextProvider is the default implementation of a SecurityContextProvider.
type SimpleSecurityContextProvider struct{}

// ModifyContainerConfig is called before the Docker createContainer call.
// The security context provider can make changes to the Config with which
// the container is created.
func (p SimpleSecurityContextProvider) ModifyContainerConfig(pod *api.Pod, container *api.Container, config *docker.Config) {
	effectiveSC := determineEffectiveSecurityContext(pod, container)
	if effectiveSC == nil {
		return
	}
	if effectiveSC.RunAsUser != nil {
		config.User = strconv.Itoa(int(*effectiveSC.RunAsUser))
	}
}

// ModifyHostConfig is called before the Docker runContainer call.
// The security context provider can make changes to the HostConfig, affecting
// security options, whether the container is privileged, volume binds, etc.
func (p SimpleSecurityContextProvider) ModifyHostConfig(pod *api.Pod, container *api.Container, hostConfig *docker.HostConfig) {
	// Apply pod security context
	if pod.Spec.SecurityContext != nil {
		// We skip application of supplemental groups to the
		// infra container to work around a runc issue which
		// requires containers to have the '/etc/group'. For
		// more information see:
		// https://github.com/opencontainers/runc/pull/313
		// This can be removed once the fix makes it into the
		// required version of docker.
		if pod.Spec.SecurityContext.SupplementalGroups != nil && container.Name != leaky.PodInfraContainerName {
			hostConfig.GroupAdd = make([]string, len(pod.Spec.SecurityContext.SupplementalGroups))
			for i, group := range pod.Spec.SecurityContext.SupplementalGroups {
				hostConfig.GroupAdd[i] = strconv.Itoa(int(group))
			}
		}
	}

	// Apply effective security context for container
	effectiveSC := determineEffectiveSecurityContext(pod, container)
	if effectiveSC == nil {
		return
	}

	if effectiveSC.Privileged != nil {
		hostConfig.Privileged = *effectiveSC.Privileged
	}

	if effectiveSC.Capabilities != nil {
		add, drop := makeCapabilites(effectiveSC.Capabilities.Add, effectiveSC.Capabilities.Drop)
		hostConfig.CapAdd = add
		hostConfig.CapDrop = drop
	}

	if effectiveSC.SELinuxOptions != nil {
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelUser, effectiveSC.SELinuxOptions.User)
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelRole, effectiveSC.SELinuxOptions.Role)
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelType, effectiveSC.SELinuxOptions.Type)
		hostConfig.SecurityOpt = modifySecurityOption(hostConfig.SecurityOpt, dockerLabelLevel, effectiveSC.SELinuxOptions.Level)
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

// makeCapabilites creates string slices from Capability slices
func makeCapabilites(capAdd []api.Capability, capDrop []api.Capability) ([]string, []string) {
	var (
		addCaps  []string
		dropCaps []string
	)
	for _, cap := range capAdd {
		addCaps = append(addCaps, string(cap))
	}
	for _, cap := range capDrop {
		dropCaps = append(dropCaps, string(cap))
	}
	return addCaps, dropCaps
}

func determineEffectiveSecurityContext(pod *api.Pod, container *api.Container) *api.SecurityContext {
	effectiveSc := securityContextFromPodSecurityContext(pod)
	containerSc := container.SecurityContext

	if effectiveSc == nil && containerSc == nil {
		return nil
	}
	if effectiveSc != nil && containerSc == nil {
		return effectiveSc
	}
	if effectiveSc == nil && containerSc != nil {
		return containerSc
	}

	if containerSc.SELinuxOptions != nil {
		effectiveSc.SELinuxOptions = new(api.SELinuxOptions)
		*effectiveSc.SELinuxOptions = *containerSc.SELinuxOptions
	}

	if containerSc.Capabilities != nil {
		effectiveSc.Capabilities = new(api.Capabilities)
		*effectiveSc.Capabilities = *containerSc.Capabilities
	}

	if containerSc.Privileged != nil {
		effectiveSc.Privileged = new(bool)
		*effectiveSc.Privileged = *containerSc.Privileged
	}

	if containerSc.RunAsUser != nil {
		effectiveSc.RunAsUser = new(int64)
		*effectiveSc.RunAsUser = *containerSc.RunAsUser
	}

	if containerSc.RunAsNonRoot != nil {
		effectiveSc.RunAsNonRoot = new(bool)
		*effectiveSc.RunAsNonRoot = *containerSc.RunAsNonRoot
	}

	return effectiveSc
}

func securityContextFromPodSecurityContext(pod *api.Pod) *api.SecurityContext {
	if pod.Spec.SecurityContext == nil {
		return nil
	}

	synthesized := &api.SecurityContext{}

	if pod.Spec.SecurityContext.SELinuxOptions != nil {
		synthesized.SELinuxOptions = &api.SELinuxOptions{}
		*synthesized.SELinuxOptions = *pod.Spec.SecurityContext.SELinuxOptions
	}
	if pod.Spec.SecurityContext.RunAsUser != nil {
		synthesized.RunAsUser = new(int64)
		*synthesized.RunAsUser = *pod.Spec.SecurityContext.RunAsUser
	}

	if pod.Spec.SecurityContext.RunAsNonRoot != nil {
		synthesized.RunAsNonRoot = new(bool)
		*synthesized.RunAsNonRoot = *pod.Spec.SecurityContext.RunAsNonRoot
	}

	return synthesized
}
