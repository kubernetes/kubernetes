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

	"k8s.io/kubernetes/pkg/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// determineEffectiveSecurityContext gets container's security context from api.Pod and api.Container.
func (m *kubeGenericRuntimeManager) determineEffectiveSecurityContext(pod *api.Pod, container *api.Container, uid *int64, username *string) *runtimeapi.LinuxContainerSecurityContext {
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	synthesized := convertToRuntimeSecurityContext(effectiveSc)
	if synthesized == nil {
		synthesized = &runtimeapi.LinuxContainerSecurityContext{}
	}

	// set RunAsUser.
	if synthesized.RunAsUser == nil {
		synthesized.RunAsUser = uid
		synthesized.RunAsUsername = username
	}

	// set namespace options and supplemental groups.
	podSc := pod.Spec.SecurityContext
	if podSc == nil {
		return synthesized
	}
	synthesized.NamespaceOptions = &runtimeapi.NamespaceOption{
		HostNetwork: &podSc.HostNetwork,
		HostIpc:     &podSc.HostIPC,
		HostPid:     &podSc.HostPID,
	}
	if podSc.FSGroup != nil {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, *podSc.FSGroup)
	}
	if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, groups...)
	}
	if podSc.SupplementalGroups != nil {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, podSc.SupplementalGroups...)
	}

	return synthesized
}

// verifyRunAsNonRoot verifies RunAsNonRoot.
func verifyRunAsNonRoot(pod *api.Pod, container *api.Container, uid int64) error {
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	if effectiveSc == nil || effectiveSc.RunAsNonRoot == nil {
		return nil
	}

	if effectiveSc.RunAsUser != nil {
		if *effectiveSc.RunAsUser == 0 {
			return fmt.Errorf("container's runAsUser breaks non-root policy")
		}
		return nil
	}

	if uid == 0 {
		return fmt.Errorf("container has runAsNonRoot and image will run as root")
	}

	return nil
}

// convertToRuntimeSecurityContext converts api.SecurityContext to runtimeapi.SecurityContext.
func convertToRuntimeSecurityContext(securityContext *api.SecurityContext) *runtimeapi.LinuxContainerSecurityContext {
	if securityContext == nil {
		return nil
	}

	return &runtimeapi.LinuxContainerSecurityContext{
		RunAsUser:      securityContext.RunAsUser,
		Privileged:     securityContext.Privileged,
		ReadonlyRootfs: securityContext.ReadOnlyRootFilesystem,
		Capabilities:   convertToRuntimeCapabilities(securityContext.Capabilities),
		SelinuxOptions: convertToRuntimeSELinuxOption(securityContext.SELinuxOptions),
	}
}

// convertToRuntimeSELinuxOption converts api.SELinuxOptions to runtimeapi.SELinuxOption.
func convertToRuntimeSELinuxOption(opts *api.SELinuxOptions) *runtimeapi.SELinuxOption {
	if opts == nil {
		return nil
	}

	return &runtimeapi.SELinuxOption{
		User:  &opts.User,
		Role:  &opts.Role,
		Type:  &opts.Type,
		Level: &opts.Level,
	}
}

// convertToRuntimeCapabilities converts api.Capabilities to runtimeapi.Capability.
func convertToRuntimeCapabilities(opts *api.Capabilities) *runtimeapi.Capability {
	if opts == nil {
		return nil
	}

	capabilities := &runtimeapi.Capability{
		AddCapabilities:  make([]string, len(opts.Add)),
		DropCapabilities: make([]string, len(opts.Drop)),
	}
	for index, value := range opts.Add {
		capabilities.AddCapabilities[index] = string(value)
	}
	for index, value := range opts.Drop {
		capabilities.DropCapabilities[index] = string(value)
	}

	return capabilities
}
