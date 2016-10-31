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
	"k8s.io/kubernetes/pkg/api"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

// determineEffectiveSecurityContext gets container's security context from api.Pod and container.SecurityContext.
func (m *kubeGenericRuntimeManager) determineEffectiveSecurityContext(pod *api.Pod, sc *api.SecurityContext) *runtimeapi.SecurityContext {
	effectiveSc := m.securityContextFromPodSecurityContext(pod)
	if effectiveSc == nil && sc == nil {
		return nil
	}
	if effectiveSc != nil && sc == nil {
		return effectiveSc
	}
	if effectiveSc == nil && sc != nil {
		return convertToRuntimeSecurityContext(sc)
	}

	if sc.SELinuxOptions != nil {
		effectiveSc.SelinuxOptions = convertToRuntimeSELinuxOption(sc.SELinuxOptions)
	}

	if sc.Capabilities != nil {
		effectiveSc.Capabilities = convertToRuntimeCapabilities(sc.Capabilities)
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

	if sc.ReadOnlyRootFilesystem != nil {
		effectiveSc.ReadonlyRootfs = sc.ReadOnlyRootFilesystem
	}

	return effectiveSc
}

// securityContextFromPodSecurityContext gets security context from api.Pod.
func (m *kubeGenericRuntimeManager) securityContextFromPodSecurityContext(pod *api.Pod) *runtimeapi.SecurityContext {
	if pod.Spec.SecurityContext == nil {
		return nil
	}

	podSc := pod.Spec.SecurityContext
	synthesized := &runtimeapi.SecurityContext{
		NamespaceOptions: &runtimeapi.NamespaceOption{
			HostNetwork: &podSc.HostNetwork,
			HostIpc:     &podSc.HostIPC,
			HostPid:     &podSc.HostPID,
		},
	}
	if podSc.SELinuxOptions != nil {
		synthesized.SelinuxOptions = convertToRuntimeSELinuxOption(podSc.SELinuxOptions)
	}
	if podSc.RunAsUser != nil {
		synthesized.RunAsUser = podSc.RunAsUser
	}
	if podSc.RunAsNonRoot != nil {
		synthesized.RunAsNonRoot = podSc.RunAsNonRoot
	}

	if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, groups...)
	}
	if podSc.SupplementalGroups != nil {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, podSc.SupplementalGroups...)
	}

	if podSc.FSGroup != nil {
		synthesized.FsGroup = podSc.FSGroup
	}

	return synthesized
}

// convertToRuntimeSecurityContext converts api.SecurityContext to runtimeapi.SecurityContext.
func convertToRuntimeSecurityContext(securityContext *api.SecurityContext) *runtimeapi.SecurityContext {
	if securityContext == nil {
		return nil
	}

	return &runtimeapi.SecurityContext{
		RunAsUser:      securityContext.RunAsUser,
		Privileged:     securityContext.Privileged,
		RunAsNonRoot:   securityContext.RunAsNonRoot,
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
