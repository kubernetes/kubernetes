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

	v1 "k8s.io/api/core/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	runtimeutil "k8s.io/kubernetes/pkg/kubelet/kuberuntime/util"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// determineEffectiveSecurityContext gets container's security context from v1.Pod and v1.Container.
func (m *kubeGenericRuntimeManager) determineEffectiveSecurityContext(pod *v1.Pod, container *v1.Container, uid *int64, username string) (*runtimeapi.LinuxContainerSecurityContext, error) {
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	synthesized := convertToRuntimeSecurityContext(effectiveSc)
	if synthesized == nil {
		synthesized = &runtimeapi.LinuxContainerSecurityContext{
			MaskedPaths:   securitycontext.ConvertToRuntimeMaskedPaths(effectiveSc.ProcMount),
			ReadonlyPaths: securitycontext.ConvertToRuntimeReadonlyPaths(effectiveSc.ProcMount),
		}
	}
	var err error

	synthesized.Seccomp, err = m.getSeccompProfile(pod.Annotations, container.Name, pod.Spec.SecurityContext, container.SecurityContext, m.seccompDefault)
	if err != nil {
		return nil, err
	}

	// set ApparmorProfile.
	synthesized.Apparmor, synthesized.ApparmorProfile, err = getAppArmorProfile(pod, container)
	if err != nil {
		return nil, err
	}

	// set RunAsUser.
	if synthesized.RunAsUser == nil {
		if uid != nil {
			synthesized.RunAsUser = &runtimeapi.Int64Value{Value: *uid}
		}
		synthesized.RunAsUsername = username
	}

	// set namespace options and supplemental groups.
	namespaceOptions, err := runtimeutil.NamespacesForPod(pod, m.runtimeHelper, m.runtimeClassManager)
	if err != nil {
		return nil, err
	}
	synthesized.NamespaceOptions = namespaceOptions
	podSc := pod.Spec.SecurityContext
	if podSc != nil {
		if podSc.FSGroup != nil {
			synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, int64(*podSc.FSGroup))
		}

		if podSc.SupplementalGroups != nil {
			for _, sg := range podSc.SupplementalGroups {
				synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, int64(sg))
			}
		}

		if podSc.SupplementalGroupsPolicy != nil {
			policyValue, ok := runtimeapi.SupplementalGroupsPolicy_value[string(*podSc.SupplementalGroupsPolicy)]
			if !ok {
				return nil, fmt.Errorf("unsupported supplementalGroupsPolicy: %s", string(*podSc.SupplementalGroupsPolicy))
			}
			synthesized.SupplementalGroupsPolicy = runtimeapi.SupplementalGroupsPolicy(policyValue)
		}
	}
	if groups := m.runtimeHelper.GetExtraSupplementalGroupsForPod(pod); len(groups) > 0 {
		synthesized.SupplementalGroups = append(synthesized.SupplementalGroups, groups...)
	}

	synthesized.NoNewPrivs = securitycontext.AddNoNewPrivileges(effectiveSc)

	synthesized.MaskedPaths = securitycontext.ConvertToRuntimeMaskedPaths(effectiveSc.ProcMount)
	synthesized.ReadonlyPaths = securitycontext.ConvertToRuntimeReadonlyPaths(effectiveSc.ProcMount)

	return synthesized, nil
}

// convertToRuntimeSecurityContext converts v1.SecurityContext to runtimeapi.SecurityContext.
func convertToRuntimeSecurityContext(securityContext *v1.SecurityContext) *runtimeapi.LinuxContainerSecurityContext {
	if securityContext == nil {
		return nil
	}

	sc := &runtimeapi.LinuxContainerSecurityContext{
		Capabilities:   convertToRuntimeCapabilities(securityContext.Capabilities),
		SelinuxOptions: convertToRuntimeSELinuxOption(securityContext.SELinuxOptions),
	}
	if securityContext.RunAsUser != nil {
		sc.RunAsUser = &runtimeapi.Int64Value{Value: int64(*securityContext.RunAsUser)}
	}
	if securityContext.RunAsGroup != nil {
		sc.RunAsGroup = &runtimeapi.Int64Value{Value: int64(*securityContext.RunAsGroup)}
	}
	if securityContext.Privileged != nil {
		sc.Privileged = *securityContext.Privileged
	}
	if securityContext.ReadOnlyRootFilesystem != nil {
		sc.ReadonlyRootfs = *securityContext.ReadOnlyRootFilesystem
	}

	return sc
}

// convertToRuntimeSELinuxOption converts v1.SELinuxOptions to runtimeapi.SELinuxOption.
func convertToRuntimeSELinuxOption(opts *v1.SELinuxOptions) *runtimeapi.SELinuxOption {
	if opts == nil {
		return nil
	}

	return &runtimeapi.SELinuxOption{
		User:  opts.User,
		Role:  opts.Role,
		Type:  opts.Type,
		Level: opts.Level,
	}
}

// convertToRuntimeCapabilities converts v1.Capabilities to runtimeapi.Capability.
func convertToRuntimeCapabilities(opts *v1.Capabilities) *runtimeapi.Capability {
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
