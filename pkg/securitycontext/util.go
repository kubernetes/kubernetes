/*
Copyright 2014 The Kubernetes Authors.

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
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/api"
)

// HasPrivilegedRequest returns the value of SecurityContext.Privileged, taking into account
// the possibility of nils
func HasPrivilegedRequest(container *v1.Container) bool {
	if container.SecurityContext == nil {
		return false
	}
	if container.SecurityContext.Privileged == nil {
		return false
	}
	return *container.SecurityContext.Privileged
}

// HasCapabilitiesRequest returns true if Adds or Drops are defined in the security context
// capabilities, taking into account nils
func HasCapabilitiesRequest(container *v1.Container) bool {
	if container.SecurityContext == nil {
		return false
	}
	if container.SecurityContext.Capabilities == nil {
		return false
	}
	return len(container.SecurityContext.Capabilities.Add) > 0 || len(container.SecurityContext.Capabilities.Drop) > 0
}

const expectedSELinuxFields = 4

// ParseSELinuxOptions parses a string containing a full SELinux context
// (user, role, type, and level) into an SELinuxOptions object.  If the
// context is malformed, an error is returned.
func ParseSELinuxOptions(context string) (*v1.SELinuxOptions, error) {
	fields := strings.SplitN(context, ":", expectedSELinuxFields)

	if len(fields) != expectedSELinuxFields {
		return nil, fmt.Errorf("expected %v fields in selinux; got %v (context: %v)", expectedSELinuxFields, len(fields), context)
	}

	return &v1.SELinuxOptions{
		User:  fields[0],
		Role:  fields[1],
		Type:  fields[2],
		Level: fields[3],
	}, nil
}

// HasNonRootUID returns true if the runAsUser is set and is greater than 0.
func HasRootUID(container *v1.Container) bool {
	if container.SecurityContext == nil {
		return false
	}
	if container.SecurityContext.RunAsUser == nil {
		return false
	}
	return *container.SecurityContext.RunAsUser == 0
}

// HasRunAsUser determines if the sc's runAsUser field is set.
func HasRunAsUser(container *v1.Container) bool {
	return container.SecurityContext != nil && container.SecurityContext.RunAsUser != nil
}

// HasRootRunAsUser returns true if the run as user is set and it is set to 0.
func HasRootRunAsUser(container *v1.Container) bool {
	return HasRunAsUser(container) && HasRootUID(container)
}

func DetermineEffectiveSecurityContext(pod *v1.Pod, container *v1.Container) *v1.SecurityContext {
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
		effectiveSc.SELinuxOptions = new(v1.SELinuxOptions)
		*effectiveSc.SELinuxOptions = *containerSc.SELinuxOptions
	}

	if containerSc.Capabilities != nil {
		effectiveSc.Capabilities = new(v1.Capabilities)
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

	if containerSc.ReadOnlyRootFilesystem != nil {
		effectiveSc.ReadOnlyRootFilesystem = new(bool)
		*effectiveSc.ReadOnlyRootFilesystem = *containerSc.ReadOnlyRootFilesystem
	}

	if containerSc.AllowPrivilegeEscalation != nil {
		effectiveSc.AllowPrivilegeEscalation = new(bool)
		*effectiveSc.AllowPrivilegeEscalation = *containerSc.AllowPrivilegeEscalation
	}

	return effectiveSc
}

func securityContextFromPodSecurityContext(pod *v1.Pod) *v1.SecurityContext {
	if pod.Spec.SecurityContext == nil {
		return nil
	}

	synthesized := &v1.SecurityContext{}

	if pod.Spec.SecurityContext.SELinuxOptions != nil {
		synthesized.SELinuxOptions = &v1.SELinuxOptions{}
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

// TODO: remove the duplicate code
func InternalDetermineEffectiveSecurityContext(pod *api.Pod, container *api.Container) *api.SecurityContext {
	effectiveSc := internalSecurityContextFromPodSecurityContext(pod)
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

	if containerSc.ReadOnlyRootFilesystem != nil {
		effectiveSc.ReadOnlyRootFilesystem = new(bool)
		*effectiveSc.ReadOnlyRootFilesystem = *containerSc.ReadOnlyRootFilesystem
	}

	if containerSc.AllowPrivilegeEscalation != nil {
		effectiveSc.AllowPrivilegeEscalation = new(bool)
		*effectiveSc.AllowPrivilegeEscalation = *containerSc.AllowPrivilegeEscalation
	}

	return effectiveSc
}

func internalSecurityContextFromPodSecurityContext(pod *api.Pod) *api.SecurityContext {
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

// AddNoNewPrivileges returns if we should add the no_new_privs option.
func AddNoNewPrivileges(sc *v1.SecurityContext) bool {
	if sc == nil {
		return false
	}

	// handle the case where the user did not set the default and did not explicitly set allowPrivilegeEscalation
	if sc.AllowPrivilegeEscalation == nil {
		return false
	}

	// handle the case where defaultAllowPrivilegeEscalation is false or the user explicitly set allowPrivilegeEscalation to true/false
	return !*sc.AllowPrivilegeEscalation
}
