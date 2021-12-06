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

package sysctl

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/apis/core/validation"
	policyvalidation "k8s.io/kubernetes/pkg/apis/policy/validation"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	AnnotationInvalidReason = "InvalidSysctlAnnotation"
	ForbiddenReason         = "SysctlForbidden"
)

// patternAllowlist takes a list of sysctls or sysctl patterns (ending in *) and
// checks validity via a sysctl and prefix map, rejecting those which are not known
// to be namespaced.
type patternAllowlist struct {
	sysctls  map[string]Namespace
	prefixes map[string]Namespace
}

var _ lifecycle.PodAdmitHandler = &patternAllowlist{}

// NewAllowlist creates a new Allowlist from a list of sysctls and sysctl pattern (ending in *).
func NewAllowlist(patterns []string) (*patternAllowlist, error) {
	w := &patternAllowlist{
		sysctls:  map[string]Namespace{},
		prefixes: map[string]Namespace{},
	}

	for _, s := range patterns {
		if !policyvalidation.IsValidSysctlPattern(s) {
			return nil, fmt.Errorf("sysctl %q must have at most %d characters and match regex %s",
				s,
				validation.SysctlMaxLength,
				policyvalidation.SysctlContainSlashPatternFmt,
			)
		}
		s = convertSysctlVariableToDotsSeparator(s)
		if strings.HasSuffix(s, "*") {
			prefix := s[:len(s)-1]
			ns := NamespacedBy(prefix)
			if ns == unknownNamespace {
				return nil, fmt.Errorf("the sysctls %q are not known to be namespaced", s)
			}
			w.prefixes[prefix] = ns
		} else {
			ns := NamespacedBy(s)
			if ns == unknownNamespace {
				return nil, fmt.Errorf("the sysctl %q are not known to be namespaced", s)
			}
			w.sysctls[s] = ns
		}
	}
	return w, nil
}

// validateSysctl checks that a sysctl is allowlisted because it is known
// to be namespaced by the Linux kernel. Note that being allowlisted is required, but not
// sufficient: the container runtime might have a stricter check and refuse to launch a pod.
//
// The parameters hostNet and hostIPC are used to forbid sysctls for pod sharing the
// respective namespaces with the host. This check is only possible for sysctls on
// the static default allowlist, not those on the custom allowlist provided by the admin.
func (w *patternAllowlist) validateSysctl(sysctl string, hostNet, hostIPC bool) error {
	sysctl = convertSysctlVariableToDotsSeparator(sysctl)
	nsErrorFmt := "%q not allowed with host %s enabled"
	if ns, found := w.sysctls[sysctl]; found {
		if ns == ipcNamespace && hostIPC {
			return fmt.Errorf(nsErrorFmt, sysctl, ns)
		}
		if ns == netNamespace && hostNet {
			return fmt.Errorf(nsErrorFmt, sysctl, ns)
		}
		return nil
	}
	for p, ns := range w.prefixes {
		if strings.HasPrefix(sysctl, p) {
			if ns == ipcNamespace && hostIPC {
				return fmt.Errorf(nsErrorFmt, sysctl, ns)
			}
			if ns == netNamespace && hostNet {
				return fmt.Errorf(nsErrorFmt, sysctl, ns)
			}
			return nil
		}
	}
	return fmt.Errorf("%q not allowlisted", sysctl)
}

// Admit checks that all sysctls given in pod's security context
// are valid according to the allowlist.
func (w *patternAllowlist) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	pod := attrs.Pod
	if pod.Spec.SecurityContext == nil || len(pod.Spec.SecurityContext.Sysctls) == 0 {
		return lifecycle.PodAdmitResult{
			Admit: true,
		}
	}

	var hostNet, hostIPC bool
	if pod.Spec.SecurityContext != nil {
		hostNet = pod.Spec.HostNetwork
		hostIPC = pod.Spec.HostIPC
	}
	for _, s := range pod.Spec.SecurityContext.Sysctls {
		if err := w.validateSysctl(s.Name, hostNet, hostIPC); err != nil {
			return lifecycle.PodAdmitResult{
				Admit:   false,
				Reason:  ForbiddenReason,
				Message: fmt.Sprintf("forbidden sysctl: %v", err),
			}
		}
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
