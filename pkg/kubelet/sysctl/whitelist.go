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

	"k8s.io/kubernetes/pkg/api/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	"k8s.io/kubernetes/pkg/api/validation"
	extvalidation "k8s.io/kubernetes/pkg/apis/extensions/validation"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	AnnotationInvalidReason = "InvalidSysctlAnnotation"
	ForbiddenReason         = "SysctlForbidden"
)

// SafeSysctlWhitelist returns the whitelist of safe sysctls and safe sysctl patterns (ending in *).
//
// A sysctl is called safe iff
// - it is namespaced in the container or the pod
// - it is isolated, i.e. has no influence on any other pod on the same node.
func SafeSysctlWhitelist() []string {
	return []string{
		"kernel.shm_rmid_forced",
		"net.ipv4.ip_local_port_range",
		"net.ipv4.tcp_syncookies",
	}
}

// Whitelist provides a list of allowed sysctls and sysctl patterns (ending in *)
// and a function to check whether a given sysctl matches this list.
type Whitelist interface {
	// Validate checks that all sysctls given in a v1.SysctlsPodAnnotationKey annotation
	// are valid according to the whitelist.
	Validate(pod *v1.Pod) error
}

// patternWhitelist takes a list of sysctls or sysctl patterns (ending in *) and
// checks validity via a sysctl and prefix map, rejecting those which are not known
// to be namespaced.
type patternWhitelist struct {
	sysctls       map[string]Namespace
	prefixes      map[string]Namespace
	annotationKey string
}

var _ lifecycle.PodAdmitHandler = &patternWhitelist{}

// NewWhitelist creates a new Whitelist from a list of sysctls and sysctl pattern (ending in *).
func NewWhitelist(patterns []string, annotationKey string) (*patternWhitelist, error) {
	w := &patternWhitelist{
		sysctls:       map[string]Namespace{},
		prefixes:      map[string]Namespace{},
		annotationKey: annotationKey,
	}

	for _, s := range patterns {
		if !extvalidation.IsValidSysctlPattern(s) {
			return nil, fmt.Errorf("sysctl %q must have at most %d characters and match regex %s",
				s,
				validation.SysctlMaxLength,
				extvalidation.SysctlPatternFmt,
			)
		}
		if strings.HasSuffix(s, "*") {
			prefix := s[:len(s)-1]
			ns := NamespacedBy(prefix)
			if ns == UnknownNamespace {
				return nil, fmt.Errorf("the sysctls %q are not known to be namespaced", s)
			}
			w.prefixes[prefix] = ns
		} else {
			ns := NamespacedBy(s)
			if ns == UnknownNamespace {
				return nil, fmt.Errorf("the sysctl %q are not known to be namespaced", s)
			}
			w.sysctls[s] = ns
		}
	}
	return w, nil
}

// validateSysctl checks that a sysctl is whitelisted because it is known
// to be namespaced by the Linux kernel. Note that being whitelisted is required, but not
// sufficient: the container runtime might have a stricter check and refuse to launch a pod.
//
// The parameters hostNet and hostIPC are used to forbid sysctls for pod sharing the
// respective namespaces with the host. This check is only possible for sysctls on
// the static default whitelist, not those on the custom whitelist provided by the admin.
func (w *patternWhitelist) validateSysctl(sysctl string, hostNet, hostIPC bool) error {
	nsErrorFmt := "%q not allowed with host %s enabled"
	if ns, found := w.sysctls[sysctl]; found {
		if ns == IpcNamespace && hostIPC {
			return fmt.Errorf(nsErrorFmt, sysctl, ns)
		}
		if ns == NetNamespace && hostNet {
			return fmt.Errorf(nsErrorFmt, sysctl, ns)
		}
		return nil
	}
	for p, ns := range w.prefixes {
		if strings.HasPrefix(sysctl, p) {
			if ns == IpcNamespace && hostIPC {
				return fmt.Errorf(nsErrorFmt, sysctl, ns)
			}
			if ns == NetNamespace && hostNet {
				return fmt.Errorf(nsErrorFmt, sysctl, ns)
			}
			return nil
		}
	}
	return fmt.Errorf("%q not whitelisted", sysctl)
}

// Admit checks that all sysctls given in a v1.SysctlsPodAnnotationKey annotation
// are valid according to the whitelist.
func (w *patternWhitelist) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	pod := attrs.Pod
	a := pod.Annotations[w.annotationKey]
	if a == "" {
		return lifecycle.PodAdmitResult{
			Admit: true,
		}
	}

	sysctls, err := v1helper.SysctlsFromPodAnnotation(a)
	if err != nil {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  AnnotationInvalidReason,
			Message: fmt.Sprintf("invalid %s annotation: %v", w.annotationKey, err),
		}
	}

	var hostNet, hostIPC bool
	if pod.Spec.SecurityContext != nil {
		hostNet = pod.Spec.HostNetwork
		hostIPC = pod.Spec.HostIPC
	}
	for _, s := range sysctls {
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
