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

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
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

type sysctlItem struct {
	name  string
	index int
}

// mustMatchPatterns implements the SysctlsStrategy interface
type mustMatchPatterns struct {
	safeWhitelist        []string
	allowedUnsafeSysctls []string
	forbiddenSysctls     []string
}

var (
	_ SysctlsStrategy = &mustMatchPatterns{}

	defaultSysctlsPatterns = []string{"*"}
)

// NewMustMatchPatterns creates a new mustMatchPatterns strategy that will provide validation.
// Passing nil means the default pattern, passing an empty list means to disallow all sysctls.
func NewMustMatchPatterns(safeWhitelist []string, allowedUnsafeSysctls []string, forbiddenSysctls []string) SysctlsStrategy {
	return &mustMatchPatterns{
		safeWhitelist:        safeWhitelist,
		allowedUnsafeSysctls: allowedUnsafeSysctls,
		forbiddenSysctls:     forbiddenSysctls,
	}
}

func (s *mustMatchPatterns) divideSysctls(sysctls []api.Sysctl) (safeSysctls, unsafeSysctls []sysctlItem) {
	for i, sysctl := range sysctls {
		found := false
		for _, ws := range s.safeWhitelist {
			if sysctl.Name == ws {
				found = true
				break
			}
		}
		if found {
			safeSysctls = append(safeSysctls, sysctlItem{name: sysctl.Name, index: i})
		} else {
			unsafeSysctls = append(unsafeSysctls, sysctlItem{name: sysctl.Name, index: i})
		}
	}
	return
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustMatchPatterns) Validate(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}

	var sysctls []api.Sysctl
	if pod.Spec.SecurityContext != nil {
		sysctls = pod.Spec.SecurityContext.Sysctls
	}

	if len(sysctls) > 0 {
		fieldPath := field.NewPath("pod", "spec", "securityContext")

		// TODO(jchaloup): check the allowedUnsafeSysctls and forbiddenSysctls does not overlap

		safeSysctls, unsafeSysctls := s.divideSysctls(sysctls)
		if len(safeSysctls) > 0 {
			// Is the sysctl forbidden?
			if len(s.forbiddenSysctls) > 0 {
				for _, sysctl := range safeSysctls {
					// TODO(jchaloup): find the correct index i of the sysctl
					allErrs = append(allErrs, s.ValidateForbiddenSysctl(sysctl.name, fieldPath.Child("forbiddenSysctls").Index(sysctl.index))...)
				}
			}
		}

		if len(unsafeSysctls) > 0 {
			// Is the sysctl allowed?
			if len(s.allowedUnsafeSysctls) > 0 {
				for _, sysctl := range unsafeSysctls {
					// TODO(jchaloup): find the correct index i of the sysctl
					allErrs = append(allErrs, s.ValidateSysctl(sysctl.name, fieldPath.Child("allowedUnsafeSysctls").Index(sysctl.index))...)
				}
			}
		}
	}

	return allErrs
}

func (s *mustMatchPatterns) ValidateForbiddenSysctl(sysctlName string, fldPath *field.Path) field.ErrorList {
	for _, s := range s.forbiddenSysctls {
		if s[len(s)-1] == '*' {
			prefix := s[:len(s)-1]
			if strings.HasPrefix(sysctlName, string(prefix)) {
				return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("sysctl %q is not allowed", sysctlName))}
			}
		} else if sysctlName == s {
			return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("sysctl %q is not allowed", sysctlName))}
		}
	}
	return nil
}

func (s *mustMatchPatterns) ValidateSysctl(sysctlName string, fldPath *field.Path) field.ErrorList {
	for _, s := range s.allowedUnsafeSysctls {
		if s[len(s)-1] == '*' {
			prefix := s[:len(s)-1]
			if strings.HasPrefix(sysctlName, string(prefix)) {
				return nil
			}
		} else if sysctlName == s {
			return nil
		}
	}
	return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("sysctl %q is not allowed", sysctlName))}
}
