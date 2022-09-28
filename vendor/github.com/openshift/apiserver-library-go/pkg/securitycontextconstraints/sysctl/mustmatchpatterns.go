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

// SafeSysctlAllowlist returns the allowlist of safe sysctls and safe sysctl patterns (ending in *).
//
// A sysctl is called safe iff
// - it is namespaced in the container or the pod
// - it is isolated, i.e. has no influence on any other pod on the same node.
func SafeSysctlAllowlist() []string {
	return []string{
		"kernel.shm_rmid_forced",
		"net.ipv4.ip_local_port_range",
		"net.ipv4.tcp_syncookies",
		"net.ipv4.ping_group_range",
		"net.ipv4.ip_unprivileged_port_start",
	}
}

// mustMatchPatterns implements the SysctlsStrategy interface
type mustMatchPatterns struct {
	safeAllowlist        []string
	allowedUnsafeSysctls []string
	forbiddenSysctls     []string
}

var (
	_ SysctlsStrategy = &mustMatchPatterns{}
)

// NewMustMatchPatterns creates a new mustMatchPatterns strategy that will provide validation.
// Passing nil means the default pattern, passing an empty list means to disallow all sysctls.
func NewMustMatchPatterns(safeAllowlist, allowedUnsafeSysctls, forbiddenSysctls []string) SysctlsStrategy {
	return &mustMatchPatterns{
		safeAllowlist:        safeAllowlist,
		allowedUnsafeSysctls: allowedUnsafeSysctls,
		forbiddenSysctls:     forbiddenSysctls,
	}
}

func (s *mustMatchPatterns) isForbidden(sysctlName string) bool {
	// Is the sysctl forbidden?
	for _, s := range s.forbiddenSysctls {
		if strings.HasSuffix(s, "*") {
			prefix := strings.TrimSuffix(s, "*")
			if strings.HasPrefix(sysctlName, prefix) {
				return true
			}
		} else if sysctlName == s {
			return true
		}
	}
	return false
}

func (s *mustMatchPatterns) isSafe(sysctlName string) bool {
	for _, ws := range s.safeAllowlist {
		if sysctlName == ws {
			return true
		}
	}
	return false
}

func (s *mustMatchPatterns) isAllowedUnsafe(sysctlName string) bool {
	for _, s := range s.allowedUnsafeSysctls {
		if strings.HasSuffix(s, "*") {
			prefix := strings.TrimSuffix(s, "*")
			if strings.HasPrefix(sysctlName, prefix) {
				return true
			}
		} else if sysctlName == s {
			return true
		}
	}
	return false
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustMatchPatterns) Validate(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}

	var sysctls []api.Sysctl
	if pod.Spec.SecurityContext != nil {
		sysctls = pod.Spec.SecurityContext.Sysctls
	}

	fieldPath := field.NewPath("pod", "spec", "securityContext").Child("sysctls")

	for i, sysctl := range sysctls {
		switch {
		case s.isForbidden(sysctl.Name):
			allErrs = append(allErrs, field.ErrorList{field.Forbidden(fieldPath.Index(i), fmt.Sprintf("sysctl %q is not allowed", sysctl.Name))}...)
		case s.isSafe(sysctl.Name):
			continue
		case s.isAllowedUnsafe(sysctl.Name):
			continue
		default:
			allErrs = append(allErrs, field.ErrorList{field.Forbidden(fieldPath.Index(i), fmt.Sprintf("unsafe sysctl %q is not allowed", sysctl.Name))}...)
		}
	}

	return allErrs
}
