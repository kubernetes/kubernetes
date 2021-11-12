/*
Copyright 2021 The Kubernetes Authors.

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
	"strings"

	"k8s.io/api/core/v1"
)

// convertSysctlVariableToDotsSeparator can return sysctl variables in dots separator format.
// The '/' separator is also accepted in place of a '.'.
// Convert the sysctl variables to dots separator format for validation.
// More info:
//   https://man7.org/linux/man-pages/man8/sysctl.8.html
//   https://man7.org/linux/man-pages/man5/sysctl.d.5.html
func convertSysctlVariableToDotsSeparator(val string) string {
	if val == "" {
		return val
	}
	firstSepIndex := strings.IndexAny(val, "./")
	if firstSepIndex == -1 || val[firstSepIndex] == '.' {
		return val
	}

	f := func(r rune) rune {
		switch r {
		case '.':
			return '/'
		case '/':
			return '.'
		}
		return r
	}
	return strings.Map(f, val)
}

// ConvertPodSysctlsVariableToDotsSeparator converts sysctls variable in the Pod.Spec.SecurityContext.Sysctls slice into a dot as a separator
// according to the linux sysctl conversion rules.
// see https://man7.org/linux/man-pages/man5/sysctl.d.5.html for more details.
func ConvertPodSysctlsVariableToDotsSeparator(securityContext *v1.PodSecurityContext) {
	if securityContext == nil {
		return
	}
	for i, sysctl := range securityContext.Sysctls {
		securityContext.Sysctls[i].Name = convertSysctlVariableToDotsSeparator(sysctl.Name)
	}
	return
}
