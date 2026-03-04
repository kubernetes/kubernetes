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
	v1 "k8s.io/api/core/v1"
	utilsysctl "k8s.io/component-helpers/node/util/sysctl"
)

// ConvertPodSysctlsVariableToDotsSeparator converts sysctls variable in the Pod.Spec.SecurityContext.Sysctls slice into a dot as a separator
// according to the linux sysctl conversion rules.
// see https://man7.org/linux/man-pages/man5/sysctl.d.5.html for more details.
func ConvertPodSysctlsVariableToDotsSeparator(securityContext *v1.PodSecurityContext) {
	if securityContext == nil {
		return
	}
	for i, sysctl := range securityContext.Sysctls {
		securityContext.Sysctls[i].Name = utilsysctl.NormalizeName(sysctl.Name)
	}
	return
}
