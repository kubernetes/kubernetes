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

package policy

import (
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

/*

Sysctls can disable security mechanisms or affect all containers on a host,
and should be disallowed except for an allowed "safe" subset.

A sysctl is considered safe if it is namespaced in the container or the Pod,
and it is isolated from other Pods or processes on the same Node.

**Restricted Fields:**
spec.securityContext.sysctls[*].name

**Allowed Values:**
'kernel.shm_rmid_forced'
'net.ipv4.ip_local_port_range'
'net.ipv4.tcp_syncookies'
'net.ipv4.ping_group_range'
'net.ipv4.ip_unprivileged_port_start'

*/

func init() {
	addCheck(CheckSysctls)
}

// CheckSysctls returns a baseline level check
// that limits the value of sysctls in 1.0+
func CheckSysctls() Check {
	return Check{
		ID:    "sysctls",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       sysctls_1_0,
			},
		},
	}
}

var (
	sysctls_allowed_1_0 = sets.NewString(
		"kernel.shm_rmid_forced",
		"net.ipv4.ip_local_port_range",
		"net.ipv4.tcp_syncookies",
		"net.ipv4.ping_group_range",
		"net.ipv4.ip_unprivileged_port_start",
	)
)

func sysctls_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var forbiddenSysctls []string

	if podSpec.SecurityContext != nil {
		for _, sysctl := range podSpec.SecurityContext.Sysctls {
			if !sysctls_allowed_1_0.Has(sysctl.Name) {
				forbiddenSysctls = append(forbiddenSysctls, sysctl.Name)
			}
		}
	}

	if len(forbiddenSysctls) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "forbidden sysctls",
			ForbiddenDetail: strings.Join(forbiddenSysctls, ", "),
		}
	}
	return CheckResult{Allowed: true}
}
