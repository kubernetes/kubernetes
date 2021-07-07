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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

func init() {
	addCheck(CheckSeccompRestricted)
}

func CheckSeccompRestricted() Check {
	return Check{
		ID:    "seccomp_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 19),
				CheckPod:       seccomp_1_19_restricted,
			},
		},
	}
}

// seccomp_1_19_restricted checks restricted policy on securityContext.seccompProfile field
func seccomp_1_19_restricted(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	forbidden := sets.NewString()
	podSeccompField := field.NewPath("spec").Child("securityContext", "seccompProfile", "type")
	podSeccompSet := false

	if podSpec.SecurityContext != nil {
		if podSpec.SecurityContext.SeccompProfile != nil {
			seccompType := podSpec.SecurityContext.SeccompProfile.Type
			if !validSeccomp(podSpec.SecurityContext.SeccompProfile.Type) {
				forbidden.Insert(fieldValue(podSeccompField, string(seccompType)))
			} else {
				podSeccompSet = true
			}
		}
	}

	visitContainersWithPath(podSpec, field.NewPath("spec"), func(c *corev1.Container, path *field.Path) {
		if c.SecurityContext != nil && c.SecurityContext.SeccompProfile != nil {
			seccompType := c.SecurityContext.SeccompProfile.Type
			if !validSeccomp(seccompType) {
				containerSeccompField := path.Child("securityContext", "seccompProfile", "type")
				forbidden.Insert(fieldValue(containerSeccompField, string(seccompType)))
			}
			return
		}

		if !podSeccompSet {
			containerSeccompField := path.Child("securityContext", "seccompProfile", "type")
			forbidden.Insert(fieldValueRequired(containerSeccompField))
		}
	})

	if len(forbidden) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccomp profile",
			ForbiddenDetail: strings.Join(forbidden.List(), ", "),
		}
	}

	return CheckResult{Allowed: true}
}
