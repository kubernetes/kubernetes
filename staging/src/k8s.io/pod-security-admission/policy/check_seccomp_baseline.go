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
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

const (
	annotationKeyPod             = "seccomp.security.alpha.kubernetes.io/pod"
	annotationKeyContainerPrefix = "container.seccomp.security.alpha.kubernetes.io/"
	missingRequiredValue         = "<missing required value>"
)

func init() {
	addCheck(CheckSeccompBaseline)
}

func fieldValue(f *field.Path, val string) string {
	return fmt.Sprintf("%s=%s", f.String(), val)
}

func fieldValueRequired(f *field.Path) string {
	return fmt.Sprintf("%s=%s", f.String(), missingRequiredValue)
}

func CheckSeccompBaseline() Check {
	return Check{
		ID:    "seccomp_baseline",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       seccomp_1_0_baseline,
			},
			{
				MinimumVersion: api.MajorMinorVersion(1, 19),
				CheckPod:       seccomp_1_19_baseline,
			},
		},
	}
}

func validSeccomp(t corev1.SeccompProfileType) bool {
	return t == corev1.SeccompProfileTypeLocalhost ||
		t == corev1.SeccompProfileTypeRuntimeDefault
}

// seccomp_1_0_baseline checks baseline policy on seccomp alpha annotation
func seccomp_1_0_baseline(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	forbidden := sets.NewString()

	if val, ok := podMetadata.Annotations[annotationKeyPod]; ok {
		if val == corev1.SeccompProfileNameUnconfined {
			podAnnotationField := field.NewPath("metadata").Child("annotations", annotationKeyPod)
			forbidden.Insert(fieldValue(podAnnotationField, val))
		}
	}

	visitContainersWithPath(podSpec, field.NewPath("spec"), func(c *corev1.Container, path *field.Path) {
		annotation := annotationKeyContainerPrefix + c.Name
		if val, ok := podMetadata.Annotations[annotation]; ok {
			if val == corev1.SeccompProfileNameUnconfined {
				containerAnnotationField := field.NewPath("metadata").
					Child("annotations", annotation)
				forbidden.Insert(fieldValue(containerAnnotationField, val))
			}
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

// seccomp_1_19_baseline checks baseline policy on securityContext.seccompProfile field
func seccomp_1_19_baseline(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	forbidden := sets.NewString()

	if podSpec.SecurityContext != nil {
		if podSpec.SecurityContext.SeccompProfile != nil {
			seccompType := podSpec.SecurityContext.SeccompProfile.Type
			if !validSeccomp(seccompType) {
				podSeccompField := field.NewPath("spec").Child("securityContext", "seccompProfile", "type")
				forbidden.Insert(fieldValue(podSeccompField, string(seccompType)))
			}
		}
	}

	visitContainersWithPath(podSpec, field.NewPath("spec"), func(c *corev1.Container, path *field.Path) {
		if c.SecurityContext != nil {
			if c.SecurityContext.SeccompProfile != nil {
				if c.SecurityContext.SeccompProfile.Type != "" {
					seccompType := c.SecurityContext.SeccompProfile.Type
					if !validSeccomp(seccompType) {
						containerSeccompField := path.Child("securityContext", "seccompProfile", "type")
						forbidden.Insert(fieldValue(containerSeccompField, string(seccompType)))
					}
				}
			}
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
