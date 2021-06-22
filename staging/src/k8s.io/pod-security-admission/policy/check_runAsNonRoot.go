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
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
Containers must be required to run as non-root users.

**Restricted Fields:**

spec.securityContext.runAsNonRoot
spec.containers[*].securityContext.runAsNonRoot
spec.initContainers[*].securityContext.runAsNonRoot

**Allowed Values:** true
*/

func init() {
	addCheck(CheckRunAsNonRoot)
}

// CheckRunAsNonRoot returns a restricted level check
// that requires runAsNonRoot=true in 1.0+
func CheckRunAsNonRoot() Check {
	return Check{
		ID:    "runAsNonRoot",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       runAsNonRoot_1_0,
			},
		},
	}
}

func runAsNonRoot_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var forbiddenPaths []string

	// TODO: how to check ephemeral containers

	containerCount := 0
	containerRunAsNonRootCount := 0
	podRunAsNonRoot := false

	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		containerCount++
		if container.SecurityContext != nil && container.SecurityContext.RunAsNonRoot != nil {
			if !*container.SecurityContext.RunAsNonRoot {
				forbiddenPaths = append(forbiddenPaths, path.Child("securityContext", "runAsNonRoot").String())
			} else {
				containerRunAsNonRootCount++
			}
		}
	})

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.RunAsNonRoot != nil {
		if !*podSpec.SecurityContext.RunAsNonRoot {
			forbiddenPaths = append(forbiddenPaths, field.NewPath("spec").Child("securityContext", "runAsNonRoot").String())
		} else {
			podRunAsNonRoot = true
		}
	}

	// pod or containers explicitly set runAsNonRoot=false
	if len(forbiddenPaths) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "runAsNonRoot != false",
			ForbiddenDetail: strings.Join(forbiddenPaths, ", "),
		}
	}

	// pod didn't set runAsNonRoot and not all containers opted into runAsNonRoot
	if podRunAsNonRoot == false && containerCount > containerRunAsNonRootCount {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "runAsNonRoot != false",
		}
	}

	return CheckResult{Allowed: true}
}
