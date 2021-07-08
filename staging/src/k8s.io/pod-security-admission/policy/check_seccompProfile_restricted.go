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
	"k8s.io/pod-security-admission/api"
)

/*

Seccomp profiles must be specified, and only runtime default and localhost profiles are allowed.

v1.19+:
**Restricted Fields:**
spec.securityContext.seccompProfile.type
spec.containers[*].securityContext.seccompProfile.type
spec.initContainers[*].securityContext.seccompProfile.type

**Allowed Values:** 'RuntimeDefault', 'Localhost'
Note: container-level fields may be undefined if pod-level field is specified.

*/

func init() {
	addCheck(CheckSeccompProfileRestricted)
}

func CheckSeccompProfileRestricted() Check {
	return Check{
		ID:    "seccompProfile_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 19),
				CheckPod:       seccompProfileRestricted_1_19,
			},
		},
	}
}

// seccompProfileRestricted_1_19 checks restricted policy on securityContext.seccompProfile field
func seccompProfileRestricted_1_19(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	// things that explicitly set seccompProfile.type to a bad value
	var badSetters []string
	badValues := sets.NewString()

	podSeccompSet := false

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.SeccompProfile != nil {
		if !validSeccomp(podSpec.SecurityContext.SeccompProfile.Type) {
			badSetters = append(badSetters, "pod")
			badValues.Insert(string(podSpec.SecurityContext.SeccompProfile.Type))
		} else {
			podSeccompSet = true
		}
	}

	// containers that explicitly set seccompProfile.type to a bad value
	var explicitlyBadContainers []string
	// containers that didn't set seccompProfile and aren't caught by a pod-level seccompProfile
	var implicitlyBadContainers []string

	visitContainers(podSpec, func(c *corev1.Container) {
		if c.SecurityContext != nil && c.SecurityContext.SeccompProfile != nil {
			// container explicitly set seccompProfile
			if !validSeccomp(c.SecurityContext.SeccompProfile.Type) {
				// container explicitly set seccompProfile to a bad value
				explicitlyBadContainers = append(explicitlyBadContainers, c.Name)
				badValues.Insert(string(c.SecurityContext.SeccompProfile.Type))
			}
		} else {
			// container did not explicitly set seccompProfile
			if !podSeccompSet {
				// no valid pod-level seccompProfile, so this container implicitly has a bad value
				implicitlyBadContainers = append(implicitlyBadContainers, c.Name)
			}
		}
	})

	if len(explicitlyBadContainers) > 0 {
		badSetters = append(
			badSetters,
			fmt.Sprintf(
				"%s %s",
				pluralize("container", "containers", len(explicitlyBadContainers)),
				joinQuote(explicitlyBadContainers),
			),
		)
	}
	// pod or containers explicitly set bad seccompProfiles
	if len(badSetters) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccompProfile",
			ForbiddenDetail: fmt.Sprintf(
				"%s must not set securityContext.seccompProfile.type to %s",
				strings.Join(badSetters, " and "),
				joinQuote(badValues.List()),
			),
		}
	}

	// pod didn't set seccompProfile and not all containers opted into seccompProfile
	if len(implicitlyBadContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccompProfile",
			ForbiddenDetail: fmt.Sprintf(
				`pod or %s %s must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
				pluralize("container", "containers", len(implicitlyBadContainers)),
				joinQuote(implicitlyBadContainers),
			),
		}
	}

	return CheckResult{Allowed: true}
}
