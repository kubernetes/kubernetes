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
				MinimumVersion:   api.MajorMinorVersion(1, 19),
				CheckPod:         withOptions(seccompProfileRestrictedV1Dot19),
				OverrideCheckIDs: []CheckID{checkSeccompBaselineID},
			},
			// Starting 1.25, windows pods would be exempted from this check using pod.spec.os field when set to windows.
			{
				MinimumVersion:   api.MajorMinorVersion(1, 25),
				CheckPod:         withOptions(seccompProfileRestrictedV1Dot25),
				OverrideCheckIDs: []CheckID{checkSeccompBaselineID},
			},
		},
	}
}

// seccompProfileRestrictedV1Dot19 checks restricted policy on securityContext.seccompProfile field
func seccompProfileRestrictedV1Dot19(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	// things that explicitly set seccompProfile.type to a bad value
	badSetters := NewViolations(opts.withFieldErrors)
	badValues := sets.NewString()

	podSeccompSet := false

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.SeccompProfile != nil {
		if !validSeccomp(podSpec.SecurityContext.SeccompProfile.Type) {
			if opts.withFieldErrors {
				badSetters.Add("pod", withBadValue(forbidden(seccompProfileTypePath), string(podSpec.SecurityContext.SeccompProfile.Type)))
			} else {
				badSetters.Add("pod")
			}
			badValues.Insert(string(podSpec.SecurityContext.SeccompProfile.Type))
		} else {
			podSeccompSet = true
		}
	}

	// containers that explicitly set seccompProfile.type to a bad value
	explicitlyBadContainers := NewViolations(opts.withFieldErrors)
	// containers that didn't set seccompProfile and aren't caught by a pod-level seccompProfile
	implicitlyBadContainers := NewViolations(opts.withFieldErrors)
	var explicitlyErrs field.ErrorList

	visitContainers(podSpec, opts, func(c *corev1.Container, path *field.Path) {
		if c.SecurityContext != nil && c.SecurityContext.SeccompProfile != nil {
			// container explicitly set seccompProfile
			if !validSeccomp(c.SecurityContext.SeccompProfile.Type) {
				// container explicitly set seccompProfile to a bad value
				explicitlyBadContainers.Add(c.Name)
				if opts.withFieldErrors {
					explicitlyErrs = append(explicitlyErrs, withBadValue(forbidden(path.Child("securityContext", "seccompProfile", "type")), string(c.SecurityContext.SeccompProfile.Type)))
				}
				badValues.Insert(string(c.SecurityContext.SeccompProfile.Type))
			}
		} else {
			// container did not explicitly set seccompProfile
			if !podSeccompSet {
				// no valid pod-level seccompProfile, so this container implicitly has a bad value
				if opts.withFieldErrors {
					implicitlyBadContainers.Add(c.Name, required(path.Child("securityContext", "seccompProfile", "type")))
				} else {
					implicitlyBadContainers.Add(c.Name)
				}
			}
		}
	})

	if !explicitlyBadContainers.Empty() {
		badSetters.Add(
			fmt.Sprintf(
				"%s %s",
				pluralize("container", "containers", explicitlyBadContainers.Len()),
				joinQuote(explicitlyBadContainers.Data()),
			),
			explicitlyErrs...,
		)
	}
	// pod or containers explicitly set bad seccompProfiles
	if !badSetters.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccompProfile",
			ForbiddenDetail: fmt.Sprintf(
				"%s must not set securityContext.seccompProfile.type to %s",
				strings.Join(badSetters.Data(), " and "),
				joinQuote(badValues.List()),
			),
			ErrList: badSetters.Errs(),
		}
	}

	// pod didn't set seccompProfile and not all containers opted into seccompProfile
	if !implicitlyBadContainers.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccompProfile",
			ForbiddenDetail: fmt.Sprintf(
				`pod or %s %s must set securityContext.seccompProfile.type to "RuntimeDefault" or "Localhost"`,
				pluralize("container", "containers", implicitlyBadContainers.Len()),
				joinQuote(implicitlyBadContainers.Data()),
			),
			ErrList: implicitlyBadContainers.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}

// seccompProfileRestrictedV1Dot25 checks restricted policy on securityContext.seccompProfile field for kubernetes
// version 1.25 and above
func seccompProfileRestrictedV1Dot25(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	// Pod API validation would have failed if podOS == Windows and if secCompProfile has been set.
	// We can admit the Windows pod even if seccompProfile has not been set.
	if podSpec.OS != nil && podSpec.OS.Name == corev1.Windows {
		return CheckResult{Allowed: true}
	}
	return seccompProfileRestrictedV1Dot19(podMetadata, podSpec, opts)
}
