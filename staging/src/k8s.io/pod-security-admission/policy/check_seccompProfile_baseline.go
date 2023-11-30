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
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
If seccomp profiles are specified, only runtime default and localhost profiles are allowed.

v1.0 - v1.18:
**Restricted Fields:**
metadata.annotations['seccomp.security.alpha.kubernetes.io/pod']
metadata.annotations['container.seccomp.security.alpha.kubernetes.io/*']

**Allowed Values:** 'runtime/default', 'docker/default', 'localhost/*', undefined

v1.19+:
**Restricted Fields:**
spec.securityContext.seccompProfile.type
spec.containers[*].securityContext.seccompProfile.type
spec.initContainers[*].securityContext.seccompProfile.type

**Allowed Values:** 'RuntimeDefault', 'Localhost', undefined
*/
const (
	annotationKeyPod             = "seccomp.security.alpha.kubernetes.io/pod"
	annotationKeyContainerPrefix = "container.seccomp.security.alpha.kubernetes.io/"

	checkSeccompBaselineID CheckID = "seccompProfile_baseline"
)

func init() {
	addCheck(CheckSeccompBaseline)
}

func CheckSeccompBaseline() Check {
	return Check{
		ID:    checkSeccompBaselineID,
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(seccompProfileBaselineV1Dot0),
			},
			{
				MinimumVersion: api.MajorMinorVersion(1, 19),
				CheckPod:       withOptions(seccompProfileBaselineV1Dot19),
			},
		},
	}
}

func validSeccomp(t corev1.SeccompProfileType) bool {
	return t == corev1.SeccompProfileTypeLocalhost ||
		t == corev1.SeccompProfileTypeRuntimeDefault
}

func validSeccompAnnotationValue(v string) bool {
	return v == corev1.SeccompProfileRuntimeDefault ||
		v == corev1.DeprecatedSeccompProfileDockerDefault ||
		strings.HasPrefix(v, corev1.SeccompLocalhostProfileNamePrefix)
}

// seccompProfileBaselineV1Dot0 checks baseline policy on seccomp alpha annotation
func seccompProfileBaselineV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	m := map[string]field.ErrorList{}
	badSetters := NewViolations(opts.withFieldErrors)

	if val, ok := podMetadata.Annotations[annotationKeyPod]; ok {
		if !validSeccompAnnotationValue(val) {
			forbiddenValue := fmt.Sprintf("%s=%q", annotationKeyPod, val)
			m[forbiddenValue] = append(m[forbiddenValue], withBadValue(forbidden(annotationsPath.Key(annotationKeyPod)), val))
		}
	}

	visitContainers(podSpec, opts, func(c *corev1.Container, path *field.Path) {
		annotation := annotationKeyContainerPrefix + c.Name
		if val, ok := podMetadata.Annotations[annotation]; ok {
			if !validSeccompAnnotationValue(val) {
				forbiddenValue := fmt.Sprintf("%s=%q", annotation, val)
				m[forbiddenValue] = append(m[forbiddenValue], withBadValue(forbidden(annotationsPath.Key(annotation)), val))
			}
		}
	})

	for forbiddenValue, errFns := range m {
		badSetters.Add(forbiddenValue, errFns...)
	}

	if !badSetters.Empty() {
		forbiddenValues := badSetters.Data()
		sort.Strings(forbiddenValues)
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seccompProfile",
			ForbiddenDetail: fmt.Sprintf(
				"forbidden %s %s",
				pluralize("annotation", "annotations", len(forbiddenValues)),
				strings.Join(forbiddenValues, ", "),
			),
			ErrList: badSetters.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}

// seccompProfileBaselineV1Dot19 checks baseline policy on securityContext.seccompProfile field
func seccompProfileBaselineV1Dot19(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	// things that explicitly set seccompProfile.type to a bad value
	badSetters := NewViolations(opts.withFieldErrors)
	badValues := sets.NewString()

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.SeccompProfile != nil {
		if !validSeccomp(podSpec.SecurityContext.SeccompProfile.Type) {
			var err *field.Error
			if opts.withFieldErrors {
				err = withBadValue(forbidden(seccompProfileTypePath), string(podSpec.SecurityContext.SeccompProfile.Type))
			}
			badSetters.Add("pod", err)
			badValues.Insert(string(podSpec.SecurityContext.SeccompProfile.Type))
		}
	}

	// containers that explicitly set seccompProfile.type to a bad value
	explicitlyBadContainers := NewViolations(opts.withFieldErrors)
	var explicitlyErrs field.ErrorList

	visitContainers(podSpec, opts, func(c *corev1.Container, path *field.Path) {
		if c.SecurityContext != nil && c.SecurityContext.SeccompProfile != nil {
			// container explicitly set seccompProfile
			if !validSeccomp(c.SecurityContext.SeccompProfile.Type) {
				// container explicitly set seccompProfile to a bad value
				explicitlyBadContainers.Add(c.Name)
				explicitlyErrs = append(explicitlyErrs, withBadValue(forbidden(path.Child("securityContext", "seccompProfile", "type")), string(c.SecurityContext.SeccompProfile.Type)))
				badValues.Insert(string(c.SecurityContext.SeccompProfile.Type))
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

	return CheckResult{Allowed: true}
}
