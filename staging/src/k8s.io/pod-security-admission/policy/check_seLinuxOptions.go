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
Setting the SELinux type is restricted, and setting a custom SELinux user or role option is forbidden.

**Restricted Fields:**
spec.securityContext.seLinuxOptions.type
spec.containers[*].securityContext.seLinuxOptions.type
spec.initContainers[*].securityContext.seLinuxOptions.type

**Allowed Values:**
undefined/empty
container_t
container_init_t
container_kvm_t

**Restricted Fields:**
spec.securityContext.seLinuxOptions.user
spec.containers[*].securityContext.seLinuxOptions.user
spec.initContainers[*].securityContext.seLinuxOptions.user
spec.securityContext.seLinuxOptions.role
spec.containers[*].securityContext.seLinuxOptions.role
spec.initContainers[*].securityContext.seLinuxOptions.role

**Allowed Values:** undefined/empty
*/

func init() {
	addCheck(CheckSELinuxOptions)
}

// CheckSELinuxOptions returns a baseline level check
// that limits seLinuxOptions type, user, and role values in 1.0+
func CheckSELinuxOptions() Check {
	return Check{
		ID:    "seLinuxOptions",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(seLinuxOptionsV1Dot0),
			},
		},
	}
}

var (
	selinux_allowed_types_1_0 = sets.NewString("", "container_t", "container_init_t", "container_kvm_t")
)

func seLinuxOptionsV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	var (
		// sources that set bad seLinuxOptions
		badSetters          = NewViolations(opts.withFieldErrors)
		badContainersErrFns []ErrFn
		badPodErrFns        []ErrFn
		// invalid type values set
		badTypes = sets.NewString()
		// was user set?
		setUser = false
		// was role set?
		setRole = false
	)

	validSELinuxOptions := func(selinuxOpts *corev1.SELinuxOptions, pathFn PathFn, isPodLevel bool) bool {
		valid := true
		if !selinux_allowed_types_1_0.Has(selinuxOpts.Type) {
			valid = false
			badTypes.Insert(selinuxOpts.Type)
			if pathFn != nil {
				badContainersErrFns = append(badContainersErrFns, forbidden(pathFn.child("securityContext", "seLinuxOptions", "type")).withBadValue(selinuxOpts.Type))
			} else if isPodLevel && opts.withFieldErrors {
				badPodErrFns = append(badPodErrFns, forbidden(seLinuxOptionsTypePath).withBadValue(selinuxOpts.Type))
			}
		}
		if len(selinuxOpts.User) > 0 {
			valid = false
			setUser = true
			if pathFn != nil {
				badContainersErrFns = append(badContainersErrFns, forbidden(pathFn.child("securityContext", "seLinuxOptions", "user")).withBadValue(selinuxOpts.User))
			} else if isPodLevel && opts.withFieldErrors {
				badPodErrFns = append(badPodErrFns, forbidden(seLinuxOptionsUserPath).withBadValue(selinuxOpts.User))
			}
		}
		if len(selinuxOpts.Role) > 0 {
			valid = false
			setRole = true
			if pathFn != nil {
				badContainersErrFns = append(badContainersErrFns, forbidden(pathFn.child("securityContext", "seLinuxOptions", "role")).withBadValue(selinuxOpts.Role))
			} else if isPodLevel && opts.withFieldErrors {
				badPodErrFns = append(badPodErrFns, forbidden(seLinuxOptionsRolePath).withBadValue(selinuxOpts.Role))
			}
		}
		return valid
	}

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.SELinuxOptions != nil {
		if !validSELinuxOptions(podSpec.SecurityContext.SELinuxOptions, nil, true) {
			badSetters.Add("pod", badPodErrFns...)
		}
	}

	var badContainers []string
	visitContainers(podSpec, opts, func(container *corev1.Container, pathFn PathFn) {
		if container.SecurityContext != nil && container.SecurityContext.SELinuxOptions != nil {
			if !validSELinuxOptions(container.SecurityContext.SELinuxOptions, pathFn, false) {
				badContainers = append(badContainers, container.Name)
			}
		}
	})

	if len(badContainers) > 0 {
		badSetters.Add(
			fmt.Sprintf(
				"%s %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
			),
			badContainersErrFns...,
		)
	}

	if !badSetters.Empty() {
		var badData []string
		if len(badTypes) > 0 {
			badData = append(badData, fmt.Sprintf(
				"%s %s",
				pluralize("type", "types", len(badTypes)),
				joinQuote(badTypes.List()),
			))
		}
		if setUser {
			badData = append(badData, "user may not be set")
		}
		if setRole {
			badData = append(badData, "role may not be set")
		}

		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "seLinuxOptions",
			ForbiddenDetail: fmt.Sprintf(
				`%s set forbidden securityContext.seLinuxOptions: %s`,
				strings.Join(badSetters.Data(), " and "),
				strings.Join(badData, "; "),
			),
			ErrList: badSetters.Errs(),
		}
	}
	return CheckResult{Allowed: true}
}
