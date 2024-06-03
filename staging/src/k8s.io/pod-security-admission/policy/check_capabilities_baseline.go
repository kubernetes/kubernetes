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

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
Adding NET_RAW or capabilities beyond the default set must be disallowed.

**Restricted Fields:**
spec.containers[*].securityContext.capabilities.add
spec.initContainers[*].securityContext.capabilities.add

**Allowed Values:**
undefined / empty
values from the default set "AUDIT_WRITE", "CHOWN", "DAC_OVERRIDE","FOWNER", "FSETID", "KILL", "MKNOD", "NET_BIND_SERVICE", "SETFCAP", "SETGID", "SETPCAP", "SETUID", "SYS_CHROOT"
*/

func init() {
	addCheck(CheckCapabilitiesBaseline)
}

const checkCapabilitiesBaselineID CheckID = "capabilities_baseline"

// CheckCapabilitiesBaseline returns a baseline level check
// that limits the capabilities that can be added in 1.0+
func CheckCapabilitiesBaseline() Check {
	return Check{
		ID:    checkCapabilitiesBaselineID,
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(capabilitiesBaselineV1Dot0),
			},
		},
	}
}

var (
	capabilities_allowed_1_0 = sets.NewString(
		"AUDIT_WRITE",
		"CHOWN",
		"DAC_OVERRIDE",
		"FOWNER",
		"FSETID",
		"KILL",
		"MKNOD",
		"NET_BIND_SERVICE",
		"SETFCAP",
		"SETGID",
		"SETPCAP",
		"SETUID",
		"SYS_CHROOT",
	)
)

func capabilitiesBaselineV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	badContainers := NewViolations(opts.withFieldErrors)
	nonDefaultCapabilities := sets.NewString()
	visitContainers(podSpec, opts, func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext != nil && container.SecurityContext.Capabilities != nil {
			valid := true
			if opts.withFieldErrors {
				forbiddenValue := sets.NewString()
				for _, c := range container.SecurityContext.Capabilities.Add {
					if !capabilities_allowed_1_0.Has(string(c)) {
						valid = false
						nonDefaultCapabilities.Insert(string(c))
						forbiddenValue.Insert(string(c))
					}
				}
				if !valid {
					badContainers.Add(container.Name, withBadValue(forbidden(path.Child("securityContext", "capabilities", "add")), forbiddenValue.List()))
				}
			} else {
				for _, c := range container.SecurityContext.Capabilities.Add {
					if !capabilities_allowed_1_0.Has(string(c)) {
						valid = false
						nonDefaultCapabilities.Insert(string(c))
					}
				}
				if !valid {
					badContainers.Add(container.Name)
				}
			}
		}
	})

	if !badContainers.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "non-default capabilities",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s must not include %s in securityContext.capabilities.add",
				pluralize("container", "containers", badContainers.Len()),
				joinQuote(badContainers.Data()),
				joinQuote(nonDefaultCapabilities.List()),
			),
			ErrList: badContainers.Errs(),
		}
	}
	return CheckResult{Allowed: true}
}
