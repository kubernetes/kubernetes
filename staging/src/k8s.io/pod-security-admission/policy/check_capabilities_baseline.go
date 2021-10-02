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

// CheckCapabilitiesBaseline returns a baseline level check
// that limits the capabilities that can be added in 1.0+
func CheckCapabilitiesBaseline() Check {
	return Check{
		ID:    "capabilities_baseline",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       capabilitiesBaseline_1_0,
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

func capabilitiesBaseline_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	nonDefaultCapabilities := sets.NewString()
	visitContainers(podSpec, func(container *corev1.Container) {
		if container.SecurityContext != nil && container.SecurityContext.Capabilities != nil {
			valid := true
			for _, c := range container.SecurityContext.Capabilities.Add {
				if !capabilities_allowed_1_0.Has(string(c)) {
					valid = false
					nonDefaultCapabilities.Insert(string(c))
				}
			}
			if !valid {
				badContainers = append(badContainers, container.Name)
			}
		}
	})

	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "non-default capabilities",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s must not include %s in securityContext.capabilities.add",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
				joinQuote(nonDefaultCapabilities.List()),
			),
		}
	}
	return CheckResult{Allowed: true}
}
