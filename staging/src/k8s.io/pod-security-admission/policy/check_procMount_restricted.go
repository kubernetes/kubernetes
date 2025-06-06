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

The default /proc masks are set up to reduce attack surface, and should be required.

**Restricted Fields:**
spec.containers[*].securityContext.procMount
spec.initContainers[*].securityContext.procMount

**Allowed Values:** undefined/null, "Default"

*/

func init() {
	addCheck(CheckProcMountRestricted)
}

// CheckProcMountRestricted returns a restricted level check that restricts.
func CheckProcMountRestricted() Check {
	return Check{
		ID:    "procMount_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			// Since the UserNamespacesPodSecurityStandards feature has been dropped,
			// there's no functional difference between 1.0 and 1.35, but this is kept around
			// for backwards compatibility.
			{
				MinimumVersion: api.MajorMinorVersion(1, 35),
				CheckPod:       procMount_1_35_restricted,
			},
		},
	}
}

// If the pod is in a restricted namespace, always block if ProcMount is set to Unmasked.
func procMount_1_35_restricted(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	forbiddenProcMountTypes := sets.NewString()
	visitContainers(podSpec, func(container *corev1.Container) {
		// allow if the security context is nil.
		if container.SecurityContext == nil {
			return
		}
		// allow if proc mount is not set.
		if container.SecurityContext.ProcMount == nil {
			return
		}
		// check if the value of the proc mount type is valid.
		if *container.SecurityContext.ProcMount != corev1.DefaultProcMount {
			badContainers = append(badContainers, container.Name)
			forbiddenProcMountTypes.Insert(string(*container.SecurityContext.ProcMount))
		}
	})
	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "procMount",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s must not set securityContext.procMount to %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
				joinQuote(forbiddenProcMountTypes.List()),
			),
		}
	}
	return CheckResult{Allowed: true}
}
