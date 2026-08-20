/*
Copyright The Kubernetes Authors.

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
	"k8s.io/pod-security-admission/api"
)

/*

Writable cgroups grant a container read-write access to its own cgroup subtree, an
elevated capability that increases attack surface and should not be allowed by the
restricted profile.

**Restricted Fields:**
spec.containers[*].securityContext.cgroupOptions.mountMode
spec.initContainers[*].securityContext.cgroupOptions.mountMode
spec.ephemeralContainers[*].securityContext.cgroupOptions.mountMode

**Allowed Values:** undefined/null, "ReadOnly"

*/

func init() {
	addCheck(CheckCgroupOptionsRestricted)
}

// CheckCgroupOptionsRestricted returns a restricted level check that forbids writable cgroups.
func CheckCgroupOptionsRestricted() Check {
	return Check{
		ID:    "cgroupOptions_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 38),
				CheckPod:       cgroupOptionsRestricted1_38,
			},
		},
	}
}

func cgroupOptionsRestricted1_38(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	visitContainers(podSpec, func(container *corev1.Container) {
		if container.SecurityContext == nil || container.SecurityContext.CgroupOptions == nil {
			return
		}
		if container.SecurityContext.CgroupOptions.MountMode == nil {
			return
		}
		if *container.SecurityContext.CgroupOptions.MountMode == corev1.CgroupMountModeWritable {
			badContainers = append(badContainers, container.Name)
		}
	})

	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "cgroupOptions",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s must not set securityContext.cgroupOptions.mountMode to %q",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
				string(corev1.CgroupMountModeWritable),
			),
		}
	}
	return CheckResult{Allowed: true}
}
