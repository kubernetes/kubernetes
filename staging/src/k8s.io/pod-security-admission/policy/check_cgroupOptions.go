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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

/*

The Restricted profile requires cgroupOptions.mountMode to be unset or "ReadOnly".

**Restricted Fields:**
spec.containers[*].securityContext.cgroupOptions.mountMode
spec.initContainers[*].securityContext.cgroupOptions.mountMode
spec.ephemeralContainers[*].securityContext.cgroupOptions.mountMode

**Allowed Values:** undefined/null, "ReadOnly"

*/

func init() {
	addCheck(CheckCgroupOptions)
}

// CheckCgroupOptions returns a restricted level check that forbids writable cgroups.
func CheckCgroupOptions() Check {
	return Check{
		ID:    "cgroupOptions",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 38),
				CheckPod:       cgroupOptions1_38,
			},
		},
	}
}

func cgroupOptions1_38(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	forbiddenMountModes := sets.New[string]()
	visitContainers(podSpec, func(container *corev1.Container) {
		if container.SecurityContext == nil || container.SecurityContext.CgroupOptions == nil {
			return
		}
		mountMode := container.SecurityContext.CgroupOptions.MountMode
		if mountMode == nil || *mountMode == corev1.CgroupMountModeReadOnly {
			return
		}
		badContainers = append(badContainers, container.Name)
		forbiddenMountModes.Insert(string(*mountMode))
	})

	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "cgroupOptions",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s must not set securityContext.cgroupOptions.mountMode to %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
				joinQuote(sets.List(forbiddenMountModes)),
			),
		}
	}
	return CheckResult{Allowed: true}
}
