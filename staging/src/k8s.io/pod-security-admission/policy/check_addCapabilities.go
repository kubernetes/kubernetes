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

func init() {
	addCheck(CheckAddCapabilities)
}

// CheckAddCapabilities returns a baseline level check
// that limits the capabilities that can be added in 1.0+
func CheckAddCapabilities() Check {
	return Check{
		ID:    "addCapabilities",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       addCapabilities_1_0,
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

func addCapabilities_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	forbiddenContainers := sets.NewString()
	forbiddenCapabilities := sets.NewString()
	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext != nil && container.SecurityContext.Capabilities != nil {
			for _, c := range container.SecurityContext.Capabilities.Add {
				if !capabilities_allowed_1_0.Has(string(c)) {
					forbiddenContainers.Insert(container.Name)
					forbiddenCapabilities.Insert(string(c))
				}
			}
		}
	})

	if len(forbiddenCapabilities) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "forbidden capabilities",
			ForbiddenDetail: fmt.Sprintf(
				"containers %q added %q",
				forbiddenContainers.List(),
				forbiddenCapabilities.List(),
			),
		}
	}
	return CheckResult{Allowed: true}
}
