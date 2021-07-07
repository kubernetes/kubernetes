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

const (
	capabilityAll            = "ALL"
	capabilityNetBindService = "CAP_NET_BIND_SERVICE"
)

func init() {
	addCheck(CheckDropCapabilities)
}

// CheckDropCapabilities returns a restricted level check
// that ensures all capabilities are dropped in 1.22+
func CheckDropCapabilities() Check {
	return Check{
		ID:    "dropCapabilities",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 22),
				CheckPod:       dropCapabilities_1_22,
			},
		},
	}
}

func dropCapabilities_1_22(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	containers := sets.NewString()
	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext == nil || container.SecurityContext.Capabilities == nil {
			containers.Insert(container.Name)
			return
		}
		found := false
		for _, c := range container.SecurityContext.Capabilities.Drop {
			if c == capabilityAll {
				found = true
				break
			}
		}
		if !found {
			containers.Insert(container.Name)
			return
		}
		for index, c := range container.SecurityContext.Capabilities.Add {
			if c != capabilityNetBindService {
				capabilityPath := path.Child("securityContext", "capabilities", "add").Index(index)
				msg := fmt.Sprintf("%s=%s", capabilityPath.String(), string(c))
				containers.Insert(msg)
			}
		}

	})
	if len(containers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "containers must drop ALL capability",
			ForbiddenDetail: strings.Join(containers.List(), ", "),
		}
	}
	return CheckResult{Allowed: true}
}
