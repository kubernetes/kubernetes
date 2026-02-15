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
Container ulimits are only allowed by the Privileged policy level.

**Restricted Fields:**
spec.containers[*].securityContext.ulimits
spec.initContainers[*].securityContext.ulimits
spec.ephemeralContainers[*].securityContext.ulimits

**Allowed Values:** undefined/null
*/

func init() {
	addCheck(CheckUlimitsBaseline)
}

// CheckUlimitsBaseline returns a baseline level check that forbids setting securityContext.ulimits in 1.0+.
func CheckUlimitsBaseline() Check {
	return Check{
		ID:    "ulimits",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       ulimitsBaseline_1_0,
			},
		},
	}
}

func ulimitsBaseline_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badContainers []string
	visitContainers(podSpec, func(container *corev1.Container) {
		if container.SecurityContext != nil && len(container.SecurityContext.Ulimits) > 0 {
			badContainers = append(badContainers, container.Name)
		}
	})
	if len(badContainers) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "ulimits",
			ForbiddenDetail: fmt.Sprintf(
				`%s %s must not set securityContext.ulimits`,
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
			),
		}
	}
	return CheckResult{Allowed: true}
}
