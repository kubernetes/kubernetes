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
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/pod-security-admission/api"
)

/*
Containers must not run as hostProcess.

**Restricted Fields:**

spec.securityContext.windowsOptions.hostProcess
spec.containers[*].securityContext.windowsOptions.hostProcess

**Allowed Values:** undefined / false
*/

func init() {
	addCheck(CheckHostProcess)
}

// CheckHostProcess returns a baseline level check
// that forbids hostProcess=true in 1.0+
func CheckHostProcess() Check {
	return Check{
		ID:    "hostProcess",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       hostProcess_1_0,
			},
		},
	}
}

func hostProcess_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var forbiddenContainers []string
	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext != nil &&
			container.SecurityContext.WindowsOptions != nil &&
			container.SecurityContext.WindowsOptions.HostProcess != nil &&
			*container.SecurityContext.WindowsOptions.HostProcess {
			forbiddenContainers = append(forbiddenContainers, container.Name)
		}
	})

	podSpecForbidden := false
	if podSpec.SecurityContext != nil &&
		podSpec.SecurityContext.WindowsOptions != nil &&
		podSpec.SecurityContext.WindowsOptions.HostProcess != nil &&
		*podSpec.SecurityContext.WindowsOptions.HostProcess {
		podSpecForbidden = true
	}

	// pod or containers explicitly set hostProcess=true
	if len(forbiddenContainers) > 0 || podSpecForbidden {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostProcess == true",
			ForbiddenDetail: strings.Join(forbiddenContainers, ", "),
		}
	}

	return CheckResult{Allowed: true}
}
