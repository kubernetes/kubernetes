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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
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
	addCheck(CheckSELinux)
}

// CheckSELinux returns a baseline level check
// that limits seLinuxOptions type, user, and role values in 1.0+
func CheckSELinux() Check {
	return Check{
		ID:    "selinux",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       checkSelinux_1_0,
			},
		},
	}
}

var (
	selinux_allowed_types_1_0 = sets.NewString("", "container_t", "container_init_t", "container_kvm_t")
)

func checkSelinux_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var forbiddenDetails []string

	checkSelinuxOptions := func(path *field.Path, opts *corev1.SELinuxOptions) {
		if !selinux_allowed_types_1_0.Has(opts.Type) {
			forbiddenDetails = append(forbiddenDetails, path.Child("securityContext", "seLinuxOptions", "type").String())
		}
		if len(opts.User) > 0 {
			forbiddenDetails = append(forbiddenDetails, path.Child("securityContext", "seLinuxOptions", "user").String())
		}
		if len(opts.Role) > 0 {
			forbiddenDetails = append(forbiddenDetails, path.Child("securityContext", "seLinuxOptions", "role").String())
		}
	}

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.SELinuxOptions != nil {
		checkSelinuxOptions(field.NewPath("spec"), podSpec.SecurityContext.SELinuxOptions)
	}

	visitContainersWithPath(podSpec, field.NewPath("spec"), func(container *corev1.Container, path *field.Path) {
		if container.SecurityContext != nil && container.SecurityContext.SELinuxOptions != nil {
			checkSelinuxOptions(path, container.SecurityContext.SELinuxOptions)
		}
	})

	if len(forbiddenDetails) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "forbidden seLinuxOptions",
			ForbiddenDetail: strings.Join(forbiddenDetails, ", "),
		}
	}
	return CheckResult{Allowed: true}
}
