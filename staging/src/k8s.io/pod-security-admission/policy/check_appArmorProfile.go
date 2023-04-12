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
	"sort"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"
)

/*
On supported hosts, the 'runtime/default' AppArmor profile is applied by default.
The baseline policy should prevent overriding or disabling the default AppArmor
profile, or restrict overrides to an allowed set of profiles.

**Restricted Fields:**
metadata.annotations['container.apparmor.security.beta.kubernetes.io/*']

**Allowed Values:** 'runtime/default', 'localhost/*', empty, undefined
*/
func init() {
	addCheck(CheckAppArmorProfile)
}

// CheckAppArmorProfile returns a baseline level check
// that limits the value of AppArmor profiles in 1.0+
func CheckAppArmorProfile() Check {
	return Check{
		ID:    "appArmorProfile",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(appArmorProfileV1Dot0),
			},
		},
	}
}

func allowedProfile(profile string) bool {
	return len(profile) == 0 ||
		profile == corev1.AppArmorBetaProfileRuntimeDefault ||
		strings.HasPrefix(profile, corev1.AppArmorBetaProfileNamePrefix)
}

func appArmorProfileV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	forbiddenAppArmorProfile := NewViolations(opts.withFieldErrors)

	for k, v := range podMetadata.Annotations {
		if strings.HasPrefix(k, corev1.AppArmorBetaContainerAnnotationKeyPrefix) && !allowedProfile(v) {
			if opts.withFieldErrors {
				forbiddenAppArmorProfile.Add(fmt.Sprintf("%s=%q", k, v), forbidden(annotationsPath.key(k)).withBadValue(v))
			} else {
				forbiddenAppArmorProfile.Add(fmt.Sprintf("%s=%q", k, v))
			}
		}
	}

	if !forbiddenAppArmorProfile.Empty() {
		forbiddenValues := forbiddenAppArmorProfile.Data()
		sort.Strings(forbiddenValues)
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: pluralize("forbidden AppArmor profile", "forbidden AppArmor profiles", len(forbiddenValues)),
			ForbiddenDetail: strings.Join(forbiddenValues, ", "),
			ErrList:         forbiddenAppArmorProfile.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}
