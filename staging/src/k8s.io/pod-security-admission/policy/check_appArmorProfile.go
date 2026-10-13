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
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/pod-security-admission/api"
)

/*
On supported hosts, the 'runtime/default' AppArmor profile is applied by default.
The baseline policy should prevent overriding or disabling the default AppArmor
profile, or restrict overrides to an allowed set of profiles.

**Restricted Fields:**
metadata.annotations['container.apparmor.security.beta.kubernetes.io/*']

**Allowed Values:** 'runtime/default', 'localhost/*', empty, undefined

**Restricted Fields:**
spec.securityContext.appArmorProfile.type
spec.containers[*].securityContext.appArmorProfile.type
spec.initContainers[*].securityContext.appArmorProfile.type
spec.ephemeralContainers[*].securityContext.appArmorProfile.type

**Allowed Values:** 'RuntimeDefault', 'Localhost', undefined
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
				CheckPod:       appArmorProfile_1_0,
			},
		},
	}
}

func allowedAnnotationValue(profile string) bool {
	return len(profile) == 0 ||
		profile == corev1.DeprecatedAppArmorBetaProfileRuntimeDefault ||
		strings.HasPrefix(profile, corev1.DeprecatedAppArmorBetaProfileNamePrefix)
}

func allowedProfileType(profile corev1.AppArmorProfileType) bool {
	switch profile {
	case corev1.AppArmorProfileTypeRuntimeDefault,
		corev1.AppArmorProfileTypeLocalhost:
		return true
	default:
		return false
	}
}

func appArmorProfile_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badSetters []string // things that explicitly set appArmorProfile.type to a bad value
	badValues := sets.NewString()

	if podSpec.SecurityContext != nil && podSpec.SecurityContext.AppArmorProfile != nil {
		if !allowedProfileType(podSpec.SecurityContext.AppArmorProfile.Type) {
			badSetters = append(badSetters, "pod")
			badValues.Insert(string(podSpec.SecurityContext.AppArmorProfile.Type))
		}
	}

	var badContainers []string // containers that set apparmorProfile.type to a bad value
	visitContainers(podSpec, func(c *corev1.Container) {
		if c.SecurityContext != nil && c.SecurityContext.AppArmorProfile != nil {
			if !allowedProfileType(c.SecurityContext.AppArmorProfile.Type) {
				badContainers = append(badContainers, c.Name)
				badValues.Insert(string(c.SecurityContext.AppArmorProfile.Type))
			}
		}
	})

	if len(badContainers) > 0 {
		badSetters = append(
			badSetters,
			fmt.Sprintf(
				"%s %s",
				pluralize("container", "containers", len(badContainers)),
				joinQuote(badContainers),
			),
		)
	}

	var forbiddenAnnotations []string
	for k, v := range podMetadata.Annotations {
		if strings.HasPrefix(k, corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix) && !allowedAnnotationValue(v) {
			forbiddenAnnotations = append(forbiddenAnnotations, fmt.Sprintf("%s=%q", k, v))
		}
	}

	badValueList := badValues.List()
	if len(forbiddenAnnotations) > 0 {
		sort.Strings(forbiddenAnnotations)
		badValueList = append(badValueList, forbiddenAnnotations...)
		badSetters = append(badSetters, pluralize("annotation", "annotations", len(forbiddenAnnotations)))
	}

	// pod or containers explicitly set bad apparmorProfiles
	if len(badSetters) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: pluralize("forbidden AppArmor profile", "forbidden AppArmor profiles", len(badValueList)),
			ForbiddenDetail: fmt.Sprintf(
				"%s must not set AppArmor profile type to %s",
				strings.Join(badSetters, " and "),
				joinQuote(badValueList),
			),
		}
	}

	return CheckResult{Allowed: true}
}
