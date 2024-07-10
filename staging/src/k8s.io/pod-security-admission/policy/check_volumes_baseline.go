/*
Copyright 2024 The Kubernetes Authors.

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

**Restricted Fields:**
spec.volumes[*].gitRepo
*/

var checkBaselineVolumesID = CheckID("baselineVolumes")

func init() {
	addCheck(CheckBaselineVolumes)
}

// CheckBaselineVolumes returns a baseline level check
// that limits usage of specific volume types in 1.0+
func CheckBaselineVolumes() Check {
	return Check{
		ID:    checkBaselineVolumesID,
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 31),
				CheckPod:       baselineVolumes1_31,
			},
		},
	}
}

func baselineVolumes1_31(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	var badVolumes []string
	badVolumeTypes := sets.NewString()

	for _, volume := range podSpec.Volumes {
		if volume.GitRepo != nil {
			badVolumes = append(badVolumes, volume.Name)
			badVolumeTypes.Insert("gitRepo")
		}
	}

	if len(badVolumes) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "restricted volume types",
			ForbiddenDetail: fmt.Sprintf(
				"%s %s %s %s %s",
				pluralize("volume", "volumes", len(badVolumes)),
				joinQuote(badVolumes),
				pluralize("uses", "use", len(badVolumes)),
				pluralize("restricted volume type", "restricted volume types", len(badVolumeTypes)),
				joinQuote(badVolumeTypes.List()),
			),
		}
	}

	return CheckResult{Allowed: true}
}
