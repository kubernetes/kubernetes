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
	"k8s.io/pod-security-admission/api"
)

/*
HostPath volumes must be forbidden.

**Restricted Fields:**

spec.volumes[*].hostPath

**Allowed Values:** undefined/nil
*/

func init() {
	addCheck(CheckHostPath)
}

// CheckHostPath returns a baseline level check
// that requires hostPath=undefined/nil in 1.0+
func CheckHostPath() Check {
	return Check{
		ID:    "hostPath",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       hostPath_1_0,
			},
		},
	}
}

func hostPath_1_0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	hostVolumes := sets.NewString()

	for _, volume := range podSpec.Volumes {
		if volume.HostPath != nil {
			hostVolumes.Insert(volume.Name)
		}
	}

	if len(hostVolumes) > 0 {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostPath volumes",
			ForbiddenDetail: fmt.Sprintf("volumes %q", hostVolumes.List()),
		}
	}

	return CheckResult{Allowed: true}
}
