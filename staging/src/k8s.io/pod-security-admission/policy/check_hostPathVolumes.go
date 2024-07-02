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
	"k8s.io/pod-security-admission/api"
)

/*
HostPath volumes must be forbidden.

**Restricted Fields:**

spec.volumes[*].hostPath

**Allowed Values:** undefined/null
*/

func init() {
	addCheck(CheckHostPathVolumes)
}

const checkHostPathVolumesID CheckID = "hostPathVolumes"

// CheckHostPathVolumes returns a baseline level check
// that requires hostPath=undefined/null in 1.0+
func CheckHostPathVolumes() Check {
	return Check{
		ID:    checkHostPathVolumesID,
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       withOptions(hostPathVolumesV1Dot0),
			},
		},
	}
}

func hostPathVolumesV1Dot0(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec, opts options) CheckResult {
	hostVolumes := NewViolations(opts.withFieldErrors)

	for i, volume := range podSpec.Volumes {
		if volume.HostPath != nil {
			if opts.withFieldErrors {
				hostVolumes.Add(volume.Name, withBadValue(forbidden(specPath.Child("volumes").Index(i).Child("hostPath")), volume.HostPath.Path))
			} else {
				hostVolumes.Add(volume.Name)
			}
		}
	}

	if !hostVolumes.Empty() {
		return CheckResult{
			Allowed:         false,
			ForbiddenReason: "hostPath volumes",
			ForbiddenDetail: fmt.Sprintf("%s %s", pluralize("volume", "volumes", hostVolumes.Len()), joinQuote(hostVolumes.Data())),
			ErrList:         hostVolumes.Errs(),
		}
	}

	return CheckResult{Allowed: true}
}
