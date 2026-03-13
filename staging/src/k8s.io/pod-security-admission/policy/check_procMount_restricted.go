/*
Copyright 2025 The Kubernetes Authors.

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
	"k8s.io/pod-security-admission/api"
)

/*

The default /proc masks are set up to reduce attack surface, and should be required by the restricted profile.

**Restricted Fields:**
spec.containers[*].securityContext.procMount
spec.initContainers[*].securityContext.procMount

**Allowed Values:** undefined/null, "Default"

*/

func init() {
	addCheck(CheckProcMountRestricted)
}

// CheckProcMountRestricted returns a restricted level check that forbids unmasked procmount.
func CheckProcMountRestricted() Check {
	return Check{
		ID:    "procMount_restricted",
		Level: api.LevelRestricted,
		Versions: []VersionedCheck{
			{
				// Prior to 1.35, the baseline "procMount" check ran procMount_1_0 to unconditionally block unmasked procMount.
				// In 1.35+, baseline conditionally relaxes for user namespace pods.
				// Starting at that version, keep running the unconditional block in the restricted profile,
				// and override the slightly weaker version of the same check from the baseline profile.
				MinimumVersion:   api.MajorMinorVersion(1, 35),
				CheckPod:         procMount_1_0,
				OverrideCheckIDs: []CheckID{"procMount"},
			},
		},
	}
}
