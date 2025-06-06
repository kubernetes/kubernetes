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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/pod-security-admission/api"
)

/*

The default /proc masks are set up to reduce attack surface, and should be required, unless
the pod is in a user namespace ("hostUsers: false")

**Restricted Fields:**
spec.containers[*].securityContext.procMount
spec.initContainers[*].securityContext.procMount

**Allowed Values:** undefined/null, "Default" if "hostUsers": false, otherwise all values are allowed.

*/

func init() {
	addCheck(CheckProcMountBaseline)
}

// CheckProcMountBaseline returns a baseline level check that restricts
// setting the value of securityContext.procMount to DefaultProcMount
// in 1.0+
func CheckProcMountBaseline() Check {
	return Check{
		ID:    "procMount_baseline",
		Level: api.LevelBaseline,
		Versions: []VersionedCheck{
			// Formerly, the 1.0 version only took user namespaces into account
			// if the now dropped UserNamespacesPodSecurityStandards feature gate was on.
			// That feature gate has since been dropped.
			// Since it was only ever off by default (alpha), the majority of clusters
			// didn't take user namespaces into account.
			// Thus, if the MinimumVersion is 1.0, use the restricted check, which
			// doesn't take user namespaces into account.
			{
				MinimumVersion: api.MajorMinorVersion(1, 0),
				CheckPod:       procMount_1_35_restricted,
			},
			{
				MinimumVersion: api.MajorMinorVersion(1, 35),
				CheckPod:       procMount_1_35_baseline,
			},
		},
	}
}

func procMount_1_35_baseline(podMetadata *metav1.ObjectMeta, podSpec *corev1.PodSpec) CheckResult {
	// Now that we've removed the UserNamespacesPodSecurityStandards feature gate (and GA this relaxation),
	// we have created a new policy version (1.1).
	// Note: pod validation will check for well formed procMount type, so avoid double validation and allow everything
	// here.
	if relaxPolicyForUserNamespacePod(podSpec) {
		return CheckResult{Allowed: true}
	}

	// If the pod is not in a user namespace, treat it as restricted is.
	return procMount_1_35_restricted(podMetadata, podSpec)
}
