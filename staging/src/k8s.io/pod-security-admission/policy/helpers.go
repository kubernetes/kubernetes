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
	"sync/atomic"

	corev1 "k8s.io/api/core/v1"
)

func joinQuote(items []string) string {
	if len(items) == 0 {
		return ""
	}
	return `"` + strings.Join(items, `", "`) + `"`
}

func pluralize(singular, plural string, count int) string {
	if count == 1 {
		return singular
	}
	return plural
}

var relaxPolicyForUserNamespacePods = &atomic.Bool{}

// RelaxPolicyForUserNamespacePods allows opting into relaxing runAsUser /
// runAsNonRoot restricted policies for user namespace pods, before the
// usernamespace feature has reached GA and propagated to the oldest supported
// nodes.
// This should only be opted into in clusters where the administrator ensures
// all nodes in the cluster enable the user namespace feature.
func RelaxPolicyForUserNamespacePods(relax bool) {
	relaxPolicyForUserNamespacePods.Store(relax)
}

// relaxPolicyForUserNamespacePod returns true if a policy should be relaxed
// because of enabled user namespaces in the provided pod spec.
func relaxPolicyForUserNamespacePod(podSpec *corev1.PodSpec) bool {
	return relaxPolicyForUserNamespacePods.Load() && podSpec != nil && podSpec.HostUsers != nil && !*podSpec.HostUsers
}
