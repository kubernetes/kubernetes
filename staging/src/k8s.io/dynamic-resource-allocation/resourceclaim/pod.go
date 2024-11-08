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

package resourceclaim

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// PodStatusEqual checks that both slices have the same number
// of entries and that the pairs of entries are semantically
// equivalent.
//
// The order of the entries matters: two slices with semantically
// equivalent entries in different order are not equal. This is
// done for the sake of performance because typically the
// order of entries doesn't change.
func PodStatusEqual(statusA, statusB []corev1.PodResourceClaimStatus) bool {
	if len(statusA) != len(statusB) {
		return false
	}
	// In most cases, status entries only get added once and not modified.
	// But this cannot be guaranteed, so for the sake of correctness in all
	// cases this code here has to check.
	for i := range statusA {
		if statusA[i].Name != statusB[i].Name {
			return false
		}
		if !ptr.Equal(statusA[i].ResourceClaimName, statusB[i].ResourceClaimName) {
			return false
		}
	}
	return true
}
