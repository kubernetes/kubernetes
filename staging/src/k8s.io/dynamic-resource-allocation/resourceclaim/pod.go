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
	"iter"
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// PodClaims returns an iterator over the names of all ResourceClaims currently
// referenced by or created for the Pod, along with a boolean indicating
// whether IsForPod must be called to verify ownership.
//
// This includes both explicit claims in pod.Spec.ResourceClaims (once created)
// and any DRA-backed extended resource claim in pod.Status.ExtendedResourceClaimStatus.
// Claims that have not been created yet (ErrClaimNotFound), are not needed
// (nil name), or use an unsupported API field (ErrAPIUnsupported) are skipped.
func PodClaims(pod *corev1.Pod) iter.Seq2[string, bool] {
	return func(yield func(string, bool) bool) {
		if pod == nil {
			return
		}
		for i := range pod.Spec.ResourceClaims {
			claimName, mustCheckOwner, err := Name(pod, &pod.Spec.ResourceClaims[i])
			if err != nil || claimName == nil {
				// Name only returns an error when a template claim has not been
				// created yet (ErrClaimNotFound) or when neither ResourceClaimName
				// nor ResourceClaimTemplateName is set (ErrAPIUnsupported). In both
				// cases, or when claimName is nil (claim not needed), there is no
				// existing ResourceClaim to yield.
				continue
			}
			if !yield(*claimName, mustCheckOwner) {
				return
			}
		}
		if pod.Status.ExtendedResourceClaimStatus != nil && pod.Status.ExtendedResourceClaimStatus.ResourceClaimName != "" {
			if !yield(pod.Status.ExtendedResourceClaimStatus.ResourceClaimName, true) {
				return
			}
		}
	}
}

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

func PodExtendedStatusEqual(statusA, statusB *corev1.PodExtendedResourceClaimStatus) bool {
	if statusA == nil && statusB == nil {
		return true
	}
	if (statusA == nil) != (statusB == nil) {
		return false
	}
	if statusA.ResourceClaimName != statusB.ResourceClaimName {
		return false
	}
	if len(statusA.RequestMappings) != len(statusB.RequestMappings) {
		return false
	}
	// In most cases, status entries only get added once and not modified.
	// But this cannot be guaranteed, so for the sake of correctness in all
	// cases this code here has to check.
	return slices.Equal(statusA.RequestMappings, statusB.RequestMappings)
}
