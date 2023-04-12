/*
Copyright 2022 The Kubernetes Authors.

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

// Package resourceclaim provides code that supports the usual pattern
// for accessing the ResourceClaim that is referenced by a PodResourceClaim:
//
// - determine the ResourceClaim name that corresponds to the PodResourceClaim
// - retrieve the ResourceClaim
// - verify that the ResourceClaim is owned by the pod if generated from a template
// - use the ResourceClaim
package resourceclaim

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Name returns the name of the ResourceClaim object that gets referenced by or
// created for the PodResourceClaim. The name is deterministic and therefore
// this function does not need any additional information and it will never
// fail.
//
// Either podClaim.ResourceClaimName or podClaim.Template must be non-nil, but not
// both. This is enforced by API validation.
//
// If podClaim.Template is not nil, the caller must check that the
// ResourceClaim is indeed the one that was created for the Pod by calling
// IsUsable.
func Name(pod *v1.Pod, podClaim *v1.PodResourceClaim) string {
	if podClaim.Source.ResourceClaimName != nil {
		return *podClaim.Source.ResourceClaimName
	}
	return pod.Name + "-" + podClaim.Name
}

// IsForPod checks that the ResourceClaim is the ephemeral volume that
// was created for the Pod. It returns an error that is informative
// enough to be returned by the caller without adding further details
// about the Pod or ResourceClaim.
func IsForPod(pod *v1.Pod, claim *resourcev1alpha2.ResourceClaim) error {
	// Checking the namespaces is just a precaution. The caller should
	// never pass in a ResourceClaim that isn't from the same namespace as the
	// Pod.
	if claim.Namespace != pod.Namespace || !metav1.IsControlledBy(claim, pod) {
		return fmt.Errorf("ResourceClaim %s/%s was not created for pod %s/%s (pod is not owner)", claim.Namespace, claim.Name, pod.Namespace, pod.Name)
	}
	return nil
}

// IsReservedForPod checks whether a claim lists the pod as one of the objects
// that the claim was reserved for.
func IsReservedForPod(pod *v1.Pod, claim *resourcev1alpha2.ResourceClaim) bool {
	for _, reserved := range claim.Status.ReservedFor {
		if reserved.UID == pod.UID {
			return true
		}
	}
	return false
}

// CanBeReserved checks whether the claim could be reserved for another object.
func CanBeReserved(claim *resourcev1alpha2.ResourceClaim) bool {
	return claim.Status.Allocation.Shareable ||
		len(claim.Status.ReservedFor) == 0
}
