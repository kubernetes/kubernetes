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
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha2 "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/utils/ptr"
)

var (
	// ErrAPIUnsupported is wrapped by the actual errors returned by Name and
	// indicates that none of the required fields are set.
	ErrAPIUnsupported = errors.New("none of the supported fields are set")

	// ErrClaimNotFound is wrapped by the actual errors returned by Name and
	// indicates that the claim has not been created yet.
	ErrClaimNotFound = errors.New("ResourceClaim not created yet")
)

// Name returns the name of the ResourceClaim object that gets referenced by or
// created for the PodResourceClaim. Three different results are possible:
//
//   - An error is returned when some field is not set as expected (either the
//     input is invalid or the API got extended and the library and the client
//     using it need to be updated) or the claim hasn't been created yet.
//
//     The error includes pod and pod claim name and the unexpected field and
//     is derived from one of the pre-defined errors in this package.
//
//   - A nil string pointer and no error when the ResourceClaim intentionally
//     didn't get created and the PodResourceClaim can be ignored.
//
//   - A pointer to the name and no error when the ResourceClaim got created.
//     In this case the boolean determines whether IsForPod must be called
//     after retrieving the ResourceClaim and before using it.
//
// Determining the name depends on Kubernetes >= 1.28.
func Name(pod *v1.Pod, podClaim *v1.PodResourceClaim) (name *string, mustCheckOwner bool, err error) {
	switch {
	case podClaim.Source.ResourceClaimName != nil:
		return podClaim.Source.ResourceClaimName, false, nil
	case podClaim.Source.ResourceClaimTemplateName != nil:
		for _, status := range pod.Status.ResourceClaimStatuses {
			if status.Name == podClaim.Name {
				return status.ResourceClaimName, true, nil
			}
		}
		return nil, false, fmt.Errorf(`pod "%s/%s": %w`, pod.Namespace, pod.Name, ErrClaimNotFound)
	default:
		return nil, false, fmt.Errorf(`pod "%s/%s", spec.resourceClaim %q: %w`, pod.Namespace, pod.Name, podClaim.Name, ErrAPIUnsupported)
	}
}

// NewNameLookup returns an object which handles determining the name of
// a ResourceClaim. In contrast to the stand-alone Name it is compatible
// also with Kubernetes < 1.28.
//
// Providing a client is optional. If none is available, then code can pass nil
// and users can set the DRA_WITH_DETERMINISTIC_RESOURCE_CLAIM_NAMES env
// variable to an arbitrary non-empty value to use the naming from Kubernetes <
// 1.28.
func NewNameLookup(client kubernetes.Interface) *Lookup {
	return &Lookup{client: client}
}

// Lookup stores the state which is necessary to look up ResourceClaim names.
type Lookup struct {
	client       kubernetes.Interface
	usePodStatus atomic.Pointer[bool]
}

// Name is a variant of the stand-alone Name with support also for Kubernetes < 1.28.
func (l *Lookup) Name(pod *v1.Pod, podClaim *v1.PodResourceClaim) (name *string, mustCheckOwner bool, err error) {
	usePodStatus := l.usePodStatus.Load()
	if usePodStatus == nil {
		if value, _ := os.LookupEnv("DRA_WITH_DETERMINISTIC_RESOURCE_CLAIM_NAMES"); value != "" {
			usePodStatus = ptr.To(false)
		} else if l.client != nil {
			// Check once. This does not detect upgrades or
			// downgrades, but that is good enough for the simple
			// test scenarios that the Kubernetes < 1.28 support is
			// meant for.
			info, err := l.client.Discovery().ServerVersion()
			if err != nil {
				return nil, false, fmt.Errorf("look up server version: %v", err)
			}
			if info.Major == "" {
				// Fake client...
				usePodStatus = ptr.To(true)
			} else {
				switch strings.Compare(info.Major, "1") {
				case -1:
					// Huh?
					usePodStatus = ptr.To(false)
				case 0:
					// info.Minor may have a suffix which makes it larger than 28.
					// We don't care about pre-releases here.
					usePodStatus = ptr.To(strings.Compare("28", info.Minor) <= 0)
				case 1:
					// Kubernetes 2? Yeah!
					usePodStatus = ptr.To(true)
				}
			}
		} else {
			// No information. Let's assume recent Kubernetes.
			usePodStatus = ptr.To(true)
		}
		l.usePodStatus.Store(usePodStatus)
	}

	if *usePodStatus {
		return Name(pod, podClaim)
	}

	switch {
	case podClaim.Source.ResourceClaimName != nil:
		return podClaim.Source.ResourceClaimName, false, nil
	case podClaim.Source.ResourceClaimTemplateName != nil:
		name := pod.Name + "-" + podClaim.Name
		return &name, true, nil
	default:
		return nil, false, fmt.Errorf(`pod "%s/%s", spec.resourceClaim %q: %w`, pod.Namespace, pod.Name, podClaim.Name, ErrAPIUnsupported)
	}
}

// IsForPod checks that the ResourceClaim is the one that
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
