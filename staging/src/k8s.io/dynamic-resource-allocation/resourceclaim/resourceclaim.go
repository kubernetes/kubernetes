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
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
// created for the PodResourceClaim.
//
// The podGroup parameter should only be non-nil when the
// DRAWorkloadResourceClaims feature gate is enabled. When podGroup is non-nil,
// this function will follow a PodResourceClaim's reference from a
// PodGroupResourceClaim to the PodGroup. When podGroup is nil, Pod claims
// referencing their PodGroup will return [ErrAPIUnsupported].
//
// Three different results are possible:
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
func Name(pod *v1.Pod, podGroup *schedulingapi.PodGroup, podClaim *v1.PodResourceClaim) (name *string, mustCheckOwner bool, err error) {
	switch {
	case podClaim.ResourceClaimName != nil:
		return podClaim.ResourceClaimName, false, nil
	case podClaim.ResourceClaimTemplateName != nil:
		for _, status := range pod.Status.ResourceClaimStatuses {
			if status.Name == podClaim.Name {
				return status.ResourceClaimName, true, nil
			}
		}
		return nil, false, fmt.Errorf("Pod %s/%s: %w", pod.Namespace, pod.Name, ErrClaimNotFound)
	case podGroup != nil && podClaim.PodGroupResourceClaim != nil:
		if err := podMatchesPodGroup(pod, podGroup); err != nil {
			return nil, false, err
		}
		podGroupClaimIndex := slices.IndexFunc(podGroup.Spec.ResourceClaims, func(podGroupClaim schedulingapi.PodGroupResourceClaim) bool {
			return podGroupClaim.Name == *podClaim.PodGroupResourceClaim
		})
		if podGroupClaimIndex < 0 {
			return nil, false, fmt.Errorf("PodGroup %s/%s does not have claim %s requested by Pod %s", podGroup.Namespace, podGroup.Name, *podClaim.PodGroupResourceClaim, pod.Name)
		}
		podGroupClaim := podGroup.Spec.ResourceClaims[podGroupClaimIndex]
		switch {
		case podGroupClaim.ResourceClaimName != nil:
			return podGroupClaim.ResourceClaimName, false, nil
		case podGroupClaim.ResourceClaimTemplateName != nil:
			for _, status := range pod.Status.ResourceClaimStatuses {
				if status.Name == podClaim.Name {
					return status.ResourceClaimName, true, nil
				}
			}
			return nil, false, fmt.Errorf("PodGroup %s/%s: %w", podGroup.Namespace, podGroup.Name, ErrClaimNotFound)
		default:
			return nil, false, fmt.Errorf("PodGroup %s/%s, spec.resourceClaim %s: %w", podGroup.Namespace, podGroup.Name, podGroupClaim.Name, ErrAPIUnsupported)
		}
	default:
		return nil, false, fmt.Errorf("Pod %s/%s, spec.resourceClaim %s: %w", pod.Namespace, pod.Name, podClaim.Name, ErrAPIUnsupported)
	}
}

// IsForPod checks that the ResourceClaim is the one that
// was created for the Pod. It returns an error that is informative
// enough to be returned by the caller without adding further details
// about the Pod or ResourceClaim.
//
// The podGroup parameter should only be non-nil when the
// DRAWorkloadResourceClaims feature gate is enabled. When the PodGroup is
// non-nil, IsForPod alternatively checks that the ResourceClaim is the one that
// was created for the PodGroup.
func IsForPod(pod *v1.Pod, podGroup *schedulingapi.PodGroup, claim *resourceapi.ResourceClaim) error {
	// Checking the namespaces is just a precaution. The caller should
	// never pass in a ResourceClaim that isn't from the same namespace as the
	// Pod.
	if claim.Namespace != pod.Namespace || !metav1.IsControlledBy(claim, pod) {
		if podGroup != nil {
			if err := podMatchesPodGroup(pod, podGroup); err != nil {
				return err
			}
			if claim.Namespace != podGroup.Namespace || !metav1.IsControlledBy(claim, podGroup) {
				return fmt.Errorf("ResourceClaim %s/%s was not created for Pod %s/%s or PodGroup %s/%s (neither Pod nor PodGroup is the owner)", claim.Namespace, claim.Name, pod.Namespace, pod.Name, podGroup.Namespace, podGroup.Name)
			}
			return nil
		}
		return fmt.Errorf("ResourceClaim %s/%s was not created for Pod %s/%s (Pod is not owner)", claim.Namespace, claim.Name, pod.Namespace, pod.Name)
	}
	return nil
}

// IsReservedForPod checks whether a claim lists the Pod or the PodGroup as one
// of the objects that the claim was reserved for.
//
// The podGroup parameter should only be non-nil when the
// DRAWorkloadResourceClaims feature gate is enabled. When the podGroup is nil,
// IsReservedForPod returns false when the claim is owned by the Pod's PodGroup
// and not the Pod itself.
func IsReservedForPod(pod *v1.Pod, podGroup *schedulingapi.PodGroup, claim *resourceapi.ResourceClaim) bool {
	checkPodGroup := podGroup != nil && podMatchesPodGroup(pod, podGroup) == nil
	for _, reserved := range claim.Status.ReservedFor {
		if reserved.UID == pod.UID ||
			(checkPodGroup && reserved.UID == podGroup.UID) {
			return true
		}
	}
	return false
}

// CanBeReserved checks whether the claim could be reserved for another object.
func CanBeReserved(claim *resourceapi.ResourceClaim) bool {
	// Currently no restrictions on sharing...
	return true
}

// BindTo constructs a consumer reference for the claim that refers to the
// object to which the claim should be bound. When the claim refers to a
// ResourceClaimName or ResourceClaimTemplateName, the claim is bound to the
// Pod. When the claim refers to a PodGroupResourceClaim and the podGroup
// parameter is non-nil, the claim is bound to the PodGroup. The podGroup
// parameter should only be non-nil when the
// DRAWorkloadResourceClaims feature gate is enabled.
func BindTo(pod *v1.Pod, podGroup *schedulingapi.PodGroup, podClaim *v1.PodResourceClaim) (resourceapi.ResourceClaimConsumerReference, error) {
	switch {
	case podClaim.ResourceClaimName != nil, podClaim.ResourceClaimTemplateName != nil:
		return resourceapi.ResourceClaimConsumerReference{
			APIGroup: v1.GroupName,
			Resource: "pods",
			Name:     pod.Name,
			UID:      pod.UID,
		}, nil
	case podGroup != nil && podClaim.PodGroupResourceClaim != nil:
		if err := podMatchesPodGroup(pod, podGroup); err != nil {
			return resourceapi.ResourceClaimConsumerReference{}, err
		}
		return resourceapi.ResourceClaimConsumerReference{
			APIGroup: schedulingapi.GroupName,
			Resource: "podgroups",
			Name:     podGroup.Name,
			UID:      podGroup.UID,
		}, nil
	default:
		return resourceapi.ResourceClaimConsumerReference{}, fmt.Errorf(`pod "%s/%s", spec.resourceClaim %q: %w`, pod.Namespace, pod.Name, podClaim.Name, ErrAPIUnsupported)
	}
}

// BaseRequestRef returns the request name if the reference is to a top-level
// request and the name of the parent request if the reference is to a subrequest.
func BaseRequestRef(requestRef string) string {
	segments := strings.Split(requestRef, "/")
	return segments[0]
}

// CreateSubRequestRef combines the names from a request and a subrequest into
// a reference to the subrequest.
func CreateSubRequestRef(requestName, subRequestName string) string {
	return fmt.Sprintf("%s/%s", requestName, subRequestName)
}

// IsSubRequestRef checks if the provided reference is to a subrequest and returns
// true if it is. Otherwise it returns false.
func IsSubRequestRef(requestRef string) bool {
	segments := strings.Split(requestRef, "/")
	return len(segments) == 2
}

// ConfigForResult returns the configs that are applicable to device
// allocated for the provided result.
func ConfigForResult(deviceConfigurations []resourceapi.DeviceAllocationConfiguration, result resourceapi.DeviceRequestAllocationResult) []resourceapi.DeviceAllocationConfiguration {
	var configs []resourceapi.DeviceAllocationConfiguration
	for _, deviceConfiguration := range deviceConfigurations {
		if deviceConfiguration.Opaque != nil &&
			isMatch(deviceConfiguration.Requests, result.Request) {
			configs = append(configs, deviceConfiguration)
		}
	}
	return configs
}

func isMatch(requests []string, requestRef string) bool {
	if len(requests) == 0 {
		return true
	}

	if slices.Contains(requests, requestRef) {
		return true
	}

	baseRequestRef := BaseRequestRef(requestRef)
	return slices.Contains(requests, baseRequestRef)
}

func podMatchesPodGroup(pod *v1.Pod, podGroup *schedulingapi.PodGroup) error {
	if pod.Spec.SchedulingGroup == nil || pod.Spec.SchedulingGroup.PodGroupName == nil {
		return fmt.Errorf("Pod %s/%s does not belong to a PodGroup", pod.Namespace, pod.Name)
	}
	if schedGroupPodGroup := *pod.Spec.SchedulingGroup.PodGroupName; schedGroupPodGroup != podGroup.Name {
		return fmt.Errorf("Pod %s/%s belongs to PodGroup %s/%s, not PodGroup %s/%s", pod.Namespace, pod.Name, pod.Namespace, schedGroupPodGroup, podGroup.Namespace, podGroup.Name)
	}
	return nil
}
