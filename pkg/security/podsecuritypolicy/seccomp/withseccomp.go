/*
Copyright 2016 The Kubernetes Authors.

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

package seccomp

import (
	"fmt"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

const (
	allowAnyProfile = "*"
)

// withSeccompProfile implements the SeccompStrategy.
type withSeccompProfile struct {
	allowedProfiles []string
}

var _ SeccompStrategy = &withSeccompProfile{}

// NewWithSeccompProfile creates a new must run as strategy or returns an error if it cannot
// be created.
func NewWithSeccompProfile(allowedProfiles []string) (SeccompStrategy, error) {
	return &withSeccompProfile{allowedProfiles}, nil
}

// Generate creates the profile based on policy rules.
func (s *withSeccompProfile) Generate(pod *api.Pod) (string, error) {
	// return the first non-wildcard profile
	for _, p := range s.allowedProfiles {
		if p != allowAnyProfile {
			return p, nil
		}
	}
	// if we reached this point then either there are no allowed profiles (empty slice)
	// or the only thing in the slice is the wildcard.  In either case just return empty
	// which means use the runtime default.
	return "", nil
}

// ValidatePod ensures that the specified values on the pod fall within the range
// of the strategy.
func (s *withSeccompProfile) ValidatePod(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}
	podSpecFieldPath := field.NewPath("pod", "spec", "securityContext", "seccompProfile")

	if len(s.allowedProfiles) == 0 &&
		pod.Spec.SecurityContext != nil &&
		pod.Spec.SecurityContext.SeccompProfile != "" {

		allErrs = append(allErrs, field.Forbidden(podSpecFieldPath, "seccomp may not be set"))
		return allErrs
	}

	if pod.Spec.SecurityContext != nil {
		if !isProfileAllowed(pod.Spec.SecurityContext.SeccompProfile, s.allowedProfiles) {
			msg := fmt.Sprintf("%s is not a valid seccomp profile. Valid values are %v", pod.Spec.SecurityContext.SeccompProfile, s.allowedProfiles)
			allErrs = append(allErrs, field.Forbidden(podSpecFieldPath, msg))
		}
	}

	return allErrs
}

// ValidateContainer ensures that the specified values on the container fall within
// the range of the strategy.
func (s *withSeccompProfile) ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}
	fieldPath := field.NewPath("pod", "spec", "containers").Key(container.Name).Child("securityContext", "seccompProfile")

	if len(s.allowedProfiles) == 0 &&
		container.SecurityContext != nil &&
		container.SecurityContext.SeccompProfile != "" {
		allErrs = append(allErrs, field.Forbidden(fieldPath, "seccomp may not be set"))
		return allErrs
	}

	if container.SecurityContext != nil {
		if !isProfileAllowed(container.SecurityContext.SeccompProfile, s.allowedProfiles) {
			msg := fmt.Sprintf("%s is not a valid seccomp profile. Valid values are %v", container.SecurityContext.SeccompProfile, s.allowedProfiles)
			allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
		}
	}

	return allErrs
}

// isProfileAllowed checks if profile is in allowedProfiles or if allowedProfiles
// contains the wildcard.
func isProfileAllowed(profile string, allowedProfiles []string) bool {
	// for backwards compatibility and PSPs without a defined list of allowed profiles.
	// If a PSP does not have allowedProfiles set then we should allow an empty profile.
	// This will mean that the runtime default is used.
	if len(allowedProfiles) == 0 && profile == "" {
		return true
	}

	for _, p := range allowedProfiles {
		if profile == p || p == allowAnyProfile {
			return true
		}
	}
	return false
}
