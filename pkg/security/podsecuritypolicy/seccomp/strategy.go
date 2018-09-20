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
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// SeccompAllowAny is the wildcard used to allow any profile.
const SeccompAllowAny = "*"

// SeccompStrategy defines the interface for all seccomp constraint strategies.
type SeccompStrategy interface {
	// Generate returns a profile based on constraint rules.
	Generate(pod *api.Pod, container *api.Container) (*string, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(fieldPath *field.Path, profile *string) field.ErrorList
}

type seccompStrategy struct {
	defaultProfile  *string
	allowedProfiles map[string]bool
	// For printing error messages (preserves order).
	allowedProfilesString string
	// does the strategy allow any profile (wildcard)
	allowAnyProfile bool
}

var _ SeccompStrategy = &seccompStrategy{}

// NewSeccompStrategy creates a new strategy that enforces seccomp profile constraints.
func NewSeccompStrategy(defaultSeccompProfile *string, allowedSeccompProfiles []string) SeccompStrategy {
	var allowedProfiles = make(map[string]bool, len(allowedSeccompProfiles))
	allowAnyProfile := false
	for _, profile := range allowedSeccompProfiles {
		if profile == SeccompAllowAny {
			allowAnyProfile = true
			continue
		}
		allowedProfiles[profile] = true
	}
	return &seccompStrategy{
		defaultProfile:        defaultSeccompProfile,
		allowedProfiles:       allowedProfiles,
		allowedProfilesString: strings.Join(allowedSeccompProfiles, ", "),
		allowAnyProfile:       allowAnyProfile,
	}
}

// Generate returns a profile from either container or pod if already set, otherwise
// it returns a profile based on the defaultProfile from the given strategy
func (s *seccompStrategy) Generate(pod *api.Pod, container *api.Container) (*string, error) {
	if container != nil && container.SecurityContext != nil {
		if p := container.SecurityContext.SeccompProfile; p != nil {
			return p, nil
		}
	}

	if pod != nil && pod.Spec.SecurityContext != nil {
		if p := pod.Spec.SecurityContext.SeccompProfile; p != nil {
			return p, nil
		}
	}
	return s.defaultProfile, nil
}

// Validate ensures that the specified seccomp profile name falls
// within the range of the strategy.
func (s *seccompStrategy) Validate(fieldPath *field.Path, profile *string) field.ErrorList {
	allErrs := field.ErrorList{}

	if !s.allowAnyProfile && len(s.allowedProfiles) == 0 && profile != nil {
		allErrs = append(allErrs, field.Forbidden(fieldPath, "seccomp must not be set"))
		return allErrs
	}

	if !s.profileAllowed(profile) {
		profileString := "<nil>"
		if profile != nil {
			profileString = *profile
		}
		msg := fmt.Sprintf("%s is not an allowed seccomp profile. Valid values are %v", profileString, s.allowedProfilesString)
		allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
	}

	return allErrs
}

// profileAllowed checks if profile is in allowedProfiles or if allowedProfiles
// contains the wildcard.
func (s *seccompStrategy) profileAllowed(podProfile *string) bool {
	if s.allowAnyProfile {
		return true
	}

	if podProfile == nil {
		// for backwards compatibility and PSPs without a defined list of allowed profiles.
		// If a PSP does not have allowedProfiles set then we should allow an empty profile.
		// This will mean that the runtime default is used.
		if len(s.allowedProfiles) == 0 {
			return true
		}
		return false
	}

	return s.allowedProfiles[*podProfile]
}
