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

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// SeccompAllowAny is the wildcard used to allow any profile.
	SeccompAllowAny = "*"
	// The annotation key specifying the default seccomp profile.
	DefaultProfileAnnotationKey = "seccomp.security.alpha.kubernetes.io/defaultProfileName"
	// The annotation key specifying the allowed seccomp profiles.
	AllowedProfilesAnnotationKey = "seccomp.security.alpha.kubernetes.io/allowedProfileNames"
)

// SeccompStrategy defines the interface for all seccomp constraint strategies.
type SeccompStrategy interface {
	// Generate returns a profile based on constraint rules.
	Generate(pod *api.Pod, container *api.Container) (*string, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidatePod(pod *api.Pod, podSCPath *field.Path) field.ErrorList
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidateContainer(pod *api.Pod, container *api.Container, contSCPath *field.Path) field.ErrorList
	validate(fieldPath *field.Path, profile *string) field.ErrorList
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

// NewSeccompStrategy takes psp, extracts default and allowed profiles from either its fields
// or annotations (fields have higher priority), and returns a new SeccompStrategy
func NewSeccompStrategy(psp *policy.PodSecurityPolicy) SeccompStrategy {
	defaultProfile := psp.Spec.DefaultSeccompProfile
	allowedProfiles := psp.Spec.AllowedSeccompProfiles

	// prioritize field values, but set to annotation ones if fields are unset
	if defaultProfile == nil {
		if p, found := psp.Annotations[DefaultProfileAnnotationKey]; found {
			defaultProfile = &p
		}
	}

	if allowedProfiles == nil {
		if profilesAnnotation, found := psp.Annotations[AllowedProfilesAnnotationKey]; found {
			allowedProfiles = strings.Split(profilesAnnotation, ",")
		}
	}

	return newSeccompStrategy(defaultProfile, allowedProfiles)
}

// newSeccompStrategy creates a new strategy that enforces seccomp profile constraints
func newSeccompStrategy(defaultSeccompProfile *string, allowedSeccompProfiles []string) SeccompStrategy {
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
		allowedProfilesString: strings.Join(allowedSeccompProfiles, ","),
		allowAnyProfile:       allowAnyProfile,
	}
}

// Generate returns a profile from either container or pod if already set, otherwise
// it returns a profile based on the defaultProfile from the given strategy
func (s *seccompStrategy) Generate(pod *api.Pod, container *api.Container) (*string, error) {
	if container != nil {
		if p, _ := getContainerSeccompProfile(pod, container.Name, container.SecurityContext, nil); p != nil {
			return p, nil
		}
	}

	if pod != nil && pod.Spec.SecurityContext != nil {
		if p, _ := getPodSeccompProfile(pod, nil); p != nil {
			return p, nil
		}
	}
	return s.defaultProfile, nil
}

// ValidatePod ensures that the specified values on the pod fall within the range
// of the strategy.
func (s *seccompStrategy) ValidatePod(pod *api.Pod, podSCPath *field.Path) field.ErrorList {
	podProfile, podSCPath := getPodSeccompProfile(pod, podSCPath)
	return s.validate(podSCPath, podProfile)
}

// ValidateContainer ensures that the specified values on the container fall within
// the range of the strategy.
func (s *seccompStrategy) ValidateContainer(pod *api.Pod, container *api.Container, contSCPath *field.Path) field.ErrorList {
	containerProfile, foundField := getContainerSeccompProfile(pod, container.Name, container.SecurityContext, contSCPath)
	return s.validate(foundField, containerProfile)
}

// validate ensures that the specified seccomp profile name falls
// within the range of the strategy.
func (s *seccompStrategy) validate(fieldPath *field.Path, profile *string) field.ErrorList {
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
func (s *seccompStrategy) profileAllowed(profile *string) bool {
	if s.allowAnyProfile {
		return true
	}

	if profile == nil {
		// for backwards compatibility and PSPs without a defined list of allowed profiles.
		// If a PSP does not have allowedProfiles set then we should allow an empty profile.
		// This will mean that the runtime default is used.
		if len(s.allowedProfiles) == 0 {
			return true
		}
		return false
	}

	return s.allowedProfiles[*profile]
}

func getPodSeccompProfile(pod *api.Pod, scPath *field.Path) (*string, *field.Path) {
	if sc := pod.Spec.SecurityContext; sc != nil && sc.SeccompProfile != nil {
		return sc.SeccompProfile, scPath
	}

	if p, found := pod.Annotations[api.SeccompPodAnnotationKey]; found {
		return &p, field.NewPath("pod", "metadata", "annotations").Key(api.SeccompPodAnnotationKey)
	}

	return nil, nil
}

func getContainerSeccompProfile(pod *api.Pod, containerName string, containerSC *api.SecurityContext, contSCPath *field.Path) (*string, *field.Path) {
	if containerSC != nil && containerSC.SeccompProfile != nil {
		return containerSC.SeccompProfile, contSCPath
	}

	if p, found := pod.Annotations[api.SeccompContainerAnnotationKeyPrefix+containerName]; found {
		return &p, field.NewPath("pod", "metadata", "annotations").Key(api.SeccompContainerAnnotationKeyPrefix + containerName)
	}

	if podProfile, foundField := getPodSeccompProfile(pod, field.NewPath("pod", "spec", "securityContext", "seccompProfile")); podProfile != nil {
		return podProfile, foundField
	}

	return nil, nil
}
