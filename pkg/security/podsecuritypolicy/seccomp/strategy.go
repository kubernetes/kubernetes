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

const (
	// AllowAny is the wildcard used to allow any profile.
	AllowAny = "*"
	// DefaultProfileAnnotationKey specifies the default seccomp profile.
	DefaultProfileAnnotationKey = "seccomp.security.alpha.kubernetes.io/defaultProfileName"
	// AllowedProfilesAnnotationKey specifies the allowed seccomp profiles.
	AllowedProfilesAnnotationKey = "seccomp.security.alpha.kubernetes.io/allowedProfileNames"
)

// Strategy defines the interface for all seccomp constraint strategies.
type Strategy interface {
	// Generate returns a profile based on constraint rules.
	Generate(annotations map[string]string, pod *api.Pod) (string, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidatePod(pod *api.Pod) field.ErrorList
	// Validate ensures that the specified values fall within the range of the strategy.
	ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList
}

type strategy struct {
	defaultProfile  string
	allowedProfiles map[string]bool
	// For printing error messages (preserves order).
	allowedProfilesString string
	// does the strategy allow any profile (wildcard)
	allowAnyProfile bool
}

var _ Strategy = &strategy{}

// NewStrategy creates a new strategy that enforces seccomp profile constraints.
func NewStrategy(pspAnnotations map[string]string) Strategy {
	var allowedProfiles map[string]bool
	allowAnyProfile := false
	if allowed, ok := pspAnnotations[AllowedProfilesAnnotationKey]; ok {
		profiles := strings.Split(allowed, ",")
		allowedProfiles = make(map[string]bool, len(profiles))
		for _, p := range profiles {
			if p == AllowAny {
				allowAnyProfile = true
				continue
			}
			allowedProfiles[p] = true
		}
	}
	return &strategy{
		defaultProfile:        pspAnnotations[DefaultProfileAnnotationKey],
		allowedProfiles:       allowedProfiles,
		allowedProfilesString: pspAnnotations[AllowedProfilesAnnotationKey],
		allowAnyProfile:       allowAnyProfile,
	}
}

// Generate returns a profile based on constraint rules.
func (s *strategy) Generate(annotations map[string]string, pod *api.Pod) (string, error) {
	if annotations[api.SeccompPodAnnotationKey] != "" {
		// Profile already set, nothing to do.
		return annotations[api.SeccompPodAnnotationKey], nil
	}
	return s.defaultProfile, nil
}

// ValidatePod ensures that the specified values on the pod fall within the range
// of the strategy.
func (s *strategy) ValidatePod(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}
	podSpecFieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.SeccompPodAnnotationKey)
	podProfile := pod.Annotations[api.SeccompPodAnnotationKey]

	if !s.allowAnyProfile && len(s.allowedProfiles) == 0 && podProfile != "" {
		allErrs = append(allErrs, field.Forbidden(podSpecFieldPath, "seccomp may not be set"))
		return allErrs
	}

	if !s.profileAllowed(podProfile) {
		msg := fmt.Sprintf("%s is not an allowed seccomp profile. Valid values are %v", podProfile, s.allowedProfilesString)
		allErrs = append(allErrs, field.Forbidden(podSpecFieldPath, msg))
	}

	return allErrs
}

// ValidateContainer ensures that the specified values on the container fall within
// the range of the strategy.
func (s *strategy) ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}
	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.SeccompContainerAnnotationKeyPrefix + container.Name)
	containerProfile := profileForContainer(pod, container)

	if !s.allowAnyProfile && len(s.allowedProfiles) == 0 && containerProfile != "" {
		allErrs = append(allErrs, field.Forbidden(fieldPath, "seccomp may not be set"))
		return allErrs
	}

	if !s.profileAllowed(containerProfile) {
		msg := fmt.Sprintf("%s is not an allowed seccomp profile. Valid values are %v", containerProfile, s.allowedProfilesString)
		allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
	}

	return allErrs
}

// profileAllowed checks if profile is in allowedProfiles or if allowedProfiles
// contains the wildcard.
func (s *strategy) profileAllowed(profile string) bool {
	// for backwards compatibility and PSPs without a defined list of allowed profiles.
	// If a PSP does not have allowedProfiles set then we should allow an empty profile.
	// This will mean that the runtime default is used.
	if len(s.allowedProfiles) == 0 && profile == "" {
		return true
	}

	return s.allowAnyProfile || s.allowedProfiles[profile]
}

// profileForContainer returns the container profile if set, otherwise the pod profile.
func profileForContainer(pod *api.Pod, container *api.Container) string {
	containerProfile, ok := pod.Annotations[api.SeccompContainerAnnotationKeyPrefix+container.Name]
	if ok {
		return containerProfile
	}
	return pod.Annotations[api.SeccompPodAnnotationKey]
}
