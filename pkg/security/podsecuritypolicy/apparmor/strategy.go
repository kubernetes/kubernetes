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

package apparmor

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/util/maps"
)

// Strategy defines the interface for all AppArmor constraint strategies.
type Strategy interface {
	// Generate updates the annotations based on constraint rules. The updates are applied to a copy
	// of the annotations, and returned.
	Generate(annotations map[string]string, container *api.Container) (map[string]string, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(pod *api.Pod, container *api.Container) field.ErrorList
}

type strategy struct {
	defaultProfile  string
	allowedProfiles map[string]bool
	// For printing error messages (preserves order).
	allowedProfilesString string
}

var _ Strategy = &strategy{}

// NewStrategy creates a new strategy that enforces AppArmor profile constraints.
func NewStrategy(pspAnnotations map[string]string) Strategy {
	var allowedProfiles map[string]bool
	if allowed, ok := pspAnnotations[apparmor.AllowedProfilesAnnotationKey]; ok {
		profiles := strings.Split(allowed, ",")
		allowedProfiles = make(map[string]bool, len(profiles))
		for _, p := range profiles {
			allowedProfiles[p] = true
		}
	}
	return &strategy{
		defaultProfile:        pspAnnotations[apparmor.DefaultProfileAnnotationKey],
		allowedProfiles:       allowedProfiles,
		allowedProfilesString: pspAnnotations[apparmor.AllowedProfilesAnnotationKey],
	}
}

func (s *strategy) Generate(annotations map[string]string, container *api.Container) (map[string]string, error) {
	copy := maps.CopySS(annotations)

	if annotations[apparmor.ContainerAnnotationKeyPrefix+container.Name] != "" {
		// Profile already set, nothing to do.
		return copy, nil
	}

	if s.defaultProfile == "" {
		// No default set.
		return copy, nil
	}

	if copy == nil {
		copy = map[string]string{}
	}
	// Add the default profile.
	copy[apparmor.ContainerAnnotationKeyPrefix+container.Name] = s.defaultProfile

	return copy, nil
}

func (s *strategy) Validate(pod *api.Pod, container *api.Container) field.ErrorList {
	if s.allowedProfiles == nil {
		// Unrestricted: allow all.
		return nil
	}

	allErrs := field.ErrorList{}
	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(apparmor.ContainerAnnotationKeyPrefix + container.Name)

	profile := apparmor.GetProfileNameFromPodAnnotations(pod.Annotations, container.Name)
	if profile == "" {
		if len(s.allowedProfiles) > 0 {
			allErrs = append(allErrs, field.Forbidden(fieldPath, "AppArmor profile must be set"))
			return allErrs
		}
		return nil
	}

	if !s.allowedProfiles[profile] {
		msg := fmt.Sprintf("%s is not an allowed profile. Allowed values: %q", profile, s.allowedProfilesString)
		allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
	}

	return allErrs
}
