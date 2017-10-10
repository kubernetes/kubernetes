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
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/interfaces"
)

type strategy struct {
	defaultProfile  string
	allowedProfiles map[string]bool
	// For printing error messages (preserves order).
	allowedProfilesString string
}

var _ interfaces.ContainerValidatorDefaulter = &strategy{}

// NewStrategy creates a new strategy that enforces AppArmor profile constraints.
func NewStrategy(pspAnnotations map[string]string) interfaces.ContainerValidatorDefaulter {
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

func (s *strategy) ValidateContainer(pod *api.Pod, container *api.Container, _ *api.SecurityContext) (*interfaces.ValidationResult, error) {
	result := &interfaces.ValidationResult{}

	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(apparmor.ContainerAnnotationKeyPrefix + container.Name)
	profile := apparmor.GetProfileNameFromPodAnnotations(pod.Annotations, container.Name)
	switch {
	case profile == "" && len(s.defaultProfile) > 0:
		// Needs defaulting
		result.Add(interfaces.RequiresDefaulting)

	case s.allowedProfiles[profile]:
		// Allowed
		result.Add(interfaces.Allowed)

	case len(s.allowedProfiles) == 0:
		// Unconstrained
		result.Add(interfaces.Allowed)

	case profile == "":
		// Missing required profile
		result.Add(interfaces.Forbidden, field.Forbidden(fieldPath, "AppArmor profile must be set"))

	default:
		// Disallowed value
		msg := fmt.Sprintf("%s is not an allowed profile. Allowed values: %q", profile, s.allowedProfilesString)
		result.Add(interfaces.Forbidden, field.Forbidden(fieldPath, msg))
	}

	return result, nil
}

func (s *strategy) DefaultContainer(pod *api.Pod, container *api.Container) error {
	if s.defaultProfile == "" {
		return nil
	}
	if pod.Annotations[apparmor.ContainerAnnotationKeyPrefix+container.Name] != "" {
		return nil
	}
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[apparmor.ContainerAnnotationKeyPrefix+container.Name] = s.defaultProfile
	return nil
}
