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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/security/apparmor"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

type withAppArmorProfile struct {
	allowedProfiles []string
}

var _ AppArmorStrategy = &withAppArmorProfile{}

// NewWithAppArmorProfile creates a new strategy that enforces AppArmor profile constraints.
func NewWithAppArmorProfile(allowedProfiles []string) AppArmorStrategy {
	return &withAppArmorProfile{allowedProfiles}
}

func (s *withAppArmorProfile) Generate(pod *api.Pod, container *api.Container) error {
	if apparmor.GetProfileName(pod, container.Name) != "" {
		// Profile already set, nothing to do.
		return nil
	}

	if len(s.allowedProfiles) == 0 {
		// AppArmor disabled, don't set a profile.
		return nil
	}

	apparmor.SetProfileName(pod, container.Name, s.allowedProfiles[0])
	return nil
}

func (s *withAppArmorProfile) Validate(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}
	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(apparmor.ContainerAnnotationKeyPrefix + container.Name)

	profile := apparmor.GetProfileName(pod, container.Name)
	if profile == "" {
		if len(s.allowedProfiles) > 0 {
			allErrs = append(allErrs, field.Forbidden(fieldPath, "AppArmor profile must be set"))
			return allErrs
		}
		return nil
	}

	if len(s.allowedProfiles) == 0 {
		allErrs = append(allErrs, field.Forbidden(fieldPath, "AppArmor profile may not be set"))
		return allErrs
	}

	if !s.isProfileAllowed(profile) {
		msg := fmt.Sprintf("%s is not an allowed profile. Allowed values: %v", profile, s.allowedProfiles)
		allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
	}

	return allErrs
}

func (s *withAppArmorProfile) isProfileAllowed(profileName string) bool {
	for _, p := range s.allowedProfiles {
		if profileName == p {
			return true
		}
	}
	return false
}
