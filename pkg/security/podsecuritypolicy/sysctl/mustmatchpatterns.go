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

package sysctl

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
)

// mustMatchPatterns implements the CapabilitiesStrategy interface
type mustMatchPatterns struct {
	patterns []string
}

var (
	_ SysctlsStrategy = &mustMatchPatterns{}

	defaultSysctlsPatterns = []string{"*"}
)

// NewMustMatchPatterns creates a new mustMatchPatterns strategy that will provide validation.
// Passing nil means the default pattern, passing an empty list means to disallow all sysctls.
func NewMustMatchPatterns(patterns []string) SysctlsStrategy {
	if patterns == nil {
		patterns = defaultSysctlsPatterns
	}
	return &mustMatchPatterns{
		patterns: patterns,
	}
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustMatchPatterns) Validate(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, s.validateAnnotation(pod, api.SysctlsPodAnnotationKey)...)
	allErrs = append(allErrs, s.validateAnnotation(pod, api.UnsafeSysctlsPodAnnotationKey)...)
	return allErrs
}

func (s *mustMatchPatterns) validateAnnotation(pod *api.Pod, key string) field.ErrorList {
	allErrs := field.ErrorList{}

	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(key)

	sysctls, err := api.SysctlsFromPodAnnotation(pod.Annotations[key])
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fieldPath, pod.Annotations[key], err.Error()))
	}

	if len(sysctls) > 0 {
		if len(s.patterns) == 0 {
			allErrs = append(allErrs, field.Invalid(fieldPath, pod.Annotations[key], "sysctls are not allowed"))
		} else {
			for i, sysctl := range sysctls {
				allErrs = append(allErrs, s.ValidateSysctl(sysctl.Name, fieldPath.Index(i))...)
			}
		}
	}

	return allErrs
}

func (s *mustMatchPatterns) ValidateSysctl(sysctlName string, fldPath *field.Path) field.ErrorList {
	for _, s := range s.patterns {
		if s[len(s)-1] == '*' {
			prefix := s[:len(s)-1]
			if strings.HasPrefix(sysctlName, string(prefix)) {
				return nil
			}
		} else if sysctlName == s {
			return nil
		}
	}
	return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("sysctl %q is not allowed", sysctlName))}
}
