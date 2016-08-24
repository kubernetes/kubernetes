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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// mustMatchPatterns implements the CapabilitiesStrategy interface
type mustMatchPatterns struct {
	patterns []string
}

var (
	_ SysctlsStrategy = &mustMatchPatterns{}

	defaultSysctlsPatterns = []string{"*"}
)

// NewMustMatchPatterns creates a new mustMatchPattern strategy that will provide validation.
// Passing nil means the default pattern, passing an empty list means to disallow all sysctls.
func NewMustMatchPatterns(patterns []string) (SysctlsStrategy, error) {
	if patterns == nil {
		patterns = defaultSysctlsPatterns
	}
	return &mustMatchPatterns{
		patterns: patterns,
	}, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustMatchPatterns) Validate(pod *api.Pod) field.ErrorList {
	allErrs := field.ErrorList{}

	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.UnsafeSysctlsPodAnnotationKey)

	sysctlAnn := pod.Annotations[api.UnsafeSysctlsPodAnnotationKey]
	unsafeSysctls, err := api.SysctlsFromPodAnnotation(sysctlAnn)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fieldPath, sysctlAnn, err.Error()))
	}

	if len(unsafeSysctls) > 0 {
		if len(s.patterns) == 0 {
			allErrs = append(allErrs, field.Invalid(fieldPath, sysctlAnn, "unsafe sysctls are not allowed"))
		} else {
			for i, sysctl := range unsafeSysctls {
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
	return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("unsafe sysctl %q is not allowed", sysctlName))}
}
