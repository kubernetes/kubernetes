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

package sysctls

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation/field"
	"strings"
)

// mustMatchPatterns implements the CapabilitiesStrategy interface
type mustMatchPatterns struct {
	patterns []string
}

var (
	_ SysctlsStrategy = &mustMatchPatterns{}

	defaultSysctlsPatterns = []string{"*"}
)

// NewMustMatchPattrens creates a new mustMatchPattern strategy that will provide validation.
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

	fieldPath := field.NewPath("pod", "metadata", "annotations").Key(api.SysctlsPodAnnotationKey)

	sysctls, err := api.SysctlsFromPodAnnotation(pod.Annotations[api.SysctlsPodAnnotationKey])
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fieldPath, pod.Annotations[api.SysctlsPodAnnotationKey], err.Error()))
	}

	if len(sysctls) > 0 {
		if s.patterns == nil {
			allErrs = append(allErrs, field.Invalid(fieldPath, pod.Annotations[api.SysctlsPodAnnotationKey], "Sysctls are not allowed to be used"))
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
			if strings.HasPrefix(sysctlName, prefix) {
				return nil
			}
		} else if sysctlName == s {
			return nil
		}
	}
	return field.ErrorList{field.Forbidden(fldPath, fmt.Sprintf("Sysctl %q is not allowed", sysctlName))}
}
