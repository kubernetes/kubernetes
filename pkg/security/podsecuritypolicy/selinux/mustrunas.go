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

package selinux

import (
	"fmt"
	"sort"
	"strings"

	policy "k8s.io/api/policy/v1beta1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

type mustRunAs struct {
	opts *api.SELinuxOptions
}

var _ SELinuxStrategy = &mustRunAs{}

func NewMustRunAs(options *policy.SELinuxStrategyOptions) (SELinuxStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxContextStrategyOptions")
	}
	if options.SELinuxOptions == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxOptions")
	}

	internalSELinuxOptions := &api.SELinuxOptions{}
	if err := v1.Convert_v1_SELinuxOptions_To_core_SELinuxOptions(options.SELinuxOptions, internalSELinuxOptions, nil); err != nil {
		return nil, err
	}
	return &mustRunAs{
		opts: internalSELinuxOptions,
	}, nil
}

// Generate creates the SELinuxOptions based on constraint rules.
func (s *mustRunAs) Generate(_ *api.Pod, _ *api.Container) (*api.SELinuxOptions, error) {
	return s.opts, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(fldPath *field.Path, _ *api.Pod, _ *api.Container, seLinux *api.SELinuxOptions) field.ErrorList {
	allErrs := field.ErrorList{}

	if seLinux == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
		return allErrs
	}
	if !equalLevels(s.opts.Level, seLinux.Level) {
		detail := fmt.Sprintf("must be %s", s.opts.Level)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("level"), seLinux.Level, detail))
	}
	if seLinux.Role != s.opts.Role {
		detail := fmt.Sprintf("must be %s", s.opts.Role)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("role"), seLinux.Role, detail))
	}
	if seLinux.Type != s.opts.Type {
		detail := fmt.Sprintf("must be %s", s.opts.Type)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("type"), seLinux.Type, detail))
	}
	if seLinux.User != s.opts.User {
		detail := fmt.Sprintf("must be %s", s.opts.User)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("user"), seLinux.User, detail))
	}

	return allErrs
}

// equalLevels compares SELinux levels for equality.
func equalLevels(expected, actual string) bool {
	if expected == actual {
		return true
	}
	// "s0:c6,c0" => [ "s0", "c6,c0" ]
	expectedParts := strings.SplitN(expected, ":", 2)
	actualParts := strings.SplitN(actual, ":", 2)

	// both SELinux levels must be in a format "sX:cY"
	if len(expectedParts) != 2 || len(actualParts) != 2 {
		return false
	}

	if !equalSensitivity(expectedParts[0], actualParts[0]) {
		return false
	}

	if !equalCategories(expectedParts[1], actualParts[1]) {
		return false
	}

	return true
}

// equalSensitivity compares sensitivities of the SELinux levels for equality.
func equalSensitivity(expected, actual string) bool {
	return expected == actual
}

// equalCategories compares categories of the SELinux levels for equality.
func equalCategories(expected, actual string) bool {
	expectedCategories := strings.Split(expected, ",")
	actualCategories := strings.Split(actual, ",")

	sort.Strings(expectedCategories)
	sort.Strings(actualCategories)

	return util.EqualStringSlices(expectedCategories, actualCategories)
}
