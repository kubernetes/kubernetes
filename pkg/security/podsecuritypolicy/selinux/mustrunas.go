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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

type mustRunAs struct {
	opts *extensions.SELinuxStrategyOptions
}

var _ SELinuxStrategy = &mustRunAs{}

func NewMustRunAs(options *extensions.SELinuxStrategyOptions) (SELinuxStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxContextStrategyOptions")
	}
	if options.SELinuxOptions == nil {
		return nil, fmt.Errorf("MustRunAs requires SELinuxOptions")
	}
	return &mustRunAs{
		opts: options,
	}, nil
}

// Generate creates the SELinuxOptions based on constraint rules.
func (s *mustRunAs) Generate(_ *api.Pod, _ *api.Container) (*api.SELinuxOptions, error) {
	return s.opts.SELinuxOptions, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(fldPath *field.Path, _ *api.Pod, _ *api.Container, seLinux *api.SELinuxOptions) field.ErrorList {
	allErrs := field.ErrorList{}

	if seLinux == nil {
		allErrs = append(allErrs, field.Required(fldPath, ""))
		return allErrs
	}
	if seLinux.Level != s.opts.SELinuxOptions.Level {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Level)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("level"), seLinux.Level, detail))
	}
	if seLinux.Role != s.opts.SELinuxOptions.Role {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Role)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("role"), seLinux.Role, detail))
	}
	if seLinux.Type != s.opts.SELinuxOptions.Type {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.Type)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("type"), seLinux.Type, detail))
	}
	if seLinux.User != s.opts.SELinuxOptions.User {
		detail := fmt.Sprintf("must be %s", s.opts.SELinuxOptions.User)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("user"), seLinux.User, detail))
	}

	return allErrs
}
