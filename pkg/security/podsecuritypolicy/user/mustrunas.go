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

package user

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	psputil "k8s.io/kubernetes/pkg/security/podsecuritypolicy/util"
)

// mustRunAs implements the RunAsUserStrategy interface
type mustRunAs struct {
	opts *policy.RunAsUserStrategyOptions
}

// NewMustRunAs provides a strategy that requires the container to run as a specific UID in a range.
func NewMustRunAs(options *policy.RunAsUserStrategyOptions) (RunAsUserStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires run as user options")
	}
	if len(options.Ranges) == 0 {
		return nil, fmt.Errorf("MustRunAs requires at least one range")
	}
	return &mustRunAs{
		opts: options,
	}, nil
}

// Generate creates the uid based on policy rules.  MustRunAs returns the first range's Min.
func (s *mustRunAs) Generate(pod *api.Pod, container *api.Container) (*int64, error) {
	return &s.opts.Ranges[0].Min, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(scPath *field.Path, _ *api.Pod, _ *api.Container, runAsNonRoot *bool, runAsUser *int64) field.ErrorList {
	allErrs := field.ErrorList{}

	if runAsUser == nil {
		allErrs = append(allErrs, field.Required(scPath.Child("runAsUser"), ""))
		return allErrs
	}

	if !s.isValidUID(*runAsUser) {
		detail := fmt.Sprintf("must be in the ranges: %v", s.opts.Ranges)
		allErrs = append(allErrs, field.Invalid(scPath.Child("runAsUser"), *runAsUser, detail))
	}
	return allErrs
}

func (s *mustRunAs) isValidUID(id int64) bool {
	for _, rng := range s.opts.Ranges {
		if psputil.UserFallsInRange(id, rng) {
			return true
		}
	}
	return false
}
