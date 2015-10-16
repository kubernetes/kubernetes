/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation/field"
)

// mustRunAs implements the RunAsUserSecurityContextConstraintsStrategy interface
type mustRunAs struct {
	opts *api.RunAsUserStrategyOptions
}

var _ RunAsUserSecurityContextConstraintsStrategy = &mustRunAs{}

// NewMustRunAs provides a strategy that requires the container to run as a specific UID.
func NewMustRunAs(options *api.RunAsUserStrategyOptions) (RunAsUserSecurityContextConstraintsStrategy, error) {
	if options == nil {
		return nil, fmt.Errorf("MustRunAs requires run as user options")
	}
	if options.UID == nil {
		return nil, fmt.Errorf("MustRunAs requires a UID")
	}
	return &mustRunAs{
		opts: options,
	}, nil
}

// Generate creates the uid based on policy rules.  MustRunAs returns the UID it is initialized with.
func (s *mustRunAs) Generate(pod *api.Pod, container *api.Container) (*int64, error) {
	return s.opts.UID, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *mustRunAs) Validate(pod *api.Pod, container *api.Container) field.ErrorList {
	allErrs := field.ErrorList{}

	securityContextPath := field.NewPath("securityContext")
	if container.SecurityContext == nil {
		detail := fmt.Sprintf("unable to validate nil security context for container %s", container.Name)
		allErrs = append(allErrs, field.Invalid(securityContextPath, container.SecurityContext, detail))
		return allErrs
	}
	if container.SecurityContext.RunAsUser == nil {
		detail := fmt.Sprintf("unable to validate nil runAsUser for container %s", container.Name)
		allErrs = append(allErrs, field.Invalid(securityContextPath.Child("runAsUser"), container.SecurityContext.RunAsUser, detail))
		return allErrs
	}

	if *s.opts.UID != *container.SecurityContext.RunAsUser {
		detail := fmt.Sprintf("UID on container %s does not match required UID.  Found %d, wanted %d", container.Name, *container.SecurityContext.RunAsUser, *s.opts.UID)
		allErrs = append(allErrs, field.Invalid(securityContextPath.Child("runAsUser"), *container.SecurityContext.RunAsUser, detail))
	}

	return allErrs
}
