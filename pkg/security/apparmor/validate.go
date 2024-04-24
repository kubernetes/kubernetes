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
	"errors"
	"fmt"
	"strings"

	"github.com/opencontainers/runc/libcontainer/apparmor"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
)

// Whether AppArmor should be disabled by default.
// Set to true if the wrong build tags are set (see validate_disabled.go).
var isDisabledBuild bool

// Validator is a interface for validating that a pod with an AppArmor profile can be run by a Node.
type Validator interface {
	Validate(pod *v1.Pod) error
	ValidateHost() error
}

// NewValidator is in order to find AppArmor FS
func NewValidator() Validator {
	if err := validateHost(); err != nil {
		return &validator{validateHostErr: err}
	}
	return &validator{}
}

type validator struct {
	validateHostErr error
}

func (v *validator) Validate(pod *v1.Pod) error {
	if !isRequired(pod) {
		return nil
	}

	if v.ValidateHost() != nil {
		return v.validateHostErr
	}

	var retErr error
	podutil.VisitContainers(&pod.Spec, podutil.AllContainers, func(container *v1.Container, containerType podutil.ContainerType) bool {
		profile := GetProfile(pod, container)
		if profile == nil {
			return true
		}

		// TODO(#64841): This would ideally be part of validation.ValidateAppArmorProfileFormat, but
		// that is called for API validation, and this is tightening validation.
		if profile.Type == v1.AppArmorProfileTypeLocalhost {
			if profile.LocalhostProfile == nil || strings.TrimSpace(*profile.LocalhostProfile) == "" {
				retErr = fmt.Errorf("invalid empty AppArmor profile name: %q", profile)
				return false
			}
		}
		return true
	})

	return retErr
}

// ValidateHost verifies that the host and runtime is capable of enforcing AppArmor profiles.
// Note, this is intentionally only check the host at kubelet startup and never re-evaluates the host
// as the expectation is that the kubelet restart will be needed to enable or disable AppArmor support.
func (v *validator) ValidateHost() error {
	return v.validateHostErr
}

// validateHost verifies that the host and runtime is capable of enforcing AppArmor profiles.
func validateHost() error {
	// Check feature-gates
	if !utilfeature.DefaultFeatureGate.Enabled(features.AppArmor) {
		return errors.New("AppArmor disabled by feature-gate")
	}

	// Check build support.
	if isDisabledBuild {
		return errors.New("binary not compiled for linux")
	}

	// Check kernel support.
	if !apparmor.IsEnabled() {
		return errors.New("AppArmor is not enabled on the host")
	}

	return nil
}
