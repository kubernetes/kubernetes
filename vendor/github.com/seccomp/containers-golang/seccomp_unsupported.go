// +build !seccomp

// SPDX-License-Identifier: Apache-2.0

// Copyright 2013-2018 Docker, Inc.

package seccomp // import "github.com/seccomp/containers-golang"

import (
	"fmt"

	"github.com/opencontainers/runtime-spec/specs-go"
)

// DefaultProfile returns a nil pointer on unsupported systems.
func DefaultProfile() *Seccomp {
	return nil
}

// LoadProfile returns an error on unsuppored systems
func LoadProfile(body string, rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	return nil, fmt.Errorf("Seccomp not supported on this platform")
}

// GetDefaultProfile returns an error on unsuppored systems
func GetDefaultProfile(rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	return nil, fmt.Errorf("Seccomp not supported on this platform")
}

// LoadProfileFromBytes takes a byte slice and decodes the seccomp profile.
func LoadProfileFromBytes(body []byte, rs *specs.Spec) (*specs.LinuxSeccomp, error) {
	return nil, fmt.Errorf("Seccomp not supported on this platform")
}

// LoadProfileFromConfig takes a Seccomp struct and a spec to retrieve a LinuxSeccomp
func LoadProfileFromConfig(config *Seccomp, specgen *specs.Spec) (*specs.LinuxSeccomp, error) {
	return nil, fmt.Errorf("Seccomp not supported on this platform")
}

// IsEnabled returns true if seccomp is enabled for the host.
func IsEnabled() bool {
	return false
}
