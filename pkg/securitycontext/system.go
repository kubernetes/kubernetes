/*
Copyright 2014 Google Inc. All rights reserved.

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

// This is a temporary holder for the security constraint settings so it can be initialized once and used throughout the
// system as we currently did with the capabilities package.  Eventually this will belong to a namespace/service account
// with a system wide context for general settings.  This replaces the Capabilities system settings that were used
// system wide for allowing/disallowing privileged container requests and setting up the pod sources that are allowed
// to use host networking
package securitycontext

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"sync"
)

var once sync.Once
var securityConstraints *api.SecurityConstraints

// Initialize the capability set.  This can only be done once per binary, subsequent calls are ignored.
func Initialize(s api.SecurityConstraints) {
	// Only do this once
	once.Do(func() {
		// we expect the security constraints to be initialized with the AllowCapabilities and AllowPrivileged options since
		// they are preexisting system options.  We will initialize everything else to some defaults for now unless they
		// are already set.
		if len(s.EnforcementPolicy) == 0 {
			s.EnforcementPolicy = api.SecurityConstraintPolicyReject
		}
		if s.SELinux == nil {
			s.SELinux = &api.SELinuxSecurityConstraints{
				AllowUserLabel:  true,
				AllowRoleLabel:  true,
				AllowTypeLabel:  true,
				AllowLevelLabel: true,
				AllowDisable:    true,
			}
		}
		if s.DefaultSecurityContext == nil {
			s.DefaultSecurityContext = &api.SecurityContext{
				Privileged: false,
			}
		}

		securityConstraints = &s
	})
}

// Setup the capability set.  It wraps Initialize for improving usibility.
func Setup(allowPrivileged bool, hostNetworkSources []string) {
	Initialize(api.SecurityConstraints{
		AllowPrivileged:    allowPrivileged,
		HostNetworkSources: hostNetworkSources,
	})
}

// SetCapabilitiesForTests.  Convenience method for testing.  This should only be called from tests.
func SetForTests(s api.SecurityConstraints) {
	securityConstraints = &s
}

// Returns a read-only copy of the system capabilities.
func Get() api.SecurityConstraints {
	if securityConstraints == nil {
		Initialize(api.SecurityConstraints{
			AllowPrivileged:    false,
			HostNetworkSources: []string{},
		})
	}
	return *securityConstraints
}
