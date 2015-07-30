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

package securitycontext

import (
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

// HasPrivilegedRequest returns the value of SecurityContext.Privileged, taking into account
// the possibility of nils
func HasPrivilegedRequest(container *api.Container) bool {
	if container.SecurityContext == nil {
		return false
	}
	if container.SecurityContext.Privileged == nil {
		return false
	}
	return *container.SecurityContext.Privileged
}

// HasCapabilitiesRequest returns true if Adds or Drops are defined in the security context
// capabilities, taking into account nils
func HasCapabilitiesRequest(container *api.Container) bool {
	if container.SecurityContext == nil {
		return false
	}
	if container.SecurityContext.Capabilities == nil {
		return false
	}
	return len(container.SecurityContext.Capabilities.Add) > 0 || len(container.SecurityContext.Capabilities.Drop) > 0
}

const expectedSELinuxContextFields = 4

// ParseSELinuxOptions parses a string containing a full SELinux context
// (user, role, type, and level) into an SELinuxOptions object.  If the
// context is malformed, an error is returned.
func ParseSELinuxOptions(context string) (*api.SELinuxOptions, error) {
	fields := strings.SplitN(context, ":", expectedSELinuxContextFields)

	if len(fields) != expectedSELinuxContextFields {
		return nil, fmt.Errorf("expected %v fields in selinuxcontext; got %v (context: %v)", expectedSELinuxContextFields, len(fields), context)
	}

	return &api.SELinuxOptions{
		User:  fields[0],
		Role:  fields[1],
		Type:  fields[2],
		Level: fields[3],
	}, nil
}
