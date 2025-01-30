/*
Copyright 2023 The Kubernetes Authors.

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

package version

import (
	"fmt"
	"sync/atomic"

	utilversion "k8s.io/apimachinery/pkg/util/version"
)

var dynamicGitVersion atomic.Value

func init() {
	// initialize to static gitVersion
	dynamicGitVersion.Store(gitVersion)
}

// SetDynamicVersion overrides the version returned as the GitVersion from Get().
// The specified version must be non-empty, a valid semantic version, and must
// match the major/minor/patch version of the default gitVersion.
func SetDynamicVersion(dynamicVersion string) error {
	if err := ValidateDynamicVersion(dynamicVersion); err != nil {
		return err
	}
	dynamicGitVersion.Store(dynamicVersion)
	return nil
}

// ValidateDynamicVersion ensures the given version is non-empty, a valid semantic version,
// and matched the major/minor/patch version of the default gitVersion.
func ValidateDynamicVersion(dynamicVersion string) error {
	return validateDynamicVersion(dynamicVersion, gitVersion)
}

func validateDynamicVersion(dynamicVersion, defaultVersion string) error {
	if len(dynamicVersion) == 0 {
		return fmt.Errorf("version must not be empty")
	}
	if dynamicVersion == defaultVersion {
		// allow no-op
		return nil
	}
	vRuntime, err := utilversion.ParseSemantic(dynamicVersion)
	if err != nil {
		return err
	}
	// must match major/minor/patch of default version
	var vDefault *utilversion.Version
	if defaultVersion == "v0.0.0-master+$Format:%H$" {
		// special-case the placeholder value which doesn't parse as a semantic version
		vDefault, err = utilversion.ParseSemantic("v0.0.0-master")
	} else {
		vDefault, err = utilversion.ParseSemantic(defaultVersion)
	}
	if err != nil {
		return err
	}
	if vRuntime.Major() != vDefault.Major() || vRuntime.Minor() != vDefault.Minor() || vRuntime.Patch() != vDefault.Patch() {
		return fmt.Errorf("version %q must match major/minor/patch of default version %q", dynamicVersion, defaultVersion)
	}
	return nil
}
