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

package api

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func isSupportedManifestVersion(value string) bool {
	switch value {
	case "v1beta1", "v1beta2":
		return true
	}
	return false
}

func isInvalid(field string, value interface{}) error {
	return fmt.Errorf("%s is invalid: '%v'", field, value)
}

func isNotSupported(field string, value interface{}) error {
	return fmt.Errorf("%s is not supported: '%v'", field, value)
}

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidateManifest(manifest *ContainerManifest) error {
	if len(manifest.Version) == 0 {
		return isInvalid("ContainerManifest.Version", manifest.Version)
	}
	if !isSupportedManifestVersion(manifest.Version) {
		return isNotSupported("ContainerManifest.Version", manifest.Version)
	}
	if len(manifest.ID) > 255 || !util.IsDNSSubdomain(manifest.ID) {
		return isInvalid("ContainerManifest.ID", manifest.ID)
	}
	// TODO(thockin): finish validation.
	return nil
}
