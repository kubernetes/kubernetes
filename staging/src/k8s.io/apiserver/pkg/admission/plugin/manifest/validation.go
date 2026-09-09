/*
Copyright The Kubernetes Authors.

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

// Package manifest provides shared utilities for loading admission configurations
// from static manifest files.
package manifest

import (
	"fmt"
	"os"
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// StaticConfigSuffix is the reserved suffix for manifest-based admission configurations.
// Resources with names ending in this suffix can only be created via static manifest
// files loaded at API server startup, not through the REST API.
// NOTE: This constant is duplicated in pkg/apis/admissionregistration/validation/static_suffix.go
// because that package cannot import from staging. Keep both in sync.
const StaticConfigSuffix = ".static.k8s.io"

// ValidateStaticManifestsDir validates the staticManifestsDir config field.
// It checks the feature gate is enabled, the path is absolute, exists, and is a directory.
func ValidateStaticManifestsDir(staticManifestsDir string) error {
	if len(staticManifestsDir) > 0 {
		if !utilfeature.DefaultFeatureGate.Enabled(features.ManifestBasedAdmissionControlConfig) {
			return field.Forbidden(field.NewPath("staticManifestsDir"), "staticManifestsDir requires the ManifestBasedAdmissionControlConfig feature gate to be enabled")
		}
		if !path.IsAbs(staticManifestsDir) {
			return field.Invalid(field.NewPath("staticManifestsDir"), staticManifestsDir, "must be an absolute file path")
		}
		info, err := os.Stat(staticManifestsDir)
		if err != nil {
			return field.Invalid(field.NewPath("staticManifestsDir"), staticManifestsDir, fmt.Sprintf("unable to read: %v", err))
		}
		if !info.IsDir() {
			return field.Invalid(field.NewPath("staticManifestsDir"), staticManifestsDir, "must be a directory")
		}
	}
	return nil
}

// ValidateManifestName checks that the object name is non-empty, has the required
// .static.k8s.io suffix, and is unique within the manifest set.
func ValidateManifestName(name, filePath string, seenNames map[string]string) error {
	if len(name) == 0 {
		return fmt.Errorf("resource in file %q must have a name", filePath)
	}
	if !strings.HasSuffix(name, StaticConfigSuffix) {
		return fmt.Errorf("%q in file %q must have a name ending with %q", name, filePath, StaticConfigSuffix)
	}
	if prevFile, ok := seenNames[name]; ok {
		return fmt.Errorf("duplicate name %q found in file %q (previously seen in %q)", name, filePath, prevFile)
	}
	seenNames[name] = filePath
	return nil
}
