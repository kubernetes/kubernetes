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

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

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
