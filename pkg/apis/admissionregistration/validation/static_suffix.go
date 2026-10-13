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

package validation

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

// StaticConfigSuffix is the reserved suffix for manifest-based admission configurations.
// Resources with names ending in this suffix can only be created via static manifest
// files loaded at API server startup, not through the REST API.
// NOTE: This constant is duplicated in staging/src/k8s.io/apiserver/pkg/admission/plugin/manifest/validation.go
// because this package cannot import from staging. Keep both in sync.
const StaticConfigSuffix = ".static.k8s.io"

// ValidateStaticSuffix validates that the name does not end with the reserved static suffix.
// When the ManifestBasedAdmissionControlConfig feature gate is enabled, returns an error
// if the name uses the reserved suffix.
func ValidateStaticSuffix(name string, fldPath *field.Path) field.ErrorList {
	var allErrors field.ErrorList
	if strings.HasSuffix(name, StaticConfigSuffix) {
		allErrors = append(allErrors, field.Invalid(fldPath, name,
			fmt.Sprintf("names ending with %q are reserved for static manifest-based configurations and cannot be managed via the API", StaticConfigSuffix)))
	}
	return allErrors
}

// WarningsForStaticSuffix returns warnings if the name uses the reserved static suffix.
func WarningsForStaticSuffix(name string) []string {
	if strings.HasSuffix(name, StaticConfigSuffix) {
		return []string{
			fmt.Sprintf("metadata.name: names ending with %q are reserved for static manifest-based configurations and cannot be managed via the API", StaticConfigSuffix),
		}
	}
	return nil
}
