/*
Copyright 2025 The Kubernetes Authors.

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

package features

import (
	"fmt"
	"regexp"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features/inplacepodresize"
)

// featureNameRegexp defines the allowed format for feature names.
// The first segment must be in UpperCamelCase. Subsequent segments (separated by '/')
// can be in either UpperCamelCase or lowerCamelCase.
var featureNameRegexp = regexp.MustCompile(`^[A-Z][a-zA-Z0-9]*(\/[a-zA-Z][a-zA-Z0-9]*)*$`)

// AllFeatures is the central registry for all declared features.
// New features are added to this list to be automatically included in both
// discovery and inference logic.
var AllFeatures = []nodedeclaredfeatures.Feature{
	inplacepodresize.Feature,
}

func init() {
	if err := ValidateFeatures(); err != nil {
		panic(err)
	}
}

func ValidateFeatures() error {
	// Perform validation on all registered features at startup.
	for i, feature := range AllFeatures {
		featureName := feature.Name()
		if len(featureName) > validation.DNS1123SubdomainMaxLength {
			return fmt.Errorf("invalid feature name %q: must be no more than %d characters", featureName, validation.DNS1123SubdomainMaxLength)
		}
		if !featureNameRegexp.MatchString(featureName) {
			return fmt.Errorf("invalid feature name %q: must start with an UpperCamelCase segment, with subsequent segments separated by '/' (e.g., MyFeature or MyFeature/mySubFeature), and contain only alphanumeric characters and slashes", featureName)
		}

		// Check for duplicate feature names.
		for j := i + 1; j < len(AllFeatures); j++ {
			if feature.Name() == AllFeatures[j].Name() {
				return fmt.Errorf("duplicate feature name %q", featureName)
			}
		}
	}
	return nil
}
