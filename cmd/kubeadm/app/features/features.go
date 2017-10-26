/*
Copyright 2017 The Kubernetes Authors.

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
	"sort"
	"strconv"
	"strings"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	// HighAvailability is alpha in v1.9
	HighAvailability = "HighAvailability"

	// SelfHosting is beta in v1.8
	SelfHosting = "SelfHosting"

	// StoreCertsInSecrets is alpha in v1.8
	StoreCertsInSecrets = "StoreCertsInSecrets"
)

var v190 = version.MustParseSemantic("v1.9.0")

// InitFeatureGates are the default feature gates for the init command
var InitFeatureGates = FeatureList{
	SelfHosting:         {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Beta}},
	StoreCertsInSecrets: {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Alpha}},
	HighAvailability:    {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Alpha}, MinimumVersion: v190},
}

// Feature represents a feature being gated
type Feature struct {
	utilfeature.FeatureSpec
	MinimumVersion *version.Version
}

// FeatureList represents a list of feature gates
type FeatureList map[string]Feature

// ValidateVersion ensures that a feature gate list is compatible with the chosen kubernetes version
func ValidateVersion(allFeatures FeatureList, requestedFeatures map[string]bool, requestedVersion string) error {
	if requestedVersion == "" {
		return nil
	}
	parsedExpVersion, err := version.ParseSemantic(requestedVersion)
	if err != nil {
		return fmt.Errorf("Error parsing version %s: %v", requestedVersion, err)
	}
	for k := range requestedFeatures {
		if minVersion := allFeatures[k].MinimumVersion; minVersion != nil {
			if !parsedExpVersion.AtLeast(minVersion) {
				return fmt.Errorf(
					"the requested kubernetes version (%s) is incompatible with the %s feature gate, which needs %s as a minimum",
					requestedVersion, k, minVersion)
			}
		}
	}
	return nil
}

// Enabled indicates whether a feature name has been enabled
func Enabled(featureList map[string]bool, featureName string) bool {
	return featureList[string(featureName)]
}

// Supports indicates whether a feature name is supported on the given
// feature set
func Supports(featureList FeatureList, featureName string) bool {
	for k := range featureList {
		if featureName == string(k) {
			return true
		}
	}
	return false
}

// Keys returns a slice of feature names for a given feature set
func Keys(featureList FeatureList) []string {
	var list []string
	for k := range featureList {
		list = append(list, string(k))
	}
	return list
}

// KnownFeatures returns a slice of strings describing the FeatureList features.
func KnownFeatures(f *FeatureList) []string {
	var known []string
	for k, v := range *f {
		pre := ""
		if v.PreRelease != utilfeature.GA {
			pre = fmt.Sprintf("%s - ", v.PreRelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.Default))
	}
	sort.Strings(known)
	return known
}

// NewFeatureGate parse a string of the form "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func NewFeatureGate(f *FeatureList, value string) (map[string]bool, error) {
	featureGate := map[string]bool{}
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}

		arr := strings.SplitN(s, "=", 2)
		if len(arr) != 2 {
			return nil, fmt.Errorf("missing bool value for feature-gate key:%s", s)
		}

		k := strings.TrimSpace(arr[0])
		v := strings.TrimSpace(arr[1])

		if !Supports(*f, k) {
			return nil, fmt.Errorf("unrecognized feature-gate key: %s", k)
		}

		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("invalid value %v for feature-gate key: %s, use true|false instead", v, k)
		}
		featureGate[k] = boolValue
	}

	return featureGate, nil
}
