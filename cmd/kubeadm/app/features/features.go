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

	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (
	// HighAvailability is alpha in v1.9 - deprecated in v1.12 (TODO remove in v1.13)
	HighAvailability = "HighAvailability"

	// CoreDNS is GA in v1.11
	CoreDNS = "CoreDNS"

	// SelfHosting is alpha in v1.8 and v1.9 - deprecated in v1.12 (TODO remove in v1.13)
	SelfHosting = "SelfHosting"

	// StoreCertsInSecrets is alpha in v1.8 and v1.9 - deprecated in v1.12 (TODO remove in v1.13)
	StoreCertsInSecrets = "StoreCertsInSecrets"

	// DynamicKubeletConfig is beta in v1.11
	DynamicKubeletConfig = "DynamicKubeletConfig"

	// Auditing is beta in 1.8
	Auditing = "Auditing"
)

var selfHostingDeprecationMessage = "featureGates:SelfHosting has been removed in v1.12"

var storeCertsInSecretsDeprecationMessage = "featureGates:StoreCertsInSecrets has been removed in v1.12"

var highAvailabilityMessage = "featureGates:HighAvailability has been removed in v1.12\n" +
	"\tThis feature has been replaced by the kubeadm join --control-plane workflow."

// InitFeatureGates are the default feature gates for the init command
var InitFeatureGates = FeatureList{
	SelfHosting:          {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Deprecated}, HiddenInHelpText: true, DeprecationMessage: selfHostingDeprecationMessage},
	StoreCertsInSecrets:  {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Deprecated}, HiddenInHelpText: true, DeprecationMessage: storeCertsInSecretsDeprecationMessage},
	HighAvailability:     {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Deprecated}, HiddenInHelpText: true, DeprecationMessage: highAvailabilityMessage},
	CoreDNS:              {FeatureSpec: utilfeature.FeatureSpec{Default: true, PreRelease: utilfeature.GA}},
	DynamicKubeletConfig: {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Beta}},
	Auditing:             {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Alpha}},
}

// Feature represents a feature being gated
type Feature struct {
	utilfeature.FeatureSpec
	MinimumVersion     *version.Version
	HiddenInHelpText   bool
	DeprecationMessage string
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
	if enabled, ok := featureList[string(featureName)]; ok {
		return enabled
	}
	return InitFeatureGates[string(featureName)].Default
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
		if v.HiddenInHelpText {
			continue
		}

		pre := ""
		if v.PreRelease != utilfeature.GA {
			pre = fmt.Sprintf("%s - ", v.PreRelease)
		}
		known = append(known, fmt.Sprintf("%s=true|false (%sdefault=%t)", k, pre, v.Default))
	}
	sort.Strings(known)
	return known
}

// NewFeatureGate parses a string of the form "key1=value1,key2=value2,..." into a
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

		featureSpec, ok := (*f)[k]
		if !ok {
			return nil, fmt.Errorf("unrecognized feature-gate key: %s", k)
		}

		if featureSpec.PreRelease == utilfeature.Deprecated {
			return nil, fmt.Errorf("feature-gate key is deprecated: %s", k)
		}

		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return nil, fmt.Errorf("invalid value %v for feature-gate key: %s, use true|false instead", v, k)
		}
		featureGate[k] = boolValue
	}

	ResolveFeatureGateDependencies(featureGate)

	return featureGate, nil
}

// CheckDeprecatedFlags takes a list of existing feature gate flags and validates against the current feature flag set.
// It used during upgrades for ensuring consistency of feature gates used in an existing cluster, that might
// be created with a previous version of kubeadm, with the set of features currently supported by kubeadm
func CheckDeprecatedFlags(f *FeatureList, features map[string]bool) map[string]string {
	deprecatedMsg := map[string]string{}
	for k := range features {
		featureSpec, ok := (*f)[k]
		if !ok {
			// This case should never happen, it is implemented only as a sentinel
			// for removal of flags executed when flags are still in use (always before deprecate, then after one cycle remove)
			deprecatedMsg[k] = fmt.Sprintf("Unknown feature gate flag: %s", k)
		}

		if featureSpec.PreRelease == utilfeature.Deprecated {
			if _, ok := deprecatedMsg[k]; !ok {
				deprecatedMsg[k] = featureSpec.DeprecationMessage
			}
		}
	}

	return deprecatedMsg
}

// ResolveFeatureGateDependencies resolve dependencies between feature gates
func ResolveFeatureGateDependencies(featureGate map[string]bool) {

	// if HighAvailability enabled and StoreCertsInSecrets disabled, both StoreCertsInSecrets
	// and SelfHosting should enabled
	if Enabled(featureGate, HighAvailability) && !Enabled(featureGate, StoreCertsInSecrets) {
		featureGate[StoreCertsInSecrets] = true
	}

	// if StoreCertsInSecrets enabled, SelfHosting should enabled
	if Enabled(featureGate, StoreCertsInSecrets) {
		featureGate[SelfHosting] = true
	}
}
