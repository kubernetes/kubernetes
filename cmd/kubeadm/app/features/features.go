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

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (

	// CoreDNS is GA in v1.11
	CoreDNS = "CoreDNS"

	// DynamicKubeletConfig is beta in v1.11
	DynamicKubeletConfig = "DynamicKubeletConfig"
)

var coreDNSMessage = "featureGates:CoreDNS has been removed in v1.13\n" +
	"\tUse kubeadm-config to select which DNS addon to install."

// InitFeatureGates are the default feature gates for the init command
var InitFeatureGates = FeatureList{
	CoreDNS: {FeatureSpec: utilfeature.FeatureSpec{Default: true, PreRelease: utilfeature.Deprecated}, HiddenInHelpText: true, DeprecationMessage: coreDNSMessage},
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

// ValidateVersion ensures that a feature gate list is compatible with the chosen Kubernetes version
func ValidateVersion(allFeatures FeatureList, requestedFeatures map[string]bool, requestedVersion string) error {
	if requestedVersion == "" {
		return nil
	}
	parsedExpVersion, err := version.ParseSemantic(requestedVersion)
	if err != nil {
		return errors.Wrapf(err, "error parsing version %s", requestedVersion)
	}
	for k := range requestedFeatures {
		if minVersion := allFeatures[k].MinimumVersion; minVersion != nil {
			if !parsedExpVersion.AtLeast(minVersion) {
				return errors.Errorf(
					"the requested Kubernetes version (%s) is incompatible with the %s feature gate, which needs %s as a minimum",
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
	for k, v := range featureList {
		if featureName == string(k) {
			return v.PreRelease != utilfeature.Deprecated
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
			return nil, errors.Errorf("missing bool value for feature-gate key:%s", s)
		}

		k := strings.TrimSpace(arr[0])
		v := strings.TrimSpace(arr[1])

		featureSpec, ok := (*f)[k]
		if !ok {
			return nil, errors.Errorf("unrecognized feature-gate key: %s", k)
		}

		if featureSpec.PreRelease == utilfeature.Deprecated {
			return nil, errors.Errorf("feature-gate key is deprecated: %s", k)
		}

		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return nil, errors.Errorf("invalid value %v for feature-gate key: %s, use true|false instead", v, k)
		}
		featureGate[k] = boolValue
	}

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
