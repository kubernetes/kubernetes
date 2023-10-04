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

package featuregate

import (
	"fmt"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/naming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

// Reader indicates whether a given feature is enabled or not.
type Reader interface {
	// Enabled returns true if the key is enabled.
	Enabled(key Feature) bool

	// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
	KnownFeatures() []string
}

var _ Reader = &envVarFeatureGate{}

// NewEnvVarFeatureGate creates a feature gate that allows for registration
// of features and checking if the features are enabled.
// On the first check, it reads environment variables prefixed
// with "KUBE_FEATURE_" and the feature name to determine the feature's state.
//
// For example, if you have a feature named "MyFeature,"
// setting an environmental variable "KUBE_FEATURE_MyFeature"
// will allow you to configure the state of that feature.
//
// Please note that environmental variables can only be set to a boolean type.
// Incorrect values will be ignored and logged.
func NewEnvVarFeatureGate() *envVarFeatureGate {
	known := map[Feature]FeatureSpec{}
	knownValue := &atomic.Value{}
	knownValue.Store(known)

	enabled := map[Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	f := &envVarFeatureGate{
		envVarFeatureGateName: naming.GetNameFromCallsite(internalPackages...),
		known:                 knownValue,
		enabled:               enabledValue,
	}
	return f
}

// envVarFeatureGate implements Reader and allows for feature registration.
type envVarFeatureGate struct {
	// envVarFeatureGateName holds the name of the file
	// that created this instance
	envVarFeatureGateName string

	// readEnvVarsOnce guards reading environmental variables
	readEnvVarsOnce sync.Once

	// lock guards writes to known, enabled
	lock sync.Mutex

	// known holds a map[Feature]FeatureSpec
	known *atomic.Value

	// enabled holds a map[Feature]bool
	// holds values set explicitly via env var
	enabled *atomic.Value
}

// Add adds features to the envVarFeatureGate.
func (f *envVarFeatureGate) Add(features map[Feature]FeatureSpec) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	known := map[Feature]FeatureSpec{}
	for k, v := range f.getKnownFeatures() {
		known[k] = v
	}

	for name, spec := range features {
		if existingSpec, found := known[name]; found {
			if existingSpec == spec {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}
		known[name] = spec
	}

	f.known.Store(known)
	return nil
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGate) Enabled(key Feature) bool {
	if v, ok := f.getEnabledMapFromEnvVar()[key]; ok {
		return v
	}
	if v, ok := f.getKnownFeatures()[key]; ok {
		return v.Default
	}
	panic(fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.envVarFeatureGateName))
}

// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
// Deprecated and GA features are hidden from the list.
func (f *envVarFeatureGate) KnownFeatures() []string {
	var known []string
	for k, v := range f.getKnownFeatures() {
		if v.PreRelease == GA || v.PreRelease == Deprecated {
			continue
		}
		known = append(known, fmt.Sprintf("%s=true|false (%s - default=%t)", k, v.PreRelease, v.Default))
	}
	sort.Strings(known)
	return known
}

// getEnabledMapFromEnvVar will fill the enabled map on the first call.
// This is the only time a known feature can be set to a value
// read from the corresponding environmental variable.
func (f *envVarFeatureGate) getEnabledMapFromEnvVar() map[Feature]bool {
	f.readEnvVarsOnce.Do(func() {
		featureGateState := map[Feature]bool{}
		for feature, featureSpec := range f.getKnownFeatures() {
			featureState, featureStateSet := os.LookupEnv(fmt.Sprintf("KUBE_FEATURE_%s", feature))
			if !featureStateSet {
				continue
			}
			boolVal, boolErr := strconv.ParseBool(featureState)
			switch {
			case boolErr != nil:
				utilruntime.HandleError(fmt.Errorf("cannot set feature gate %v to %v, due to %v", feature, featureState, boolErr))
			case featureSpec.LockToDefault:
				if boolVal != featureSpec.Default {
					utilruntime.HandleError(fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", feature, featureState, featureSpec.Default))
					break
				}
				featureGateState[feature] = featureSpec.Default
			default:
				featureGateState[feature] = boolVal
			}
		}
		f.enabled.Store(featureGateState)
	})
	return f.enabled.Load().(map[Feature]bool)
}

// getKnownFeatures returns a copy of the map of known feature names to feature specs.
func (f *envVarFeatureGate) getKnownFeatures() map[Feature]FeatureSpec {
	retval := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		retval[k] = v
	}
	return retval
}
