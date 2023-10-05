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

package features

import (
	"fmt"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/naming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/component-base/featuregate"
)

// internalPackages are packages that ignored when creating a name for featureGates. These packages are in the common
// call chains, so they'd be unhelpful as names.
var internalPackages = []string{"k8s.io/client-go/features/env_var_feature_gate.go"}

// Reader indicates whether a given feature is enabled or not.
type Reader interface {
	// Enabled returns true if the key is enabled.
	Enabled(key featuregate.Feature) bool
}

var _ Reader = &envVarFeatureGate{}

// newEnvVarFeatureGate creates a feature gate that allows for registration
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
func newEnvVarFeatureGate() *envVarFeatureGate {
	enabled := map[featuregate.Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	f := &envVarFeatureGate{
		envVarFeatureGateName: naming.GetNameFromCallsite(internalPackages...),
		known:                 map[featuregate.Feature]featuregate.FeatureSpec{},
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

	// known holds known feature gates
	known map[featuregate.Feature]featuregate.FeatureSpec

	// enabled holds a map[Feature]bool
	// with values explicitly set via env var
	enabled *atomic.Value
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGate) Enabled(key featuregate.Feature) bool {
	if v, ok := f.getEnabledMapFromEnvVar()[key]; ok {
		return v
	}
	if v, ok := f.known[key]; ok {
		return v.Default
	}
	panic(fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.envVarFeatureGateName))
}

// getEnabledMapFromEnvVar will fill the enabled map on the first call.
// This is the only time a known feature can be set to a value
// read from the corresponding environmental variable.
func (f *envVarFeatureGate) getEnabledMapFromEnvVar() map[featuregate.Feature]bool {
	f.readEnvVarsOnce.Do(func() {
		featureGateState := map[featuregate.Feature]bool{}
		for feature, featureSpec := range f.known {
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
	return f.enabled.Load().(map[featuregate.Feature]bool)
}

// add persist features in the envVarFeatureGate.
// note that this method is not thread safe.
// it is meant to only be called during pkg initialisation.
func (f *envVarFeatureGate) add(features map[featuregate.Feature]featuregate.FeatureSpec) error {
	existing := map[featuregate.Feature]featuregate.FeatureSpec{}
	for k, v := range f.known {
		existing[k] = v
	}

	for name, spec := range features {
		if existingSpec, found := existing[name]; found {
			if existingSpec == spec {
				continue
			}
			return fmt.Errorf("feature gate %q with different spec already exists: %v", name, existingSpec)
		}
		existing[name] = spec
	}

	f.known = existing
	return nil
}
