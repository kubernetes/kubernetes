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
)

// internalPackages are packages that ignored when creating a name for featureGates. These packages are in the common
// call chains, so they'd be unhelpful as names.
var internalPackages = []string{"k8s.io/client-go/features/env_var_feature_gate.go"}

// Reader indicates whether a given feature is enabled or not.
type Reader interface {
	// Enabled returns true if the key is enabled.
	Enabled(key Feature) bool
}

var _ Reader = &envVarFeatureGate{}

// newEnvVarFeatureGate returns a new Reader that recognizes the provided features.
//
// On the first call to Enabled, the effective state of all known features is loaded from
// environment variables. The environment variable read for a given feature is formed by
// concatenating the prefix "KUBE_FEATURE_" with the feature's name.
//
// For example, if you have a feature named "MyFeature,"
// setting an environmental variable "KUBE_FEATURE_MyFeature"
// will allow you to configure the state of that feature.
//
// Please note that environmental variables can only be set to a boolean type.
// Incorrect values will be ignored and logged.
func newEnvVarFeatureGate(features map[Feature]FeatureSpec) *envVarFeatureGate {
	enabled := map[Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	known := map[Feature]FeatureSpec{}
	for name, spec := range features {
		known[name] = spec
	}

	return &envVarFeatureGate{
		envVarFeatureGateName: naming.GetNameFromCallsite(internalPackages...),
		known:                 known,
		enabled:               enabledValue,
	}
}

// envVarFeatureGate implements Reader and allows for feature registration.
type envVarFeatureGate struct {
	// envVarFeatureGateName holds the name of the file
	// that created this instance
	envVarFeatureGateName string

	// readEnvVarsOnce guards reading environmental variables
	readEnvVarsOnce sync.Once

	// known holds known feature gates
	known map[Feature]FeatureSpec

	// enabled holds a map[Feature]bool
	// with values explicitly set via env var
	enabled *atomic.Value
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGate) Enabled(key Feature) bool {
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
func (f *envVarFeatureGate) getEnabledMapFromEnvVar() map[Feature]bool {
	f.readEnvVarsOnce.Do(func() {
		featureGateState := map[Feature]bool{}
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
	return f.enabled.Load().(map[Feature]bool)
}
