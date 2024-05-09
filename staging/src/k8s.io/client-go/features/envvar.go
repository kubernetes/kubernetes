/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/naming"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
)

// internalPackages are packages that ignored when creating a name for featureGates. These packages are in the common
// call chains, so they'd be unhelpful as names.
var internalPackages = []string{"k8s.io/client-go/features/envvar.go"}

var _ Gates = &envVarFeatureGates{}

// NewEnvVarFeatureGates creates a feature gate that allows for registration
// of features and checking if the features are enabled.
//
// On the first call to Enabled, the effective state of all known features is loaded from
// environment variables. The environment variable read for a given feature is formed by
// concatenating the prefix "KUBE_FEATURE_" with the feature's name.
//
// For example, if you have a feature named "MyFeature"
// setting an environmental variable "KUBE_FEATURE_MyFeature"
// will allow you to configure the state of that feature.
//
// Please note that environmental variables can only be set to the boolean value.
// Incorrect values will be ignored and logged.
func NewEnvVarFeatureGates(features map[Feature]FeatureSpec) *envVarFeatureGates {
	known := map[Feature]FeatureSpec{}
	for name, spec := range features {
		known[name] = spec
	}

	fg := &envVarFeatureGates{
		callSiteName: naming.GetNameFromCallsite(internalPackages...),
		known:        known,
	}
	fg.enabled.Store(map[Feature]bool{})

	return fg
}

// envVarFeatureGates implements Gates and allows for feature registration.
type envVarFeatureGates struct {
	// callSiteName holds the name of the file
	// that created this instance
	callSiteName string

	// readEnvVarsOnce guards reading environmental variables
	readEnvVarsOnce sync.Once

	// known holds known feature gates
	known map[Feature]FeatureSpec

	lock sync.Mutex

	// enabled holds a map[Feature]bool
	// with values explicitly set via env var
	enabled atomic.Value

	// readEnvVars holds the boolean value which
	// indicates whether readEnvVarsOnce has been called.
	readEnvVars atomic.Bool
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGates) Enabled(key Feature) bool {
	if v, ok := f.getEnabledMapFromEnvVar()[key]; ok {
		return v
	}
	if v, ok := f.known[key]; ok {
		return v.Default
	}
	panic(fmt.Errorf("feature %q is not registered in FeatureGates %q", key, f.callSiteName))
}

// Set parses a string of the form "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *envVarFeatureGates) Set(value string) error {
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := strings.TrimSpace(arr[0])
		if len(arr) != 2 {
			return fmt.Errorf("missing bool value for %s", k)
		}
		v := strings.TrimSpace(arr[1])
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid value of %s=%s, err: %v", k, v, err)
		}

		gateName := Feature(k)
		featureSpec, ok := f.known[gateName]
		if !ok {
			return fmt.Errorf("unrecognized feature gate: %s", k)
		}
		if featureSpec.LockToDefault && featureSpec.Default != boolValue {
			return fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", k, v, featureSpec.Default)
		}

		func() {
			f.lock.Lock()
			defer f.lock.Unlock()
			currGates := f.enabled.Load().(map[Feature]bool)
			currGates[Feature(k)] = boolValue
			f.enabled.Store(currGates)
		}()
	}

	return nil
}

// getEnabledMapFromEnvVar will fill the enabled map on the first call.
// This is the only time a known feature can be set to a value
// read from the corresponding environmental variable.
func (f *envVarFeatureGates) getEnabledMapFromEnvVar() map[Feature]bool {
	f.readEnvVarsOnce.Do(func() {
		featureGatesState := map[Feature]bool{}
		for feature, featureSpec := range f.known {
			featureState, featureStateSet := os.LookupEnv(fmt.Sprintf("KUBE_FEATURE_%s", feature))
			if !featureStateSet {
				continue
			}
			boolVal, boolErr := strconv.ParseBool(featureState)
			switch {
			case boolErr != nil:
				utilruntime.HandleError(fmt.Errorf("cannot set feature gate %q to %q, due to %v", feature, featureState, boolErr))
			case featureSpec.LockToDefault:
				if boolVal != featureSpec.Default {
					utilruntime.HandleError(fmt.Errorf("cannot set feature gate %q to %q, feature is locked to %v", feature, featureState, featureSpec.Default))
					break
				}
				featureGatesState[feature] = featureSpec.Default
			default:
				featureGatesState[feature] = boolVal
			}
		}
		f.enabled.Store(featureGatesState)
		f.readEnvVars.Store(true)

		for feature, featureSpec := range f.known {
			if featureState, ok := featureGatesState[feature]; ok {
				klog.V(1).InfoS("Feature gate updated state", "feature", feature, "enabled", featureState)
				continue
			}
			klog.V(1).InfoS("Feature gate default state", "feature", feature, "enabled", featureSpec.Default)
		}
	})
	return f.enabled.Load().(map[Feature]bool)
}

func (f *envVarFeatureGates) hasAlreadyReadEnvVar() bool {
	return f.readEnvVars.Load()
}
