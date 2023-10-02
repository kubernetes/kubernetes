package featuregate

import (
	"context"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"

	"k8s.io/apimachinery/pkg/util/naming"
	featuremetrics "k8s.io/component-base/metrics/prometheus/feature"
)

// envVarFeatureGate implements FeatureGate as well as pflag.Value for flag parsing.
type envVarFeatureGate struct {
	envVarFeatureGateName string

	special map[Feature]func(map[Feature]FeatureSpec, map[Feature]bool, bool)

	checkEnvVarsOnce sync.Once

	// lock guards writes to known, enabled, and reads/writes of closed
	lock sync.Mutex
	// known holds a map[Feature]FeatureSpec
	known *atomic.Value
	// enabled holds a map[Feature]bool
	enabled *atomic.Value
	// closed is set to true when AddFlag is called, and prevents subsequent calls to Add
	closed bool
}

func NewEnvVarFeatureGate() *envVarFeatureGate {
	known := map[Feature]FeatureSpec{}
	for k, v := range defaultFeatures {
		known[k] = v
	}

	knownValue := &atomic.Value{}
	knownValue.Store(known)

	enabled := map[Feature]bool{}
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	f := &envVarFeatureGate{
		envVarFeatureGateName: naming.GetNameFromCallsite(internalPackages...),
		known:                 knownValue,
		special:               specialFeatures,
		enabled:               enabledValue,
	}
	return f
}

// String returns a string containing all enabled feature gates, formatted as "key1=value1,key2=value2,...".
func (f *envVarFeatureGate) String() string {
	pairs := []string{}
	for k, v := range f.getEnabledMap() {
		pairs = append(pairs, fmt.Sprintf("%s=%t", k, v))
	}
	sort.Strings(pairs)
	return strings.Join(pairs, ",")
}

func (f *envVarFeatureGate) Type() string {
	return "mapStringBool"
}

// Add adds features to the envVarFeatureGate.
func (f *envVarFeatureGate) Add(features map[Feature]FeatureSpec) error {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		return fmt.Errorf("cannot add a feature gate after adding it to the flag set")
	}

	// Copy existing state
	known := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
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

	// Persist updated state
	f.known.Store(known)

	return nil
}

// GetAll returns a copy of the map of known feature names to feature specs.
func (f *envVarFeatureGate) GetAll() map[Feature]FeatureSpec {
	retval := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		retval[k] = v
	}
	return retval
}

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGate) getEnabledMap() map[Feature]bool {
	f.checkEnvVarsOnce.Do(func() {
		featureGateState := map[Feature]bool{}
		features := f.GetAll()
		for feature, featureSpec := range features {
			featureState := os.Getenv(fmt.Sprintf("KUBE_FEATURE_%s", feature))
			boolVal, boolErr := strconv.ParseBool(featureState)

			switch {
			case len(featureState) == 0:
				featureGateState[feature] = featureSpec.Default

			case boolErr != nil:
				utilruntime.HandleError(fmt.Errorf("cannot set feature gate %v to %v, due to %v", feature, featureState, boolErr))

			case featureSpec.LockToDefault:
				if boolVal != featureSpec.Default {
					utilruntime.HandleError(fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", feature, featureState, featureSpec.Default))
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

// Enabled returns true if the key is enabled.  If the key is not known, this call will panic.
func (f *envVarFeatureGate) Enabled(key Feature) bool {
	if v, ok := f.getEnabledMap()[key]; ok {
		return v
	}
	if v, ok := f.known.Load().(map[Feature]FeatureSpec)[key]; ok {
		return v.Default
	}

	panic(fmt.Errorf("feature %q is not registered in FeatureGate %q", key, f.envVarFeatureGateName))
}

func (f *envVarFeatureGate) AddMetrics() {
	for feature, featureSpec := range f.GetAll() {
		featuremetrics.RecordFeatureInfo(context.Background(), string(feature), string(featureSpec.PreRelease), f.Enabled(feature))
	}
}

// KnownFeatures returns a slice of strings describing the FeatureGate's known features.
// Deprecated and GA features are hidden from the list.
func (f *envVarFeatureGate) KnownFeatures() []string {
	var known []string
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		if v.PreRelease == GA || v.PreRelease == Deprecated {
			continue
		}
		known = append(known, fmt.Sprintf("%s=true|false (%s - default=%t)", k, v.PreRelease, v.Default))
	}
	sort.Strings(known)
	return known
}

// DeepCopy returns a deep copy of the FeatureGate object, such that gates can be
// set on the copy without mutating the original. This is useful for validating
// config against potential feature gate changes before committing those changes.
func (f *envVarFeatureGate) DeepCopy() FeatureGate {
	// Copy existing state.
	known := map[Feature]FeatureSpec{}
	for k, v := range f.known.Load().(map[Feature]FeatureSpec) {
		known[k] = v
	}
	enabled := map[Feature]bool{}
	for k, v := range f.enabled.Load().(map[Feature]bool) {
		enabled[k] = v
	}

	// Store copied state in new atomics.
	knownValue := &atomic.Value{}
	knownValue.Store(known)
	enabledValue := &atomic.Value{}
	enabledValue.Store(enabled)

	// Construct a new envVarFeatureGate around the copied state.
	// Note that specialFeatures is treated as immutable by convention,
	// and we maintain the value of f.closed across the copy.
	return &envVarFeatureGate{
		special: specialFeatures,
		known:   knownValue,
		enabled: enabledValue,
		closed:  f.closed,
	}
}
