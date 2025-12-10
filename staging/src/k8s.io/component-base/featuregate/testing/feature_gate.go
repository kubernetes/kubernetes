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

package testing

import (
	"fmt"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)

var (
	overrideLock          sync.Mutex
	featureFlagOverride   map[featuregate.Feature]string
	versionsOverride      string
	versionsOverrideValue string
)

func init() {
	featureFlagOverride = map[featuregate.Feature]string{}
}

type FeatureOverrides = map[featuregate.Feature]bool

// SetFeatureGateDuringTest sets the specified gate to the specified value for duration of the test.
// Fails when it detects second call to the same flag or is unable to set or restore feature flag.
// When disabling a feature, this automatically disables all dependents.
//
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set the same feature flag will cause fatal.
//
// Example use:
//
// featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.<FeatureName>, true)
func SetFeatureGateDuringTest(tb TB, gate featuregate.FeatureGate, f featuregate.Feature, value bool) {
	tb.Helper()
	SetFeatureGatesDuringTest(tb, gate, FeatureOverrides{f: value})
}

// SetFeatureGatesDuringTest sets the specified map of feature gate values for duration of the test.
// Fails when it detects second call to the same flag or is unable to set or restore feature flag.
// When disabling a feature, this automatically disables all dependents that weren't explicitly set.
//
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set the same feature flag will cause fatal.
//
// Example use:
//
//	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
//		features.<FeatureName>: true,
//	    features.<FeatureName>: false,
//	})
func SetFeatureGatesDuringTest(tb TB, gate featuregate.FeatureGate, features FeatureOverrides) {
	tb.Helper()
	originalEmuVer := gate.(featuregate.MutableVersionedFeatureGate).EmulationVersion()
	originalValues := map[string]bool{}
	var originalUnset []featuregate.Feature
	overrides := FeatureOverrides{}

	// Specially handle AllAlpha and AllBeta
	allAlphaValue, allAlpha := features["AllAlpha"]
	allBetaValue, allBeta := features["AllBeta"]
	if allAlpha || allBeta {
		// Iterate over individual gates so their individual values get restored
		for k, v := range gate.(featuregate.MutableFeatureGate).GetAll() {
			if (allAlpha && v.PreRelease == featuregate.Alpha) || (allBeta && v.PreRelease == featuregate.Beta) {
				// Setting AllAlpha or AllBeta on their own only sets unset features, but for
				// testing we want to override ALL alpha/beta features. So we explicitly set each
				// alpha/beta feature in addition to AllAlpha/AllBeta
				if v.PreRelease == featuregate.Alpha {
					overrides[k] = allAlphaValue
				} else {
					overrides[k] = allBetaValue
				}
			}
		}
	}

	// Explicit features take precedence, so merge them in now.
	for f, v := range features {
		overrides[f] = v
	}

	// Automatically disable dependents when disabling a dependency.
	dependencies := gate.Dependencies()
	for f := range dependencies {
		if _, overridden := overrides[f]; overridden || !gate.Enabled(f) {
			continue // Don't automatically disable features that have been explicitly set.
		}
		// If the feature gate was default-enabled and has an explicitly
		// disabled dependency, then automatically disable it.
		if disabled, disabledDep := hasDisabledDependency(f, dependencies, features); disabled {
			tb.Logf("Disabling feature %s since it depends on disabled feature %s", f, disabledDep)
			overrides[f] = false
		}
	}

	for f := range overrides {
		originalValues[string(f)] = gate.Enabled(f)
		if !gate.(featuregate.MutableVersionedFeatureGate).ExplicitlySet(f) {
			originalUnset = append(originalUnset, f)
		}
		tb.Cleanup(detectParallelOverride(tb, featuregate.Feature(f)))
	}

	m := map[string]bool{}
	for f, v := range overrides {
		m[string(f)] = v
	}
	if err := gate.(featuregate.MutableFeatureGate).SetFromMap(m); err != nil {
		tb.Errorf("Failed to set feature gates: %v", err)
		for f, v := range features {
			if s := suggestChangeEmulationVersion(tb, gate, f, v); s != "" {
				tb.Errorf("error setting %s=%v: %v. %s", f, v, err, s)
			}
		}
	}

	tb.Cleanup(func() {
		tb.Helper()
		emuVer := gate.(featuregate.MutableVersionedFeatureGate).EmulationVersion()
		if !emuVer.EqualTo(originalEmuVer) {
			tb.Fatalf("change of feature gate emulation version from %s to %s in the chain of SetFeatureGateDuringTest is not allowed\nuse SetFeatureGateEmulationVersionDuringTest to change emulation version in tests",
				originalEmuVer.String(), emuVer.String())
		}
		// To avoid violating feature dependencies, first atomicaly restore all original values,
		// then reset features that were unset (the value should be unchanged).
		if err := gate.(featuregate.MutableVersionedFeatureGate).SetFromMap(originalValues); err != nil {
			tb.Errorf("error restoring features %v: %v", originalValues, err)
		}
		for _, f := range originalUnset {
			if err := gate.(featuregate.MutableVersionedFeatureGate).ResetFeatureValueToDefault(f); err != nil {
				tb.Errorf("error resetting %s: %v", f, err)
			}
		}
	})
}

// hasDisabledDependency recursively walks the dependencies for feature f, and checks whether any are explicitly disabled in the features map.
func hasDisabledDependency(f featuregate.Feature, dependencies map[featuregate.Feature][]featuregate.Feature, features map[featuregate.Feature]bool) (bool, featuregate.Feature) {
	if enabled, set := features[f]; set {
		return !enabled, f
	}
	for _, dep := range dependencies[f] {
		if disabled, disabledDep := hasDisabledDependency(dep, dependencies, features); disabled {
			return disabled, disabledDep
		}
	}
	return false, ""
}

func suggestChangeEmulationVersion(tb TB, gate featuregate.FeatureGate, f featuregate.Feature, value bool) string {
	mutableVersionedFeatureGate, ok := gate.(featuregate.MutableVersionedFeatureGate)
	if !ok {
		return ""
	}

	emuVer := mutableVersionedFeatureGate.EmulationVersion()
	versionedSpecs, ok := mutableVersionedFeatureGate.GetAllVersioned()[f]
	if !ok {
		return ""
	}
	if len(versionedSpecs) > 1 {
		// check if the feature is locked
		lastLifecycle := versionedSpecs[len(versionedSpecs)-1]
		if lastLifecycle.LockToDefault && !lastLifecycle.Version.GreaterThan(emuVer) && lastLifecycle.Default != value {
			// if the feature is locked, set the emulation version to the previous version when the feature is not locked.
			return fmt.Sprintf("Feature %s is locked at version %s. Try adding SetFeatureGateEmulationVersionDuringTest(t, gate, version.MustParse(\"1.%d\")) at the beginning of your test.", f, emuVer.String(), lastLifecycle.Version.SubtractMinor(1).Minor())
		}
	}
	return ""
}

// SetFeatureGateVersionsDuringTest sets the specified gate to the specified emulation version and min compatibility version for duration of the test.
// Fails when it detects second call to set a different emulation version or min compatibility version, or is unable to set or restore emulation version and min compatibility version.
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set a different emulation version or min compatibility version will cause fatal.
// Example use:

// featuregatetesting.SetFeatureGateVersionsDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.31"), version.MustParse("1.31"))
func SetFeatureGateVersionsDuringTest(tb TB, gate featuregate.FeatureGate, emuVer, minCompatVer *version.Version) {
	tb.Helper()
	versions := fmt.Sprintf("emu=%s,min=%s", emuVer.String(), minCompatVer.String())
	detectParallelOverrideCleanup := detectParallelOverrideVersions(tb, versions)

	mutableGate := gate.(featuregate.MutableVersionedFeatureGate)
	originalEmuVer := mutableGate.EmulationVersion()
	originalMinCompatVer := mutableGate.MinCompatibilityVersion()

	if err := mutableGate.SetEmulationVersionAndMinCompatibilityVersion(emuVer, minCompatVer); err != nil {
		tb.Fatalf("failed to set versions (emu=%s, min=%s) during test: %v", emuVer.String(), minCompatVer.String(), err)
	}

	tb.Cleanup(func() {
		tb.Helper()
		detectParallelOverrideCleanup()
		if err := mutableGate.SetEmulationVersionAndMinCompatibilityVersion(originalEmuVer, originalMinCompatVer); err != nil {
			tb.Fatalf("failed to restore versions (emu=%s, min=%s) during test: %v", originalEmuVer.String(), originalMinCompatVer.String(), err)
		}
	})
}

// SetFeatureGateEmulationVersionDuringTest sets the specified gate to the specified emulation version for duration of the test.
// Fails when it detects second call to set a different emulation version or is unable to set or restore emulation version.
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set a different emulation version will cause fatal.
// Example use:

// featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.31"))
func SetFeatureGateEmulationVersionDuringTest(tb TB, gate featuregate.FeatureGate, ver *version.Version) {
	tb.Helper()
	SetFeatureGateVersionsDuringTest(tb, gate, ver, ver.SubtractMinor(1))
}

func detectParallelOverride(tb TB, f featuregate.Feature) func() {
	tb.Helper()
	overrideLock.Lock()
	defer overrideLock.Unlock()
	beforeOverrideTestName := featureFlagOverride[f]
	if beforeOverrideTestName != "" && !sameTestOrSubtest(tb, beforeOverrideTestName) {
		tb.Fatalf("Detected parallel setting of a feature gate by both %q and %q", beforeOverrideTestName, tb.Name())
	}
	featureFlagOverride[f] = tb.Name()

	return func() {
		tb.Helper()
		overrideLock.Lock()
		defer overrideLock.Unlock()
		if afterOverrideTestName := featureFlagOverride[f]; afterOverrideTestName != tb.Name() {
			tb.Fatalf("Detected parallel setting of a feature gate between both %q and %q", afterOverrideTestName, tb.Name())
		}
		featureFlagOverride[f] = beforeOverrideTestName
	}
}

func detectParallelOverrideVersions(tb TB, vers string) func() {
	tb.Helper()
	overrideLock.Lock()
	defer overrideLock.Unlock()
	beforeOverrideTestName := versionsOverride
	beforeOverrideValue := versionsOverrideValue
	if vers == beforeOverrideValue {
		return func() {}
	}
	if beforeOverrideTestName != "" && !sameTestOrSubtest(tb, beforeOverrideTestName) {
		tb.Fatalf("Detected parallel setting of feature gate versions by both %q and %q", beforeOverrideTestName, tb.Name())
	}
	versionsOverride = tb.Name()
	versionsOverrideValue = vers

	return func() {
		tb.Helper()
		overrideLock.Lock()
		defer overrideLock.Unlock()
		if afterOverrideTestName := versionsOverride; afterOverrideTestName != tb.Name() {
			tb.Fatalf("Detected parallel setting of feature gate versions between both %q and %q", afterOverrideTestName, tb.Name())
		}
		versionsOverride = beforeOverrideTestName
		versionsOverrideValue = beforeOverrideValue
	}
}

func sameTestOrSubtest(tb TB, testName string) bool {
	// Assumes that "/" is not used in test names.
	return tb.Name() == testName || strings.HasPrefix(tb.Name(), testName+"/")
}

type TB interface {
	Cleanup(func())
	Logf(format string, args ...any)
	Error(args ...any)
	Errorf(format string, args ...any)
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Name() string
}
