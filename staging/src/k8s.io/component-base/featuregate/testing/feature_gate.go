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
	overrideLock                  sync.Mutex
	featureFlagOverride           map[featuregate.Feature]string
	emulationVersionOverride      string
	emulationVersionOverrideValue *version.Version
)

func init() {
	featureFlagOverride = map[featuregate.Feature]string{}
}

// SetFeatureGateDuringTest sets the specified gate to the specified value for duration of the test.
// Fails when it detects second call to the same flag or is unable to set or restore feature flag.
//
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set the same feature flag will cause fatal.
//
// Example use:
//
// featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.<FeatureName>, true)
func SetFeatureGateDuringTest(tb TB, gate featuregate.FeatureGate, f featuregate.Feature, value bool) {
	tb.Helper()
	detectParallelOverrideCleanup := detectParallelOverride(tb, f)
	originalValue := gate.Enabled(f)
	originalEmuVer := gate.(featuregate.MutableVersionedFeatureGate).EmulationVersion()
	originalExplicitlySet := gate.(featuregate.MutableVersionedFeatureGate).ExplicitlySet(f)

	// Specially handle AllAlpha and AllBeta
	if f == "AllAlpha" || f == "AllBeta" {
		// Iterate over individual gates so their individual values get restored
		for k, v := range gate.(featuregate.MutableFeatureGate).GetAll() {
			if k == "AllAlpha" || k == "AllBeta" {
				continue
			}
			if (f == "AllAlpha" && v.PreRelease == featuregate.Alpha) || (f == "AllBeta" && v.PreRelease == featuregate.Beta) {
				SetFeatureGateDuringTest(tb, gate, k, value)
			}
		}
	}

	if err := gate.(featuregate.MutableFeatureGate).Set(fmt.Sprintf("%s=%v", f, value)); err != nil {
		if s := suggestChangeEmulationVersion(tb, gate, f, value); s != "" {
			tb.Errorf("error setting %s=%v: %v. %s", f, value, err, s)
		} else {
			tb.Errorf("error setting %s=%v: %v", f, value, err)
		}
	}

	tb.Cleanup(func() {
		tb.Helper()
		detectParallelOverrideCleanup()
		emuVer := gate.(featuregate.MutableVersionedFeatureGate).EmulationVersion()
		if !emuVer.EqualTo(originalEmuVer) {
			tb.Fatalf("change of feature gate emulation version from %s to %s in the chain of SetFeatureGateDuringTest is not allowed\nuse SetFeatureGateEmulationVersionDuringTest to change emulation version in tests",
				originalEmuVer.String(), emuVer.String())
		}
		if originalExplicitlySet {
			if err := gate.(featuregate.MutableFeatureGate).Set(fmt.Sprintf("%s=%v", f, originalValue)); err != nil {
				tb.Errorf("error restoring %s=%v: %v", f, originalValue, err)
			}
		} else {
			if err := gate.(featuregate.MutableVersionedFeatureGate).ResetFeatureValueToDefault(f); err != nil {
				tb.Errorf("error restoring %s=%v: %v", f, originalValue, err)
			}
		}
	})
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

// SetFeatureGateEmulationVersionDuringTest sets the specified gate to the specified emulation version for duration of the test.
// Fails when it detects second call to set a different emulation version or is unable to set or restore emulation version.
// WARNING: Can leak set variable when called in test calling t.Parallel(), however second attempt to set a different emulation version will cause fatal.
// Example use:

// featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.31"))
func SetFeatureGateEmulationVersionDuringTest(tb TB, gate featuregate.FeatureGate, ver *version.Version) {
	tb.Helper()
	detectParallelOverrideCleanup := detectParallelOverrideEmulationVersion(tb, ver)
	originalEmuVer := gate.(featuregate.MutableVersionedFeatureGate).EmulationVersion()
	if err := gate.(featuregate.MutableVersionedFeatureGate).SetEmulationVersion(ver); err != nil {
		tb.Fatalf("failed to set emulation version to %s during test: %v", ver.String(), err)
	}
	tb.Cleanup(func() {
		tb.Helper()
		detectParallelOverrideCleanup()
		if err := gate.(featuregate.MutableVersionedFeatureGate).SetEmulationVersion(originalEmuVer); err != nil {
			tb.Fatalf("failed to restore emulation version to %s during test", originalEmuVer.String())
		}
	})
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

func detectParallelOverrideEmulationVersion(tb TB, ver *version.Version) func() {
	tb.Helper()
	overrideLock.Lock()
	defer overrideLock.Unlock()
	beforeOverrideTestName := emulationVersionOverride
	beforeOverrideValue := emulationVersionOverrideValue
	if ver.EqualTo(beforeOverrideValue) {
		return func() {}
	}
	if beforeOverrideTestName != "" && !sameTestOrSubtest(tb, beforeOverrideTestName) {
		tb.Fatalf("Detected parallel setting of a feature gate emulation version by both %q and %q", beforeOverrideTestName, tb.Name())
	}
	emulationVersionOverride = tb.Name()
	emulationVersionOverrideValue = ver

	return func() {
		tb.Helper()
		overrideLock.Lock()
		defer overrideLock.Unlock()
		if afterOverrideTestName := emulationVersionOverride; afterOverrideTestName != tb.Name() {
			tb.Fatalf("Detected parallel setting of a feature gate emulation version between both %q and %q", afterOverrideTestName, tb.Name())
		}
		emulationVersionOverride = beforeOverrideTestName
		emulationVersionOverrideValue = beforeOverrideValue
	}
}

func sameTestOrSubtest(tb TB, testName string) bool {
	// Assumes that "/" is not used in test names.
	return tb.Name() == testName || strings.HasPrefix(tb.Name(), testName+"/")
}

type TB interface {
	Cleanup(func())
	Error(args ...any)
	Errorf(format string, args ...any)
	Fatal(args ...any)
	Fatalf(format string, args ...any)
	Helper()
	Name() string
}
