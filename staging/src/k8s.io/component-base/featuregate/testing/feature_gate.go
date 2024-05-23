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
	"testing"

	"k8s.io/component-base/featuregate"
)

var (
	overrideLock        sync.Mutex
	featureFlagOverride map[featuregate.Feature]string
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
func SetFeatureGateDuringTest(tb testing.TB, gate featuregate.FeatureGate, f featuregate.Feature, value bool) {
	tb.Helper()
	detectParallelOverrideCleanup := detectParallelOverride(tb, f)
	originalEnabled := gate.(featuregate.MutableVersionedFeatureGateForTests).EnabledRawMap()

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
		tb.Errorf("error setting %s=%v: %v", f, value, err)
	}

	tb.Cleanup(func() {
		tb.Helper()
		detectParallelOverrideCleanup()
		gate.(featuregate.MutableVersionedFeatureGateForTests).Reset(originalEnabled)
	})
}

func detectParallelOverride(tb testing.TB, f featuregate.Feature) func() {
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

func sameTestOrSubtest(tb testing.TB, testName string) bool {
	// Assumes that "/" is not used in test names.
	return tb.Name() == testName || strings.HasPrefix(tb.Name(), testName+"/")
}
