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

package testing

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"k8s.io/client-go/features"
)

func TestDriveInitDefaultFeatureGates(t *testing.T) {
	restoreDefaultFeatureGatesOnComplete(t)
	featureGates := features.FeatureGates()
	assertFunctionPanicsWithMessage(t, func() { featureGates.Enabled("FakeFeatureGate") }, "features.FeatureGates().Enabled", fmt.Sprintf("feature %q is not registered in FeatureGate", "FakeFeatureGate"))

	fakeFeatureGates := &alwaysEnabledFakeGates{}
	require.True(t, fakeFeatureGates.Enabled("FakeFeatureGate"))

	features.ReplaceFeatureGates(fakeFeatureGates)
	featureGates = features.FeatureGates()

	require.IsType(t, (*alwaysEnabledFakeGates)(nil), featureGates)
	require.True(t, featureGates.Enabled("FakeFeatureGate"))
}

func TestSetFeatureGatesDuringTest(t *testing.T) {
	restoreDefaultFeatureGatesOnComplete(t)
	unsafeSkipCheckingKnownFeaturesForDurationOfTest(t)
	featureA := features.Feature("FeatureA")
	featureB := features.Feature("FeatureB")
	fakeGates := &fakeFeatureGates{map[features.Feature]bool{featureA: true, featureB: true}}
	features.ReplaceFeatureGates(fakeGates)
	t.Cleanup(func() {
		// since cleanup functions will be called in last added, first called order.
		// check if the original feature gates were restored
		require.True(t, features.FeatureGates().Enabled(featureA), "the original feature gates weren't restored")
		require.IsType(t, (*fakeFeatureGates)(nil), features.FeatureGates())
	})

	SetFeatureDuringTest(t, featureA, false)

	require.False(t, features.FeatureGates().Enabled(featureA))
	require.True(t, features.FeatureGates().Enabled(featureB))
}

func TestSetFeatureGatesDuringTestNegative(t *testing.T) {
	restoreDefaultFeatureGatesOnComplete(t)
	featureA := features.Feature("FeatureA")
	fakeGates := &fakeFeatureGates{map[features.Feature]bool{featureA: true}}
	fakeTesting := &fakeT{fakeTestName: "fakeTestNameOne", TB: t}

	overrideCleanup, err := overrideFeatureGatesLocked(fakeTesting, fakeGates)
	require.NoError(t, err)
	t.Cleanup(overrideCleanup)

	fakeTesting.fakeTestName = "fakeTestNameTwo"
	_, err = overrideFeatureGatesLocked(fakeTesting, fakeGates)
	require.Error(t, err)
}

type fakeFeatureGates struct {
	features map[features.Feature]bool
}

func (f *fakeFeatureGates) Enabled(feature features.Feature) bool {
	return f.features[feature]
}

type alwaysEnabledFakeGates struct{}

func (f *alwaysEnabledFakeGates) Enabled(features.Feature) bool {
	return true
}

type fakeT struct {
	fakeTestName string
	testing.TB
}

func (t *fakeT) Name() string {
	return t.fakeTestName
}

func assertFunctionPanicsWithMessage(t *testing.T, f func(), fName, errMessage string) {
	didPanic, panicMessage := didFunctionPanic(f)
	if !didPanic {
		t.Fatalf("function %q did not panicked", fName)
	}

	panicError, ok := panicMessage.(error)
	if !ok || !strings.Contains(panicError.Error(), errMessage) {
		t.Fatalf("func %q should panic with error message:\t%#v\n\tPanic value:\t%#v\n", fName, errMessage, panicMessage)
	}
}

func didFunctionPanic(f func()) (didPanic bool, panicMessage interface{}) {
	didPanic = true

	defer func() {
		panicMessage = recover()
	}()

	f()
	didPanic = false

	return
}

func restoreDefaultFeatureGatesOnComplete(t *testing.T) {
	t.Helper()
	originalGates := features.FeatureGates()
	t.Cleanup(func() {
		features.ReplaceFeatureGates(originalGates)
	})
}
