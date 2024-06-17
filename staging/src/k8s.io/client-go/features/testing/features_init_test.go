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
	featureGates := features.FeatureGates()
	assertFunctionPanicsWithMessage(t, func() { featureGates.Enabled("FakeFeatureGate") }, "features.FeatureGates().Enabled", fmt.Sprintf("feature %q is not registered in FeatureGate", "FakeFeatureGate"))

	fakeGates := &fakeFeatureGates{features: map[features.Feature]bool{"FakeFeatureGate": true}}
	require.True(t, fakeGates.Enabled("FakeFeatureGate"))

	features.ReplaceFeatureGates(fakeGates)
	featureGates = features.FeatureGates()

	assertFeatureGatesType(t, featureGates)
	require.True(t, featureGates.Enabled("FakeFeatureGate"))
}

func TestSetFeatureGatesDuringTest(t *testing.T) {
	featureA := features.Feature("FeatureA")
	featureB := features.Feature("FeatureB")
	fakeGates := &fakeFeatureGates{map[features.Feature]bool{featureA: true, featureB: true}}
	features.ReplaceFeatureGates(fakeGates)
	t.Cleanup(func() {
		// since cleanup functions will be called in last added, first called order.
		// check if the original feature wasn't restored
		require.True(t, features.FeatureGates().Enabled(featureA), "the original feature = %v wasn't restored", featureA)
	})

	SetFeatureDuringTest(t, featureA, false)

	require.False(t, features.FeatureGates().Enabled(featureA))
	require.True(t, features.FeatureGates().Enabled(featureB))
}

func TestSetFeatureGatesDuringTestPanics(t *testing.T) {
	fakeGates := &fakeFeatureGates{features: map[features.Feature]bool{"FakeFeatureGate": true}}

	features.ReplaceFeatureGates(fakeGates)
	assertFunctionPanicsWithMessage(t, func() { SetFeatureDuringTest(t, "UnknownFeature", false) }, "SetFeatureDuringTest", fmt.Sprintf("feature %q is not registered in featureGates", "UnknownFeature"))

	readOnlyGates := &readOnlyAlwaysDisabledFeatureGates{}
	features.ReplaceFeatureGates(readOnlyGates)
	assertFunctionPanicsWithMessage(t, func() { SetFeatureDuringTest(t, "FakeFeature", false) }, "SetFeatureDuringTest", fmt.Sprintf("clientfeatures.FeatureGates(): %T does not implement featureGatesSetter interface", readOnlyGates))
}

func TestOverridesForSetFeatureGatesDuringTest(t *testing.T) {
	scenarios := []struct {
		name           string
		firstTestName  string
		secondTestName string
		expectError    bool
	}{
		{
			name:           "concurrent tests setting the same feature fail",
			firstTestName:  "fooTest",
			secondTestName: "barTest",
			expectError:    true,
		},

		{
			name:           "same test setting the same feature does not fail",
			firstTestName:  "fooTest",
			secondTestName: "fooTest",
			expectError:    false,
		},

		{
			name:           "subtests setting the same feature don't not fail",
			firstTestName:  "fooTest",
			secondTestName: "fooTest/scenario1",
			expectError:    false,
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			featureA := features.Feature("FeatureA")
			fakeGates := &fakeFeatureGates{map[features.Feature]bool{featureA: true}}
			fakeTesting := &fakeT{fakeTestName: scenario.firstTestName, TB: t}

			features.ReplaceFeatureGates(fakeGates)
			require.NoError(t, setFeatureDuringTestInternal(fakeTesting, featureA, true))
			require.True(t, features.FeatureGates().Enabled(featureA))

			fakeTesting.fakeTestName = scenario.secondTestName
			err := setFeatureDuringTestInternal(fakeTesting, featureA, false)
			require.Equal(t, scenario.expectError, err != nil)
		})
	}
}

type fakeFeatureGates struct {
	features map[features.Feature]bool
}

func (f *fakeFeatureGates) Enabled(feature features.Feature) bool {
	featureValue, ok := f.features[feature]
	if !ok {
		panic(fmt.Errorf("feature %q is not registered in featureGates", feature))
	}
	return featureValue
}

func (f *fakeFeatureGates) Set(feature features.Feature, value bool) error {
	f.features[feature] = value
	return nil
}

type readOnlyAlwaysDisabledFeatureGates struct{}

func (f *readOnlyAlwaysDisabledFeatureGates) Enabled(feature features.Feature) bool {
	return false
}

type fakeT struct {
	fakeTestName string
	testing.TB
}

func (t *fakeT) Name() string {
	return t.fakeTestName
}

func assertFeatureGatesType(t *testing.T, fg features.Gates) {
	_, ok := fg.(*fakeFeatureGates)
	if !ok {
		t.Fatalf("passed features.FeatureGates() is NOT of type *alwaysEnabledFakeGates, it is of type = %T", fg)
	}
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
