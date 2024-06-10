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

package testing

import (
	gotest "testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"
)

func TestSpecialGates(t *gotest.T) {
	gate := featuregate.NewFeatureGate()
	gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"alpha_default_on":         {PreRelease: featuregate.Alpha, Default: true},
		"alpha_default_on_set_off": {PreRelease: featuregate.Alpha, Default: true},
		"alpha_default_off":        {PreRelease: featuregate.Alpha, Default: false},
		"alpha_default_off_set_on": {PreRelease: featuregate.Alpha, Default: false},

		"beta_default_on":         {PreRelease: featuregate.Beta, Default: true},
		"beta_default_on_set_off": {PreRelease: featuregate.Beta, Default: true},
		"beta_default_off":        {PreRelease: featuregate.Beta, Default: false},
		"beta_default_off_set_on": {PreRelease: featuregate.Beta, Default: false},

		"stable_default_on":         {PreRelease: featuregate.GA, Default: true},
		"stable_default_on_set_off": {PreRelease: featuregate.GA, Default: true},
		"stable_default_off":        {PreRelease: featuregate.GA, Default: false},
		"stable_default_off_set_on": {PreRelease: featuregate.GA, Default: false},
	})
	gate.Set("alpha_default_on_set_off=false")
	gate.Set("beta_default_on_set_off=false")
	gate.Set("stable_default_on_set_off=false")
	gate.Set("alpha_default_off_set_on=true")
	gate.Set("beta_default_off_set_on=true")
	gate.Set("stable_default_off_set_on=true")

	before := map[featuregate.Feature]bool{
		"AllAlpha": false,
		"AllBeta":  false,

		"alpha_default_on":         true,
		"alpha_default_on_set_off": false,
		"alpha_default_off":        false,
		"alpha_default_off_set_on": true,

		"beta_default_on":         true,
		"beta_default_on_set_off": false,
		"beta_default_off":        false,
		"beta_default_off_set_on": true,

		"stable_default_on":         true,
		"stable_default_on_set_off": false,
		"stable_default_off":        false,
		"stable_default_off_set_on": true,
	}
	expect(t, gate, before)
	t.Cleanup(func() {
		expect(t, gate, before)
		cleanup()
	})

	SetFeatureGateDuringTest(t, gate, "AllAlpha", true)
	expect(t, gate, map[featuregate.Feature]bool{
		"AllAlpha": true,
		"AllBeta":  false,

		"alpha_default_on":         true,
		"alpha_default_on_set_off": true,
		"alpha_default_off":        true,
		"alpha_default_off_set_on": true,

		"beta_default_on":         true,
		"beta_default_on_set_off": false,
		"beta_default_off":        false,
		"beta_default_off_set_on": true,

		"stable_default_on":         true,
		"stable_default_on_set_off": false,
		"stable_default_off":        false,
		"stable_default_off_set_on": true,
	})

	SetFeatureGateDuringTest(t, gate, "AllBeta", true)
	expect(t, gate, map[featuregate.Feature]bool{
		"AllAlpha": true,
		"AllBeta":  true,

		"alpha_default_on":         true,
		"alpha_default_on_set_off": true,
		"alpha_default_off":        true,
		"alpha_default_off_set_on": true,

		"beta_default_on":         true,
		"beta_default_on_set_off": true,
		"beta_default_off":        true,
		"beta_default_off_set_on": true,

		"stable_default_on":         true,
		"stable_default_on_set_off": false,
		"stable_default_off":        false,
		"stable_default_off_set_on": true,
	})
}

func expect(t *gotest.T, gate featuregate.FeatureGate, expect map[featuregate.Feature]bool) {
	t.Helper()
	for k, v := range expect {
		if gate.Enabled(k) != v {
			t.Errorf("Expected %v=%v, got %v", k, v, gate.Enabled(k))
		}
	}
}

func TestSetFeatureGateInTest(t *gotest.T) {
	gate := featuregate.NewFeatureGate()
	err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"feature": {PreRelease: featuregate.Alpha, Default: false},
	})
	require.NoError(t, err)

	assert.False(t, gate.Enabled("feature"))
	SetFeatureGateDuringTest(t, gate, "feature", true)
	SetFeatureGateDuringTest(t, gate, "feature", true)

	assert.True(t, gate.Enabled("feature"))
	t.Run("Subtest", func(t *gotest.T) {
		assert.True(t, gate.Enabled("feature"))
	})

	t.Run("ParallelSubtest", func(t *gotest.T) {
		assert.True(t, gate.Enabled("feature"))
		// Calling t.Parallel in subtest will resume the main test body
		t.Parallel()
		assert.True(t, gate.Enabled("feature"))
	})
	assert.True(t, gate.Enabled("feature"))

	t.Run("OverwriteInSubtest", func(t *gotest.T) {
		SetFeatureGateDuringTest(t, gate, "feature", false)
		assert.False(t, gate.Enabled("feature"))
	})
	assert.True(t, gate.Enabled("feature"))
}

func TestSpecialGatesVersioned(t *gotest.T) {
	originalEmulationVersion := version.MustParse("1.31")
	gate := featuregate.NewVersionedFeatureGate(originalEmulationVersion)

	err := gate.AddVersioned(map[featuregate.Feature]featuregate.VersionedSpecs{
		"alpha_default_on": {
			{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Alpha},
		},
		"alpha_default_off": {
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		},
		"beta_default_on": {
			{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		},
		"beta_default_on_set_off": {
			{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		},
		"beta_default_off": {
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		},
		"beta_default_off_set_on": {
			{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
			{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		},
	})
	require.NoError(t, err)

	require.NoError(t, gate.Set("beta_default_on_set_off=false"))
	require.NoError(t, gate.Set("beta_default_off_set_on=true"))

	before := map[featuregate.Feature]bool{
		"AllAlpha": false,
		"AllBeta":  false,

		"alpha_default_on":  true,
		"alpha_default_off": false,

		"beta_default_on":         true,
		"beta_default_on_set_off": false,
		"beta_default_off":        false,
		"beta_default_off_set_on": true,
	}
	expect(t, gate, before)
	t.Cleanup(func() {
		expect(t, gate, before)
		cleanup()
	})

	t.Run("OverwriteInSubtest", func(t *gotest.T) {
		SetFeatureGateDuringTest(t, gate, "AllAlpha", true)
		expect(t, gate, map[featuregate.Feature]bool{
			"AllAlpha": true,
			"AllBeta":  false,

			"alpha_default_on":  true,
			"alpha_default_off": true,

			"beta_default_on":         true,
			"beta_default_on_set_off": false,
			"beta_default_off":        false,
			"beta_default_off_set_on": true,
		})

		SetFeatureGateDuringTest(t, gate, "AllBeta", true)
		expect(t, gate, map[featuregate.Feature]bool{
			"AllAlpha": true,
			"AllBeta":  true,

			"alpha_default_on":  true,
			"alpha_default_off": true,

			"beta_default_on":         true,
			"beta_default_on_set_off": true,
			"beta_default_off":        true,
			"beta_default_off_set_on": true,
		})
	})
	expect(t, gate, before)

	t.Run("OverwriteInSubtestAtDifferentVersion", func(t *gotest.T) {
		SetFeatureGateEmulationVersionDuringTest(t, gate, version.MustParse("1.28"))
		SetFeatureGateDuringTest(t, gate, "AllAlpha", true)
		expect(t, gate, map[featuregate.Feature]bool{
			"AllAlpha": true,
			"AllBeta":  false,

			"alpha_default_on":  true,
			"alpha_default_off": true,

			"beta_default_on":         false,
			"beta_default_on_set_off": true,
			"beta_default_off":        false,
			"beta_default_off_set_on": true,
		})
	})

}

func TestDetectLeakToMainTest(t *gotest.T) {
	t.Cleanup(cleanup)
	gate := featuregate.NewFeatureGate()
	err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"feature": {PreRelease: featuregate.Alpha, Default: false},
	})
	require.NoError(t, err)

	// Subtest setting feature gate and calling parallel will leak it out
	t.Run("LeakingSubtest", func(t *gotest.T) {
		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateDuringTest(fakeT, gate, "feature", true)
		// Calling t.Parallel in subtest will resume the main test body
		t.Parallel()
		// Leaked false from main test
		assert.False(t, gate.Enabled("feature"))
	})
	// Leaked true from subtest
	assert.True(t, gate.Enabled("feature"))
	fakeT := &ignoreFatalT{T: t}
	SetFeatureGateDuringTest(fakeT, gate, "feature", false)
	assert.True(t, fakeT.fatalRecorded)
}

func TestDetectLeakToOtherSubtest(t *gotest.T) {
	t.Cleanup(cleanup)
	gate := featuregate.NewFeatureGate()
	err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"feature": {PreRelease: featuregate.Alpha, Default: false},
	})
	require.NoError(t, err)

	subtestName := "Subtest"
	// Subtest setting feature gate and calling parallel will leak it out
	t.Run(subtestName, func(t *gotest.T) {
		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateDuringTest(fakeT, gate, "feature", true)
		t.Parallel()
	})
	// Add suffix to name to prevent tests with the same prefix.
	t.Run(subtestName+"Suffix", func(t *gotest.T) {
		// Leaked true
		assert.True(t, gate.Enabled("feature"))

		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateDuringTest(fakeT, gate, "feature", false)
		assert.True(t, fakeT.fatalRecorded)
	})
}

func TestCannotDetectLeakFromSubtest(t *gotest.T) {
	t.Cleanup(cleanup)
	gate := featuregate.NewFeatureGate()
	err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"feature": {PreRelease: featuregate.Alpha, Default: false},
	})
	require.NoError(t, err)

	SetFeatureGateDuringTest(t, gate, "feature", false)
	// Subtest setting feature gate and calling parallel will leak it out
	t.Run("Subtest", func(t *gotest.T) {
		SetFeatureGateDuringTest(t, gate, "feature", true)
		t.Parallel()
	})
	// Leaked true
	assert.True(t, gate.Enabled("feature"))
}

func TestCannotDetectLeakFromTwoSubtestsWithDifferentFeatures(t *gotest.T) {
	t.Cleanup(cleanup)
	gate := featuregate.NewFeatureGate()
	err := gate.Add(map[featuregate.Feature]featuregate.FeatureSpec{
		"feature1": {PreRelease: featuregate.Alpha, Default: false},
		"feature2": {PreRelease: featuregate.Alpha, Default: false},
	})
	require.NoError(t, err)

	assert.False(t, gate.Enabled("feature1"))
	assert.False(t, gate.Enabled("feature2"))
	subtestName := "Subtest"
	// Subtest setting feature gate and calling parallel will leak it out
	t.Run(subtestName, func(t *gotest.T) {
		SetFeatureGateDuringTest(t, gate, "feature1", true)
		t.Parallel()
		assert.True(t, gate.Enabled("feature1"))
		assert.False(t, gate.Enabled("feature2"))
	})
	// Leaked true
	assert.True(t, gate.Enabled("feature1"))
	assert.False(t, gate.Enabled("feature2"))
	// Add suffix to name to prevent tests with the same prefix.
	t.Run(subtestName+"Suffix", func(t *gotest.T) {
		// Leaked true
		assert.True(t, gate.Enabled("feature1"))
		assert.False(t, gate.Enabled("feature2"))
		SetFeatureGateDuringTest(t, gate, "feature2", true)
		assert.True(t, gate.Enabled("feature1"))
		assert.True(t, gate.Enabled("feature2"))
	})
}

func TestDetectEmulationVersionLeakToMainTest(t *gotest.T) {
	t.Cleanup(cleanup)
	originalEmulationVersion := version.MustParse("1.31")
	newEmulationVersion := version.MustParse("1.30")
	gate := featuregate.NewVersionedFeatureGate(originalEmulationVersion)
	assert.True(t, gate.EmulationVersion().EqualTo(originalEmulationVersion))

	// Subtest setting feature gate and calling parallel will leak it out
	t.Run("LeakingSubtest", func(t *gotest.T) {
		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateEmulationVersionDuringTest(fakeT, gate, newEmulationVersion)
		// Calling t.Parallel in subtest will resume the main test body
		t.Parallel()
		// Leaked from main test
		assert.True(t, gate.EmulationVersion().EqualTo(originalEmulationVersion))
	})
	// Leaked from subtest
	assert.True(t, gate.EmulationVersion().EqualTo(newEmulationVersion))
	fakeT := &ignoreFatalT{T: t}
	SetFeatureGateEmulationVersionDuringTest(fakeT, gate, originalEmulationVersion)
	assert.True(t, fakeT.fatalRecorded)
}

func TestNoLeakFromSameEmulationVersionToMainTest(t *gotest.T) {
	t.Cleanup(cleanup)
	originalEmulationVersion := version.MustParse("1.31")
	newEmulationVersion := version.MustParse("1.31")
	gate := featuregate.NewVersionedFeatureGate(originalEmulationVersion)
	assert.True(t, gate.EmulationVersion().EqualTo(originalEmulationVersion))

	// Subtest setting feature gate and calling parallel will leak it out
	t.Run("LeakingSubtest", func(t *gotest.T) {
		SetFeatureGateEmulationVersionDuringTest(t, gate, newEmulationVersion)
		// Calling t.Parallel in subtest will resume the main test body
		t.Parallel()
		// Leaked from main test
		assert.True(t, gate.EmulationVersion().EqualTo(originalEmulationVersion))
	})
	// Leaked from subtest
	assert.True(t, gate.EmulationVersion().EqualTo(newEmulationVersion))
	SetFeatureGateEmulationVersionDuringTest(t, gate, originalEmulationVersion)
}

func TestDetectEmulationVersionLeakToOtherSubtest(t *gotest.T) {
	t.Cleanup(cleanup)
	originalEmulationVersion := version.MustParse("1.31")
	newEmulationVersion := version.MustParse("1.30")
	gate := featuregate.NewVersionedFeatureGate(originalEmulationVersion)
	assert.True(t, gate.EmulationVersion().EqualTo(originalEmulationVersion))

	subtestName := "Subtest"
	// Subtest setting feature gate and calling parallel will leak it out
	t.Run(subtestName, func(t *gotest.T) {
		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateEmulationVersionDuringTest(fakeT, gate, newEmulationVersion)
		t.Parallel()
	})
	// Add suffix to name to prevent tests with the same prefix.
	t.Run(subtestName+"Suffix", func(t *gotest.T) {
		// Leaked newEmulationVersion
		assert.True(t, gate.EmulationVersion().EqualTo(newEmulationVersion))

		fakeT := &ignoreFatalT{T: t}
		SetFeatureGateEmulationVersionDuringTest(fakeT, gate, originalEmulationVersion)
		assert.True(t, fakeT.fatalRecorded)
	})
}

type ignoreFatalT struct {
	*gotest.T
	fatalRecorded bool
}

func (f *ignoreFatalT) Fatal(args ...any) {
	f.T.Helper()
	f.fatalRecorded = true
	newArgs := []any{"[IGNORED]"}
	newArgs = append(newArgs, args...)
	f.T.Log(newArgs...)
}

func (f *ignoreFatalT) Fatalf(format string, args ...any) {
	f.T.Helper()
	f.fatalRecorded = true
	f.T.Logf("[IGNORED] "+format, args...)
}

func cleanup() {
	featureFlagOverride = map[featuregate.Feature]string{}
	emulationVersionOverride = ""
	emulationVersionOverrideValue = nil
}
