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
	"k8s.io/client-go/features"
)

func expect(t *gotest.T, gate features.Gates, expect map[features.Feature]bool) {
	t.Helper()
	for k, v := range expect {
		if gate.Enabled(k) != v {
			t.Errorf("Expected %v=%v, got %v", k, v, gate.Enabled(k))
		}
	}
}

func TestSetFeatureGateInTest(t *gotest.T) {
	gate := features.NewEnvVarFeatureGates(map[features.Feature]features.FeatureSpec{
		"feature": {PreRelease: features.Alpha, Default: false},
	})

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

func TestDetectLeakToMainTest(t *gotest.T) {
	t.Cleanup(func() {
		featureFlagOverride = map[features.Feature]string{}
	})
	gate := features.NewEnvVarFeatureGates(map[features.Feature]features.FeatureSpec{
		"feature": {PreRelease: features.Alpha, Default: false},
	})

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
	t.Cleanup(func() {
		featureFlagOverride = map[features.Feature]string{}
	})
	gate := features.NewEnvVarFeatureGates(map[features.Feature]features.FeatureSpec{
		"feature": {PreRelease: features.Alpha, Default: false},
	})

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
	t.Cleanup(func() {
		featureFlagOverride = map[features.Feature]string{}
	})
	gate := features.NewEnvVarFeatureGates(map[features.Feature]features.FeatureSpec{
		"feature": {PreRelease: features.Alpha, Default: false},
	})

	SetFeatureGateDuringTest(t, gate, "feature", false)
	// Subtest setting feature gate and calling parallel will leak it out
	t.Run("Subtest", func(t *gotest.T) {
		SetFeatureGateDuringTest(t, gate, "feature", true)
		t.Parallel()
	})
	// Leaked true
	assert.True(t, gate.Enabled("feature"))
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
