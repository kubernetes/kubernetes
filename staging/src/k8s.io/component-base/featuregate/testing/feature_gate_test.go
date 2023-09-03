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

	cleanupAlpha := SetFeatureGateDuringTest(t, gate, "AllAlpha", true)
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

	cleanupBeta := SetFeatureGateDuringTest(t, gate, "AllBeta", true)
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

	// run cleanups in reverse order like defer would
	cleanupBeta()
	cleanupAlpha()
	expect(t, gate, before)
}

func expect(t *gotest.T, gate featuregate.FeatureGate, expect map[featuregate.Feature]bool) {
	t.Helper()
	for k, v := range expect {
		if gate.Enabled(k) != v {
			t.Errorf("Expected %v=%v, got %v", k, v, gate.Enabled(k))
		}
	}
}
