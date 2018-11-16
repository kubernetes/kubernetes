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
	"os"
	"strings"
	"testing"

	"k8s.io/apiserver/pkg/util/feature"
)

// VerifyFeatureGatesUnchanged ensures the provided gate does not change any values when tests() are completed.
// Intended to be placed into unit test packages that mess with feature gates.
//
// Example use:
//
// import (
//   "testing"
//
//   utilfeature "k8s.io/apiserver/pkg/util/feature"
//   utilfeaturetesting "k8s.io/apiserver/pkg/util/feature/testing"
//   _ "k8s.io/kubernetes/pkg/features"
// )
//
// func TestMain(m *testing.M) {
//   utilfeaturetesting.VerifyFeatureGatesUnchanged(utilfeature.DefaultFeatureGate, m.Run)
// }
func VerifyFeatureGatesUnchanged(gate feature.FeatureGate, tests func() int) {
	originalGates := gate.DeepCopy()
	originalSet := fmt.Sprint(gate)

	rc := tests()

	finalSet := fmt.Sprint(gate)
	if finalSet != originalSet {
		for _, kv := range strings.Split(finalSet, ",") {
			k := strings.Split(kv, "=")[0]
			if originalGates.Enabled(feature.Feature(k)) != gate.Enabled(feature.Feature(k)) {
				fmt.Println(fmt.Sprintf("VerifyFeatureGatesUnchanged: mutated %s feature gate from %v to %v", k, originalGates.Enabled(feature.Feature(k)), gate.Enabled(feature.Feature(k))))
				rc = 1
			}
		}
	}

	if rc != 0 {
		os.Exit(rc)
	}
}

// SetFeatureGateDuringTest sets the specified gate to the specified value, and returns a function that restores the original value.
// Failures to set or restore cause the test to fail.
//
// Example use:
//
// defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.<FeatureName>, true)()
func SetFeatureGateDuringTest(t *testing.T, gate feature.FeatureGate, feature feature.Feature, value bool) func() {
	originalValue := gate.Enabled(feature)

	if err := gate.Set(fmt.Sprintf("%s=%v", feature, value)); err != nil {
		t.Errorf("error setting %s=%v: %v", feature, value, err)
	}

	return func() {
		if err := gate.Set(fmt.Sprintf("%s=%v", feature, originalValue)); err != nil {
			t.Errorf("error restoring %s=%v: %v", feature, originalValue, err)
		}
	}
}
