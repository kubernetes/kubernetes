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
	"testing"

	"k8s.io/apiserver/pkg/util/feature"
)

// SetFeatureGateDuringTest sets the specified gate to the specified value, and returns a function that restores the original value.
// Failures to set or restore cause the test to fail.
//
// Example use:
//
// defer utilfeaturetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.<FeatureName>, true)()
func SetFeatureGateDuringTest(tb testing.TB, gate feature.FeatureGate, f feature.Feature, value bool) func() {
	originalValue := gate.Enabled(f)

	if err := gate.(feature.MutableFeatureGate).Set(fmt.Sprintf("%s=%v", f, value)); err != nil {
		tb.Errorf("error setting %s=%v: %v", f, value, err)
	}

	return func() {
		if err := gate.(feature.MutableFeatureGate).Set(fmt.Sprintf("%s=%v", f, originalValue)); err != nil {
			tb.Errorf("error restoring %s=%v: %v", f, originalValue, err)
		}
	}
}
