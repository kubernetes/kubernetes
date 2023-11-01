/*
Copyright 2016 The Kubernetes Authors.

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

package featuregate

var (
	// DefaultMutable is a mutable version of Default.
	// Only top-level commands/options setup and the k8s.io/component-base/featuregate/testing package should make use of this.
	// Tests that need to modify feature gates for the duration of their test should use:
	//   defer featuregatetesting.SetFeatureGateDuringTest(t, featuregate.Default, features.<FeatureName>, <value>)()
	DefaultMutable MutableFeatureGate = NewFeatureGate()

	// Default is a shared global FeatureGate.
	// Top-level commands/options setup that needs to modify this feature gate should use DefaultMutable.
	Default FeatureGate = DefaultMutable
)

// Enabled returns true if the key is enabled in the default FeatureGate.
func Enabled(key Feature) bool {
	return Default.Enabled(key)
}

// KnownFeatures returns a slice of strings describing the default FeatureGate's known features.
func KnownFeatures() []string {
	return Default.KnownFeatures()
}
