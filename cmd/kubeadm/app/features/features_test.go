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

package features

import (
	"reflect"
	"testing"

	"k8s.io/component-base/featuregate"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestKnownFeatures(t *testing.T) {
	var someFeatures = FeatureList{
		"feature2": {FeatureSpec: featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Alpha}},
		"feature1": {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}},
		"feature3": {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.GA}},
		"hidden":   {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.GA}, HiddenInHelpText: true},
	}

	r := KnownFeatures(&someFeatures)

	if len(r) != 3 {
		t.Errorf("KnownFeatures returned %d values, expected 3", len(r))
	}

	// check the first value is feature1 (the list should be sorted); prerelease and default should be present
	f1 := "feature1=true|false (BETA - default=false)"
	if r[0] != f1 {
		t.Errorf("KnownFeatures returned %s values, expected %s", r[0], f1)
	}
	// check the second value is feature2; prerelease and default should be present
	f2 := "feature2=true|false (ALPHA - default=true)"
	if r[1] != f2 {
		t.Errorf("KnownFeatures returned %s values, expected %s", r[1], f2)
	}
	// check the second value is feature3; prerelease should not be shown for GA features; default should be present
	f3 := "feature3=true|false (default=false)"
	if r[2] != f3 {
		t.Errorf("KnownFeatures returned %s values, expected %s", r[2], f3)
	}
}

func TestNewFeatureGate(t *testing.T) {
	var someFeatures = FeatureList{
		"feature1":   {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}},
		"feature2":   {FeatureSpec: featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Alpha}},
		"deprecated": {FeatureSpec: featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Deprecated}},
	}

	var tests = []struct {
		value                string
		expectedError        bool
		expectedFeaturesGate map[string]bool
	}{
		{ //invalid value (missing =)
			value:         "invalidValue",
			expectedError: true,
		},
		{ //invalid value (missing =)
			value:         "feature1=true,invalidValue",
			expectedError: true,
		},
		{ //invalid value (not a boolean)
			value:         "feature1=notABoolean",
			expectedError: true,
		},
		{ //invalid value (not a boolean)
			value:         "feature1=true,feature2=notABoolean",
			expectedError: true,
		},
		{ //unrecognized feature-gate key
			value:         "unknownFeature=false",
			expectedError: true,
		},
		{ //unrecognized feature-gate key
			value:         "feature1=true,unknownFeature=false",
			expectedError: true,
		},
		{ //deprecated feature-gate key
			value:         "deprecated=true",
			expectedError: true,
		},
		{ //one feature
			value:                "feature1=true",
			expectedError:        false,
			expectedFeaturesGate: map[string]bool{"feature1": true},
		},
		{ //two features
			value:                "feature1=true,feature2=false",
			expectedError:        false,
			expectedFeaturesGate: map[string]bool{"feature1": true, "feature2": false},
		},
	}

	for _, test := range tests {
		t.Run(test.value, func(t *testing.T) {
			r, err := NewFeatureGate(&someFeatures, test.value)

			if !test.expectedError && err != nil {
				t.Errorf("NewFeatureGate failed when not expected: %v", err)
				return
			} else if test.expectedError && err == nil {
				t.Error("NewFeatureGate didn't failed when expected")
				return
			}

			if !reflect.DeepEqual(r, test.expectedFeaturesGate) {
				t.Errorf("NewFeatureGate returned a unexpected value")
			}
		})
	}
}

func TestValidateVersion(t *testing.T) {
	var someFeatures = FeatureList{
		"feature1": {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}},
		"feature2": {FeatureSpec: featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Alpha}, MinimumVersion: constants.MinimumControlPlaneVersion.WithPreRelease("alpha.1")},
	}

	var tests = []struct {
		name              string
		requestedVersion  string
		requestedFeatures map[string]bool
		expectedError     bool
	}{
		{
			name:              "no min version",
			requestedFeatures: map[string]bool{"feature1": true},
			expectedError:     false,
		},
		{
			name:              "min version but correct value given",
			requestedFeatures: map[string]bool{"feature2": true},
			requestedVersion:  constants.MinimumControlPlaneVersion.String(),
			expectedError:     false,
		},
		{
			name:              "min version and incorrect value given",
			requestedFeatures: map[string]bool{"feature2": true},
			requestedVersion:  "v1.11.2",
			expectedError:     true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := ValidateVersion(someFeatures, test.requestedFeatures, test.requestedVersion)
			if !test.expectedError && err != nil {
				t.Errorf("ValidateVersion failed when not expected: %v", err)
				return
			} else if test.expectedError && err == nil {
				t.Error("ValidateVersion didn't failed when expected")
				return
			}
		})
	}
}

// TestEnabledDefaults tests that Enabled returns the default values for
// each feature gate when no feature gates are specified.
func TestEnabledDefaults(t *testing.T) {
	for featureName, feature := range InitFeatureGates {
		featureList := make(map[string]bool)

		enabled := Enabled(featureList, featureName)
		if enabled != feature.Default {
			t.Errorf("Enabled returned %v instead of default value %v for feature %s", enabled, feature.Default, featureName)
		}
	}
}

func TestCheckDeprecatedFlags(t *testing.T) {
	dummyMessage := "dummy message"
	var someFeatures = FeatureList{
		"feature1":   {FeatureSpec: featuregate.FeatureSpec{Default: false, PreRelease: featuregate.Beta}},
		"deprecated": {FeatureSpec: featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Deprecated}, DeprecationMessage: dummyMessage},
	}

	var tests = []struct {
		name        string
		features    map[string]bool
		expectedMsg map[string]string
	}{
		{
			name:        "deprecated feature",
			features:    map[string]bool{"deprecated": true},
			expectedMsg: map[string]string{"deprecated": dummyMessage},
		},
		{
			name:        "valid feature",
			features:    map[string]bool{"feature1": true},
			expectedMsg: map[string]string{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			msg := CheckDeprecatedFlags(&someFeatures, test.features)
			if !reflect.DeepEqual(test.expectedMsg, msg) {
				t.Error("CheckDeprecatedFlags didn't returned expected message")
			}
		})
	}
}
