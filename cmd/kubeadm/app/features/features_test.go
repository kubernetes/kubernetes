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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

func TestKnownFeatures(t *testing.T) {
	var someFeatures = FeatureList{
		"feature2": {FeatureSpec: utilfeature.FeatureSpec{Default: true, PreRelease: utilfeature.Alpha}},
		"feature1": {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Beta}},
		"feature3": {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.GA}},
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
		"feature1": {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Beta}},
		"feature2": {FeatureSpec: utilfeature.FeatureSpec{Default: true, PreRelease: utilfeature.Alpha}},
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

		r, err := NewFeatureGate(&someFeatures, test.value)

		if !test.expectedError && err != nil {
			t.Errorf("NewFeatureGate failed when not expected: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("NewFeatureGate didn't failed when expected")
			continue
		}

		if !reflect.DeepEqual(r, test.expectedFeaturesGate) {
			t.Errorf("NewFeatureGate returned a unexpected value")
		}
	}
}

func TestValidateVersion(t *testing.T) {
	var someFeatures = FeatureList{
		"feature1": {FeatureSpec: utilfeature.FeatureSpec{Default: false, PreRelease: utilfeature.Beta}},
		"feature2": {FeatureSpec: utilfeature.FeatureSpec{Default: true, PreRelease: utilfeature.Alpha}, MinimumVersion: v190},
	}

	var tests = []struct {
		requestedVersion  string
		requestedFeatures map[string]bool
		expectedError     bool
	}{
		{ //no min version
			requestedFeatures: map[string]bool{"feature1": true},
			expectedError:     false,
		},
		{ //min version but correct value given
			requestedFeatures: map[string]bool{"feature2": true},
			requestedVersion:  "v1.9.0",
			expectedError:     false,
		},
		{ //min version and incorrect value given
			requestedFeatures: map[string]bool{"feature2": true},
			requestedVersion:  "v1.8.2",
			expectedError:     true,
		},
	}

	for _, test := range tests {
		err := ValidateVersion(someFeatures, test.requestedFeatures, test.requestedVersion)
		if !test.expectedError && err != nil {
			t.Errorf("ValidateVersion failed when not expected: %v", err)
			continue
		} else if test.expectedError && err == nil {
			t.Error("ValidateVersion didn't failed when expected")
			continue
		}
	}
}

func TestResolveFeatureGateDependencies(t *testing.T) {

	var tests = []struct {
		inputFeatures    map[string]bool
		expectedFeatures map[string]bool
	}{
		{ // no flags
			inputFeatures:    map[string]bool{},
			expectedFeatures: map[string]bool{},
		},
		{ // others flags
			inputFeatures:    map[string]bool{CoreDNS: true},
			expectedFeatures: map[string]bool{CoreDNS: true},
		},
		{ // just StoreCertsInSecrets flags
			inputFeatures:    map[string]bool{StoreCertsInSecrets: true},
			expectedFeatures: map[string]bool{StoreCertsInSecrets: true, SelfHosting: true},
		},
		{ // just HighAvailability flags
			inputFeatures:    map[string]bool{HighAvailability: true},
			expectedFeatures: map[string]bool{HighAvailability: true, StoreCertsInSecrets: true, SelfHosting: true},
		},
	}

	for _, test := range tests {
		ResolveFeatureGateDependencies(test.inputFeatures)
		if !reflect.DeepEqual(test.inputFeatures, test.expectedFeatures) {
			t.Errorf("ResolveFeatureGateDependencies failed, expected: %v, got: %v", test.inputFeatures, test.expectedFeatures)

		}
	}
}
