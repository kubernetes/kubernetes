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
		"feature2": {Default: true, PreRelease: utilfeature.Alpha},
		"feature1": {Default: false, PreRelease: utilfeature.Beta},
		"feature3": {Default: false, PreRelease: utilfeature.GA},
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
	// check the second value is feature3; prerelease should not shown fo GA features; default should be present
	f3 := "feature3=true|false (default=false)"
	if r[2] != f3 {
		t.Errorf("KnownFeatures returned %s values, expected %s", r[2], f3)
	}
}

func TestNewFeatureGate(t *testing.T) {
	var someFeatures = FeatureList{
		"feature1": {Default: false, PreRelease: utilfeature.Beta},
		"feature2": {Default: true, PreRelease: utilfeature.Alpha},
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
