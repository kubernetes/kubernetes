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

package featuregate

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEnvVarFeatureGate(t *testing.T) {
	defaultTestFeatures := map[Feature]FeatureSpec{
		"TestAlpha": {
			Default:       false,
			LockToDefault: false,
			PreRelease:    "Alpha",
		},
		"TestBeta": {
			Default:       true,
			LockToDefault: false,
			PreRelease:    "Beta",
		},
	}
	expectedDefaultFeaturesState := map[Feature]bool{"TestAlpha": false, "TestBeta": true}

	copyExpectedStateMap := func(toCopy map[Feature]bool) map[Feature]bool {
		m := map[Feature]bool{}
		for k, v := range toCopy {
			m[k] = v
		}
		return m
	}

	scenarios := []struct {
		name                  string
		features              map[Feature]FeatureSpec
		envVariables          map[string]string
		expectedFeaturesState map[Feature]bool
	}{
		{
			name: "can add empty features",
		},
		{
			name:                  "no env var, features get Defaults assigned",
			features:              defaultTestFeatures,
			expectedFeaturesState: expectedDefaultFeaturesState,
		},
		{
			name:                  "incorrect env var, feature gets Default assigned",
			features:              defaultTestFeatures,
			envVariables:          map[string]string{"TestAlpha": "true"},
			expectedFeaturesState: expectedDefaultFeaturesState,
		},
		{
			name:         "correct env var changes the feature gets state",
			features:     defaultTestFeatures,
			envVariables: map[string]string{"KUBE_FEATURE_TestAlpha": "true"},
			expectedFeaturesState: func() map[Feature]bool {
				expectedDefaultFeaturesStateCopy := copyExpectedStateMap(expectedDefaultFeaturesState)
				expectedDefaultFeaturesStateCopy["TestAlpha"] = true
				return expectedDefaultFeaturesStateCopy
			}(),
		},
		{
			name:                  "incorrect env var value gets ignored",
			features:              defaultTestFeatures,
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": "TrueFalse"},
			expectedFeaturesState: expectedDefaultFeaturesState,
		},
		{
			name: "a feature LockToDefault wins",
			features: map[Feature]FeatureSpec{
				"TestAlpha": {
					Default:       true,
					LockToDefault: true,
					PreRelease:    "Alpha",
				},
			},
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": "False"},
			expectedFeaturesState: map[Feature]bool{"TestAlpha": true},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for k, v := range scenario.envVariables {
				os.Setenv(k, v)
			}
			defer func() {
				for k, _ := range scenario.envVariables {
					os.Unsetenv(k)
				}
			}()
			target := NewEnvVarFeatureGate()
			err := target.Add(scenario.features)
			require.NoError(t, err)

			for expectedFeature, expectedValue := range scenario.expectedFeaturesState {
				actualValue := target.Enabled(expectedFeature)
				require.Equal(t, actualValue, expectedValue, "expected feature=%v, to be=%v, not=%v", expectedFeature, expectedValue, actualValue)
			}
		})
	}
}

func TestAddError(t *testing.T) {
	feature := map[Feature]FeatureSpec{
		"TestAlpha": {
			Default:       true,
			LockToDefault: true,
			PreRelease:    "Alpha",
		},
	}
	target := NewEnvVarFeatureGate()
	err := target.Add(feature)
	require.NoError(t, err)

	alphaSpec := feature["TestAlpha"]
	alphaSpec.Default = false
	feature["TestAlpha"] = alphaSpec
	err = target.Add(feature)
	require.NotNilf(t, err, "expected an error but got none")
}

func TestEnabledPanic(t *testing.T) {
	target := NewEnvVarFeatureGate()
	require.Panics(t, func() { target.Enabled("UnknownFeature") })
}
