/*
Copyright 2024 The Kubernetes Authors.

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
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEnvVarFeatureGates(t *testing.T) {
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
		name                                string
		features                            map[Feature]FeatureSpec
		envVariables                        map[string]string
		expectedFeaturesState               map[Feature]bool
		expectedInternalEnabledFeatureState map[Feature]bool
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
			expectedInternalEnabledFeatureState: map[Feature]bool{"TestAlpha": true},
		},
		{
			name:                  "incorrect env var value gets ignored",
			features:              defaultTestFeatures,
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": "TrueFalse"},
			expectedFeaturesState: expectedDefaultFeaturesState,
		},
		{
			name:                  "empty env var value gets ignored",
			features:              defaultTestFeatures,
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": ""},
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
		{
			name: "setting a feature to LockToDefault changes the internal state",
			features: map[Feature]FeatureSpec{
				"TestAlpha": {
					Default:       true,
					LockToDefault: true,
					PreRelease:    "Alpha",
				},
			},
			envVariables:                        map[string]string{"KUBE_FEATURE_TestAlpha": "True"},
			expectedFeaturesState:               map[Feature]bool{"TestAlpha": true},
			expectedInternalEnabledFeatureState: map[Feature]bool{"TestAlpha": true},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for k, v := range scenario.envVariables {
				t.Setenv(k, v)
			}
			target := NewEnvVarFeatureGates(scenario.features)

			for expectedFeature, expectedValue := range scenario.expectedFeaturesState {
				actualValue := target.Enabled(expectedFeature)
				require.Equal(t, actualValue, expectedValue, "expected feature=%v, to be=%v, not=%v", expectedFeature, expectedValue, actualValue)
			}

			enabledInternalMap := target.enabled.Load().(map[Feature]bool)
			require.Len(t, enabledInternalMap, len(scenario.expectedInternalEnabledFeatureState))

			for expectedFeature, expectedInternalPresence := range scenario.expectedInternalEnabledFeatureState {
				featureInternalValue, featureSet := enabledInternalMap[expectedFeature]
				require.Equal(t, expectedInternalPresence, featureSet, "feature %v present = %v, expected = %v", expectedFeature, featureSet, expectedInternalPresence)

				expectedFeatureInternalValue := scenario.expectedFeaturesState[expectedFeature]
				require.Equal(t, expectedFeatureInternalValue, featureInternalValue)
			}
		})
	}
}

func TestEnvVarFeatureGatesEnabledPanic(t *testing.T) {
	target := NewEnvVarFeatureGates(nil)
	require.PanicsWithError(t, fmt.Errorf("feature %q is not registered in FeatureGates %q", "UnknownFeature", target.callSiteName).Error(), func() { target.Enabled("UnknownFeature") })
}

func TestHasAlreadyReadEnvVar(t *testing.T) {
	target := NewEnvVarFeatureGates(nil)
	require.False(t, target.hasAlreadyReadEnvVar())

	_ = target.getEnabledMapFromEnvVar()
	require.True(t, target.hasAlreadyReadEnvVar())
}
