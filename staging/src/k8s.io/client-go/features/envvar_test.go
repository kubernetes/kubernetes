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

	"github.com/onsi/gomega"
	"k8s.io/ktesting"
)

var defaultTestFeatures = map[Feature]FeatureSpec{
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

func TestEnvVarFeatureGates(t *testing.T) {
	tCtx := ktesting.Init(t)
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
		setMethodFeatures     map[Feature]bool
		setForTestingFeatures map[Feature]bool

		expectedFeaturesState                           map[Feature]bool
		expectedInternalEnabledViaEnvVarFeatureState    map[Feature]bool
		expectedInternalEnabledViaSetMethodFeatureState map[Feature]bool
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
			expectedInternalEnabledViaEnvVarFeatureState: map[Feature]bool{"TestAlpha": true},
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
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": "True"},
			expectedFeaturesState: map[Feature]bool{"TestAlpha": true},
			expectedInternalEnabledViaEnvVarFeatureState: map[Feature]bool{"TestAlpha": true},
		},
		{
			name:                  "setting a feature via the Set method works",
			features:              defaultTestFeatures,
			setMethodFeatures:     map[Feature]bool{"TestAlpha": true},
			expectedFeaturesState: map[Feature]bool{"TestAlpha": true},
			expectedInternalEnabledViaSetMethodFeatureState: map[Feature]bool{"TestAlpha": true},
		},
		{
			name:                  "setting a feature via the Set method wins",
			features:              defaultTestFeatures,
			setMethodFeatures:     map[Feature]bool{"TestAlpha": false},
			envVariables:          map[string]string{"KUBE_FEATURE_TestAlpha": "True"},
			expectedFeaturesState: map[Feature]bool{"TestAlpha": false},
			expectedInternalEnabledViaEnvVarFeatureState:    map[Feature]bool{"TestAlpha": true},
			expectedInternalEnabledViaSetMethodFeatureState: map[Feature]bool{"TestAlpha": false},
		},
		{
			name: "setting a feature via the SetForTesting method works even when it is locked",
			features: map[Feature]FeatureSpec{
				"TestAlpha": {
					Default:       true,
					LockToDefault: true,
					PreRelease:    "Alpha",
				},
			},
			setForTestingFeatures:                           map[Feature]bool{"TestAlpha": false},
			expectedFeaturesState:                           map[Feature]bool{"TestAlpha": false},
			expectedInternalEnabledViaSetMethodFeatureState: map[Feature]bool{"TestAlpha": false},
		},
	}
	for _, scenario := range scenarios {
		tCtx.Run(scenario.name, func(tCtx ktesting.TContext) {
			for k, v := range scenario.envVariables {
				tCtx.Setenv(k, v)
			}
			target := newEnvVarFeatureGates(scenario.features)

			for k, v := range scenario.setMethodFeatures {
				err := target.Set(k, v)
				tCtx.AssertNoError(err)
			}
			for k, v := range scenario.setForTestingFeatures {
				err := target.SetForTesting(k, v)
				tCtx.AssertNoError(err)
			}
			for expectedFeature, expectedValue := range scenario.expectedFeaturesState {
				actualValue := target.Enabled(expectedFeature)
				tCtx.Assert(actualValue).Should(gomega.Equal(expectedValue), "feature %v", expectedFeature)
			}

			enabledViaEnvVarInternalMap := target.enabledViaEnvVar.Load().(map[Feature]bool)
			tCtx.Assert(enabledViaEnvVarInternalMap).Should(gomega.HaveLen(len(scenario.expectedInternalEnabledViaEnvVarFeatureState)))
			for expectedFeatureName, expectedFeatureValue := range scenario.expectedInternalEnabledViaEnvVarFeatureState {
				actualFeatureValue, wasExpectedFeatureFound := enabledViaEnvVarInternalMap[expectedFeatureName]
				if !wasExpectedFeatureFound {
					tCtx.Errorf("feature %v has not been found in enabledViaEnvVarInternalMap", expectedFeatureName)
				}
				tCtx.Assert(actualFeatureValue).Should(gomega.Equal(expectedFeatureValue), "feature %v", expectedFeatureName)
			}

			enabledViaSetMethodInternalMap := target.enabledViaSetMethod
			tCtx.Assert(enabledViaSetMethodInternalMap).Should(gomega.HaveLen(len(scenario.expectedInternalEnabledViaSetMethodFeatureState)))
			for expectedFeatureName, expectedFeatureValue := range scenario.expectedInternalEnabledViaSetMethodFeatureState {
				actualFeatureValue, wasExpectedFeatureFound := enabledViaSetMethodInternalMap[expectedFeatureName]
				if !wasExpectedFeatureFound {
					tCtx.Errorf("feature %v has not been found in enabledViaSetMethod", expectedFeatureName)
				}
				tCtx.Assert(actualFeatureValue).Should(gomega.Equal(expectedFeatureValue), "feature %v", expectedFeatureName)
			}
		})
	}
}

func TestEnvVarFeatureGatesEnabledPanic(t *testing.T) {
	tCtx := ktesting.Init(t)
	target := newEnvVarFeatureGates(nil)
	tCtx.Require(func() { target.Enabled("UnknownFeature") }).Should(gomega.PanicWith(gomega.MatchError(fmt.Errorf("feature %q is not registered in FeatureGates %q", "UnknownFeature", target.callSiteName))))
}

func TestHasAlreadyReadEnvVar(t *testing.T) {
	tCtx := ktesting.Init(t)
	target := newEnvVarFeatureGates(nil)
	tCtx.Assert(target.hasAlreadyReadEnvVar()).Should(gomega.BeFalseBecause("env var has not been read yet"))

	_ = target.getEnabledMapFromEnvVar()
	tCtx.Assert(target.hasAlreadyReadEnvVar()).Should(gomega.BeTrueBecause("env var has been read"))
}

func TestEnvVarFeatureGatesSetNegative(t *testing.T) {
	tCtx := ktesting.Init(t)
	scenarios := []struct {
		name         string
		features     map[Feature]FeatureSpec
		featureName  Feature
		featureValue bool

		expectedErr func(string) error
	}{
		{
			name:     "empty feature name returns an error",
			features: defaultTestFeatures,
			expectedErr: func(callSiteName string) error {
				return fmt.Errorf("feature %q is not registered in FeatureGates %q", "", callSiteName)
			},
		},
		{
			name:        "setting unknown feature returns an error",
			features:    defaultTestFeatures,
			featureName: "Unknown",
			expectedErr: func(callSiteName string) error {
				return fmt.Errorf("feature %q is not registered in FeatureGates %q", "Unknown", callSiteName)
			},
		},
		{
			name:         "setting locked feature returns an error",
			features:     map[Feature]FeatureSpec{"LockedFeature": {LockToDefault: true, Default: true}},
			featureName:  "LockedFeature",
			featureValue: false,
			expectedErr: func(_ string) error {
				return fmt.Errorf("cannot set feature gate %q to %v, feature is locked to %v", "LockedFeature", false, true)
			},
		},
	}

	for _, scenario := range scenarios {
		tCtx.Run(scenario.name, func(tCtx ktesting.TContext) {
			target := newEnvVarFeatureGates(scenario.features)

			err := target.Set(scenario.featureName, scenario.featureValue)
			tCtx.Require(err).Should(gomega.Equal(scenario.expectedErr(target.callSiteName)))
		})
	}
}
