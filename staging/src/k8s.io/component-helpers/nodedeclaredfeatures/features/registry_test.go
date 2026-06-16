/*
Copyright The Kubernetes Authors.

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

package features_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures/features"
	testhelpers "k8s.io/component-helpers/nodedeclaredfeatures/testing"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

// TestFeatureRequirementsConsistency checks that each feature's Discover method
// actually checks the gates it declares in Requirements
// and returns the correct boolean value.
func TestFeatureRequirementsConsistency(t *testing.T) {
	for _, registeredFeature := range features.AllFeatures {
		t.Run(registeredFeature.Name(), func(t *testing.T) {
			reqs := registeredFeature.Requirements()
			if reqs == nil {
				t.Fatalf("Feature %s returned nil Requirements", registeredFeature.Name())
			}

			if len(reqs.EnabledFeatureGates) == 0 {
				t.Fatalf("Feature %s must declare at least one feature gate in Requirements", registeredFeature.Name())
			}

			// Feature Gates Enabled
			mockFG := testhelpers.NewMockFeatureGate(t)
			for _, gate := range reqs.EnabledFeatureGates {
				mockFG.SetEnabled(gate, true)
			}

			discoverCfg := &types.NodeConfiguration{
				FeatureGates: mockFG,
				Version:      version.MustParse("1.36.0"),
			}
			if reqs.RequiredRuntimeFeatures != nil {
				discoverCfg.RuntimeFeatures = *reqs.RequiredRuntimeFeatures
			}

			featureEnabled := registeredFeature.Discover(discoverCfg)
			if !featureEnabled {
				t.Fatalf("Discover should return true when all requirements are met for feature %s", registeredFeature.Name())
			}

			// Feature Gate Disabled
			gateToDisable := reqs.EnabledFeatureGates[0]
			disabledMockFG := testhelpers.NewMockFeatureGate(t)
			for _, gate := range reqs.EnabledFeatureGates {
				expectedReturn := true
				if gate == gateToDisable {
					expectedReturn = false
				}
				disabledMockFG.SetEnabled(gate, expectedReturn)
			}

			discoverCfg.FeatureGates = disabledMockFG
			featureEnabled = registeredFeature.Discover(discoverCfg)
			if featureEnabled {
				t.Fatalf("Discover should return false when gate %s is disabled for feature %s", gateToDisable, registeredFeature.Name())
			}

		})
	}
}
