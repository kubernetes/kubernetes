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

package features

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	testhelpers "k8s.io/component-helpers/nodedeclaredfeatures/testing"
)

// TestFeatureRequirementsConsistency checks that each feature's Discover method
// actually checks the gates and static configs it declares in Requirements
// and returns the correct boolean value.
func TestFeatureRequirementsConsistency(t *testing.T) {
	for _, feature := range AllFeatures {
		t.Run(feature.Name(), func(t *testing.T) {
			reqs := feature.Requirements()
			if reqs == nil {
				t.Fatalf("Feature %s returned nil Requirements", feature.Name())
			}

			// All Requirements Met
			mockFG := testhelpers.NewMockFeatureGate(t)
			staticConfig := nodedeclaredfeatures.StaticConfiguration{}
			assert.NotEmpty(t, reqs.EnabledFeatureGates, "Feature %s must declare at least one feature gate in Requirements", feature.Name())
			for _, gate := range reqs.EnabledFeatureGates {
				mockFG.EXPECT().CheckEnabled(gate).Return(true, nil)
			}
			for key, value := range reqs.StaticConfig {
				if key == "CPUManagerPolicy" {
					staticConfig.CPUManagerPolicy = value
				}
			}

			discoverCfg := &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates: mockFG,
				StaticConfig: staticConfig,
				Version:      version.MustParse("1.36.0"),
			}

			featureEnabled, err := feature.Discover(discoverCfg)
			require.NoError(t, err)
			assert.True(t, featureEnabled, "Discover should return true when all requirements are met")
			mockFG.AssertExpectations(t)

			// Feature Gate Disabled
			if len(reqs.EnabledFeatureGates) > 0 {
				gateToDisable := reqs.EnabledFeatureGates[0]
				disabledMockFG := testhelpers.NewMockFeatureGate(t)
				for _, gate := range reqs.EnabledFeatureGates {
					expectedReturn := true
					if gate == gateToDisable {
						expectedReturn = false
					}
					disabledMockFG.EXPECT().CheckEnabled(gate).Return(expectedReturn, nil)
				}

				discoverCfg.FeatureGates = disabledMockFG
				// Static config remains the same as case 1
				discoverCfg.StaticConfig = staticConfig
				featureEnabled, err := feature.Discover(discoverCfg)
				require.NoError(t, err)
				assert.False(t, featureEnabled, "Discover should return false when gate %s is disabled for feature %s", gateToDisable, feature.Name())
				disabledMockFG.AssertExpectations(t)
			}

			//  Static Config Mismatch
			if len(reqs.StaticConfig) > 0 {
				mismatchStaticConfig := nodedeclaredfeatures.StaticConfiguration{
					CPUManagerPolicy: "none",
				}
				if val, ok := reqs.StaticConfig["CPUManagerPolicy"]; ok && val != "none" {
					mismatchStaticConfig = nodedeclaredfeatures.StaticConfiguration{CPUManagerPolicy: "static"}
				}
				discoverCfg.FeatureGates = mockFG
				discoverCfg.StaticConfig = mismatchStaticConfig
				featureEnabled, err := feature.Discover(discoverCfg)
				require.NoError(t, err)
				assert.True(t, featureEnabled, "Discover should return true when CPUManagerPolicy is none for feature %s", feature.Name())
				mockFG.AssertExpectations(t)
			}
		})
	}
}
