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

package usernamespaceshostnetwork

import (
	"testing"

	"k8s.io/component-helpers/nodedeclaredfeatures"
	test "k8s.io/component-helpers/nodedeclaredfeatures/testing"

	"github.com/stretchr/testify/assert"
)

func TestDiscoverFeature(t *testing.T) {
	tests := []struct {
		name            string
		featureGate     bool
		runtimeFeatures nodedeclaredfeatures.RuntimeFeatures
		expected        bool
	}{
		{
			name:        "feature gate disabled",
			featureGate: false,
			runtimeFeatures: nodedeclaredfeatures.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: false,
		},
		{
			name:        "feature gate enabled but no runtime support",
			featureGate: true,
			runtimeFeatures: nodedeclaredfeatures.RuntimeFeatures{
				UserNamespacesHostNetwork: false,
			},
			expected: false,
		},
		{
			name:        "feature gate enabled and runtime supports it",
			featureGate: true,
			runtimeFeatures: nodedeclaredfeatures.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: true,
		},
		{
			name:        "runtime support is on",
			featureGate: true,
			runtimeFeatures: nodedeclaredfeatures.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: true,
		},
		{
			name:        "runtime support is off",
			featureGate: true,
			runtimeFeatures: nodedeclaredfeatures.RuntimeFeatures{
				UserNamespacesHostNetwork: false,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockFG := test.NewMockFeatureGate(t)
			mockFG.EXPECT().Enabled(UserNamespacesHostNetworkSupportFeatureGate).Return(tt.featureGate)

			cfg := &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:    mockFG,
				RuntimeFeatures: tt.runtimeFeatures,
			}

			result := Feature.Discover(cfg)
			assert.Equal(t, tt.expected, result)
		})
	}
}
