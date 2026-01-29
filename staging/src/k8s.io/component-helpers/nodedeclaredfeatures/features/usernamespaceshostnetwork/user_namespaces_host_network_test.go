/*
Copyright 2025 The Kubernetes Authors.

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
		runtimeHandlers []nodedeclaredfeatures.RuntimeHandlerInfo
		expected        bool
	}{
		{
			name:        "feature gate disabled",
			featureGate: false,
			runtimeHandlers: []nodedeclaredfeatures.RuntimeHandlerInfo{
				{Name: "runc", SupportsUserNamespacesHostNetwork: true},
			},
			expected: false,
		},
		{
			name:        "feature gate enabled but no runtime support",
			featureGate: true,
			runtimeHandlers: []nodedeclaredfeatures.RuntimeHandlerInfo{
				{Name: "runc", SupportsUserNamespacesHostNetwork: false},
			},
			expected: false,
		},
		{
			name:        "feature gate enabled and runtime supports it",
			featureGate: true,
			runtimeHandlers: []nodedeclaredfeatures.RuntimeHandlerInfo{
				{Name: "runc", SupportsUserNamespacesHostNetwork: true},
			},
			expected: true,
		},
		{
			name:        "multiple handlers, at least one supports it",
			featureGate: true,
			runtimeHandlers: []nodedeclaredfeatures.RuntimeHandlerInfo{
				{Name: "runc", SupportsUserNamespacesHostNetwork: false},
				{Name: "runsc", SupportsUserNamespacesHostNetwork: true},
			},
			expected: true,
		},
		{
			name:            "feature gate enabled but no handlers",
			featureGate:     true,
			runtimeHandlers: []nodedeclaredfeatures.RuntimeHandlerInfo{},
			expected:        false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockFG := test.NewMockFeatureGate(t)
			mockFG.EXPECT().Enabled(UserNamespacesHostNetworkSupportFeatureGate).Return(tt.featureGate)

			cfg := &nodedeclaredfeatures.NodeConfiguration{
				FeatureGates:    mockFG,
				RuntimeHandlers: tt.runtimeHandlers,
			}

			result := Feature.Discover(cfg)
			assert.Equal(t, tt.expected, result)
		})
	}
}
