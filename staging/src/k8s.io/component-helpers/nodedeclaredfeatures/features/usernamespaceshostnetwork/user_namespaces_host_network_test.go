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

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

type fakeFeatureGate struct {
	features map[string]bool
}

func (m *fakeFeatureGate) Enabled(key string) bool {
	return m.features[key]
}

func TestDiscoverFeature(t *testing.T) {
	tests := []struct {
		name            string
		featureGate     bool
		runtimeFeatures types.RuntimeFeatures
		expected        bool
	}{
		{
			name:        "feature gate disabled",
			featureGate: false,
			runtimeFeatures: types.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: false,
		},
		{
			name:        "feature gate enabled but no runtime support",
			featureGate: true,
			runtimeFeatures: types.RuntimeFeatures{
				UserNamespacesHostNetwork: false,
			},
			expected: false,
		},
		{
			name:        "feature gate enabled and runtime supports it",
			featureGate: true,
			runtimeFeatures: types.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: true,
		},
		{
			name:        "runtime support is on",
			featureGate: true,
			runtimeFeatures: types.RuntimeFeatures{
				UserNamespacesHostNetwork: true,
			},
			expected: true,
		},
		{
			name:        "runtime support is off",
			featureGate: true,
			runtimeFeatures: types.RuntimeFeatures{
				UserNamespacesHostNetwork: false,
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockFG := &fakeFeatureGate{
				features: map[string]bool{
					UserNamespacesHostNetworkSupportFeatureGate: tt.featureGate,
				},
			}

			cfg := &types.NodeConfiguration{
				FeatureGates:    mockFG,
				RuntimeFeatures: tt.runtimeFeatures,
			}

			result := Feature.Discover(cfg)
			if result != tt.expected {
				t.Fatalf("Feature.Discover() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestRequirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil {
		t.Fatalf("Requirements returned nil")
	}
	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != UserNamespacesHostNetworkSupportFeatureGate {
		t.Fatalf("unexpected required feature gates: %v", reqs.EnabledFeatureGates)
	}
	if reqs.RequiredRuntimeFeatures == nil || !reqs.RequiredRuntimeFeatures.UserNamespacesHostNetwork {
		t.Fatalf("unexpected required runtime features: %v", reqs.RequiredRuntimeFeatures)
	}
}

func TestName(t *testing.T) {
	if Feature.Name() != UserNamespacesHostNetworkSupport {
		t.Fatalf("expected Name to be %s, got %s", UserNamespacesHostNetworkSupport, Feature.Name())
	}
}

func TestInferForScheduling(t *testing.T) {
	falseVal := false
	trueVal := true

	tests := []struct {
		name        string
		hostNetwork bool
		hostUsers   *bool
		expected    bool
	}{
		{
			name:        "HostNetwork true, HostUsers false",
			hostNetwork: true,
			hostUsers:   &falseVal,
			expected:    true,
		},
		{
			name:        "HostNetwork true, HostUsers nil",
			hostNetwork: true,
			hostUsers:   nil,
			expected:    false,
		},
		{
			name:        "HostNetwork true, HostUsers true",
			hostNetwork: true,
			hostUsers:   &trueVal,
			expected:    false,
		},
		{
			name:        "HostNetwork false, HostUsers false",
			hostNetwork: false,
			hostUsers:   &falseVal,
			expected:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec := &v1.PodSpec{
				HostNetwork: tt.hostNetwork,
				HostUsers:   tt.hostUsers,
			}
			podInfo := &types.PodInfo{Spec: podSpec}
			result := Feature.InferForScheduling(podInfo)
			if result != tt.expected {
				t.Fatalf("InferForScheduling() = %v, want %v", result, tt.expected)
			}
		})
	}
}
