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

package extendwebsocketstokubelet

import (
	"testing"

	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestRequirements(t *testing.T) {
	feature := &extendWebSocketsToKubeletFeature{}
	if feature.Name() != ExtendWebSocketsToKubeletFeatureGate {
		t.Fatalf("expected Name to be %s, got %s", ExtendWebSocketsToKubeletFeatureGate, feature.Name())
	}
	reqs := feature.Requirements()
	if reqs == nil {
		t.Fatalf("Feature %s returned nil Requirements", feature.Name())
	}

	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != ExtendWebSocketsToKubeletFeatureGate {
		t.Fatalf("Feature %s Requirements should declare exactly the %s feature gate", feature.Name(), ExtendWebSocketsToKubeletFeatureGate)
	}
}

func TestDiscover(t *testing.T) {
	tests := []struct {
		name               string
		featureGateEnabled bool
		expected           bool
	}{
		{
			name:               "feature enabled",
			featureGateEnabled: true,
			expected:           true,
		},
		{
			name:               "feature disabled",
			featureGateEnabled: false,
			expected:           false,
		},
	}

	feature := &extendWebSocketsToKubeletFeature{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{ExtendWebSocketsToKubeletFeatureGate: tc.featureGateEnabled},
			}
			enabled := feature.Discover(config)
			if want, got := tc.expected, enabled; want != got {
				t.Fatalf("want=%v,got=%v", want, got)
			}
		})
	}
}

func TestInferForSchedulingAndUpdate(t *testing.T) {
	feature := &extendWebSocketsToKubeletFeature{}
	podInfo := &types.PodInfo{}
	if feature.InferForScheduling(podInfo) {
		t.Fatalf("expect InferForScheduling to be false")
	}
	if feature.InferForUpdate(nil, podInfo) {
		t.Fatalf("expect InferForUpdate to be false")
	}
}

func TestMaxVersion(t *testing.T) {
	feature := &extendWebSocketsToKubeletFeature{}
	if feature.MaxVersion() != nil {
		t.Fatalf("expect MaxVersion to be nil")
	}
}
