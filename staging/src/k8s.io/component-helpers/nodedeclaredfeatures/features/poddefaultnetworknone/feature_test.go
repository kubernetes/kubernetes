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

package poddefaultnetworknone

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
	"k8s.io/utils/ptr"
)

type fakeFeatureGate struct {
	features map[string]bool
}

func (m *fakeFeatureGate) Enabled(key string) bool {
	return m.features[key]
}

func TestDiscover(t *testing.T) {
	tests := []struct {
		name        string
		featureGate bool
		runtime     bool
		expected    bool
	}{
		{name: "gate disabled, runtime supports", featureGate: false, runtime: true, expected: false},
		{name: "gate enabled, runtime does not support", featureGate: true, runtime: false, expected: false},
		{name: "gate enabled, runtime supports", featureGate: true, runtime: true, expected: true},
		{name: "gate disabled, runtime does not support", featureGate: false, runtime: false, expected: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates:    &fakeFeatureGate{features: map[string]bool{PodDefaultNetworkFeatureGate: tt.featureGate}},
				RuntimeFeatures: types.RuntimeFeatures{DefaultNetworkNone: tt.runtime},
			}
			if got := Feature.Discover(cfg); got != tt.expected {
				t.Fatalf("Discover() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestRequirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil {
		t.Fatalf("Requirements returned nil")
	}
	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != PodDefaultNetworkFeatureGate {
		t.Fatalf("unexpected required feature gates: %v", reqs.EnabledFeatureGates)
	}
	if reqs.RequiredRuntimeFeatures == nil || !reqs.RequiredRuntimeFeatures.DefaultNetworkNone {
		t.Fatalf("unexpected required runtime features: %v", reqs.RequiredRuntimeFeatures)
	}
}

func TestName(t *testing.T) {
	if Feature.Name() != PodDefaultNetworkNone {
		t.Fatalf("expected Name to be %s, got %s", PodDefaultNetworkNone, Feature.Name())
	}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name           string
		defaultNetwork *v1.PodDefaultNetwork
		expected       bool
	}{
		{name: "unset", defaultNetwork: nil, expected: false},
		{name: "Pod", defaultNetwork: ptr.To(v1.PodDefaultNetworkPod), expected: false},
		{name: "Host", defaultNetwork: ptr.To(v1.PodDefaultNetworkHost), expected: false},
		{name: "None", defaultNetwork: ptr.To(v1.PodDefaultNetworkNone), expected: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podInfo := &types.PodInfo{Spec: &v1.PodSpec{DefaultNetwork: tt.defaultNetwork}}
			if got := Feature.InferForScheduling(podInfo); got != tt.expected {
				t.Fatalf("InferForScheduling() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	oldPod := &types.PodInfo{Spec: &v1.PodSpec{}}
	newPod := &types.PodInfo{Spec: &v1.PodSpec{DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone)}}
	if Feature.InferForUpdate(oldPod, newPod) {
		t.Fatalf("InferForUpdate() = true, want false: defaultNetwork is immutable")
	}
}
