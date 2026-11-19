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

package dranodeallocatableresources

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestRequirements(t *testing.T) {
	feature := &draNodeAllocatableResourcesFeature{}
	if feature.Name() != DRANodeAllocatableResourcesFeature {
		t.Fatalf("expected Name to be %s, got %s", DRANodeAllocatableResourcesFeature, feature.Name())
	}
	reqs := feature.Requirements()
	if reqs == nil {
		t.Fatalf("Feature %s returned nil Requirements", feature.Name())
	}

	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != DRANodeAllocatableResourcesFeatureGate {
		t.Fatalf("Feature %s Requirements should declare exactly the %s feature gate", feature.Name(), DRANodeAllocatableResourcesFeatureGate)
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

	feature := &draNodeAllocatableResourcesFeature{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{DRANodeAllocatableResourcesFeatureGate: tc.featureGateEnabled},
			}
			enabled := feature.Discover(config)
			if want, got := tc.expected, enabled; want != got {
				t.Fatalf("want=%v,got=%v", want, got)
			}
		})
	}
}

func TestInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		podInfo  *types.PodInfo
		expected bool
	}{
		{
			name:     "no status",
			podInfo:  &types.PodInfo{Spec: &v1.PodSpec{}},
			expected: false,
		},
		{
			name:     "no extra node allocatable resources",
			podInfo:  &types.PodInfo{Spec: &v1.PodSpec{}, Status: &v1.PodStatus{}},
			expected: false,
		},
		{
			name: "extra node allocatable resources",
			podInfo: &types.PodInfo{Spec: &v1.PodSpec{}, Status: &v1.PodStatus{
				ExtraNodeAllocatableResources: []v1.ExtraNodeAllocatableResource{{
					Limits: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
				}},
			}},
			expected: true,
		},
	}

	feature := &draNodeAllocatableResourcesFeature{}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := feature.InferForScheduling(tc.podInfo); got != tc.expected {
				t.Fatalf("InferForScheduling() = %v, want %v", got, tc.expected)
			}
		})
	}
}

func TestInferForUpdate(t *testing.T) {
	feature := &draNodeAllocatableResourcesFeature{}
	if feature.InferForUpdate(nil, &types.PodInfo{}) {
		t.Fatalf("expect InferForUpdate to be false")
	}
}

func TestMaxVersion(t *testing.T) {
	feature := &draNodeAllocatableResourcesFeature{}
	if feature.MaxVersion() != nil {
		t.Fatalf("expect MaxVersion to be nil")
	}
}
