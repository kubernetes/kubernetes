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

package podlevelresources

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestEphemeralStorageFeatureRequirements(t *testing.T) {
	reqs := EphemeralStorageFeature.Requirements()
	if reqs == nil {
		t.Fatalf("Feature %s returned nil Requirements", EphemeralStorageFeature.Name())
	}
	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != PodLevelResourcesEphemeralStorageFeatureGate {
		t.Fatalf("Feature %s Requirements should declare %s, got %v", EphemeralStorageFeature.Name(), PodLevelResourcesEphemeralStorageFeatureGate, reqs.EnabledFeatureGates)
	}
}

func TestEphemeralStorageFeatureDiscover(t *testing.T) {
	tests := []struct {
		name        string
		featureGate bool
		expected    bool
	}{
		{
			name:        "FeatureGateEnabled",
			featureGate: true,
			expected:    true,
		},
		{
			name:        "FeatureGateDisabled",
			featureGate: false,
			expected:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{PodLevelResourcesEphemeralStorageFeatureGate: tt.featureGate},
			}
			if got := EphemeralStorageFeature.Discover(cfg); got != tt.expected {
				t.Fatalf("Discover() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestEphemeralStorageFeatureInferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		podInfo  *types.PodInfo
		expected bool
	}{
		{
			name:     "NilPodInfo",
			podInfo:  nil,
			expected: false,
		},
		{
			name:     "NilSpec",
			podInfo:  &types.PodInfo{},
			expected: false,
		},
		{
			name:     "NilPodLevelResources",
			podInfo:  &types.PodInfo{Spec: &v1.PodSpec{}},
			expected: false,
		},
		{
			name: "OnlyPodLevelCPUAndMemory",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("1"),
							v1.ResourceMemory: resource.MustParse("1Gi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("2"),
							v1.ResourceMemory: resource.MustParse("2Gi"),
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "OnlyContainerLevelEphemeralStorage",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "c1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceEphemeralStorage: resource.MustParse("1Gi")},
								Limits:   v1.ResourceList{v1.ResourceEphemeralStorage: resource.MustParse("2Gi")},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "PodLevelEphemeralStorageRequest",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceEphemeralStorage: resource.MustParse("5Gi"),
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "PodLevelEphemeralStorageLimit",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceEphemeralStorage: resource.MustParse("10Gi"),
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := EphemeralStorageFeature.InferForScheduling(tt.podInfo); got != tt.expected {
				t.Fatalf("InferForScheduling() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestEphemeralStorageFeatureInferForUpdate(t *testing.T) {
	oldPod := &types.PodInfo{Spec: &v1.PodSpec{}}
	newPod := &types.PodInfo{
		Spec: &v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{v1.ResourceEphemeralStorage: resource.MustParse("5Gi")},
			},
		},
	}
	if EphemeralStorageFeature.InferForUpdate(oldPod, newPod) {
		t.Fatalf("InferForUpdate() = true, want false")
	}
}
