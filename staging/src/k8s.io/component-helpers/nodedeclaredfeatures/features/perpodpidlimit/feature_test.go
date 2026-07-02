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

package perpodpidlimit

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestPerPodPIDLimitFeature_Requirements(t *testing.T) {
	feature := &perPodPIDLimitFeature{}
	reqs := feature.Requirements()
	if reqs == nil {
		t.Fatalf("Feature %s returned nil Requirements", feature.Name())
	}
	if len(reqs.EnabledFeatureGates) != 1 || reqs.EnabledFeatureGates[0] != PerPodPIDLimitFeatureGate {
		t.Fatalf("Feature %s Requirements should declare exactly the %s feature gate", feature.Name(), PerPodPIDLimitFeatureGate)
	}
}

func TestPerPodPIDLimitFeature_Discover(t *testing.T) {
	tests := []struct {
		name        string
		featureGate bool
		cgroupsv2   bool
		expected    bool
	}{
		{
			name:        "GateEnabled_Cgroupsv2",
			featureGate: true,
			cgroupsv2:   true,
			expected:    true,
		},
		{
			name:        "GateEnabled_Cgroupsv1",
			featureGate: true,
			cgroupsv2:   false,
			expected:    false,
		},
		{
			name:        "GateDisabled_Cgroupsv2",
			featureGate: false,
			cgroupsv2:   true,
			expected:    false,
		},
		{
			name:        "GateDisabled_Cgroupsv1",
			featureGate: false,
			cgroupsv2:   false,
			expected:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{PerPodPIDLimitFeatureGate: tt.featureGate},
				StaticConfig: types.StaticConfiguration{Cgroupsv2: tt.cgroupsv2},
			}
			if got := Feature.Discover(cfg); got != tt.expected {
				t.Fatalf("Discover() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPerPodPIDLimitFeature_InferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		podInfo  *types.PodInfo
		expected bool
	}{
		{
			name: "PodWithPIDLimit",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("2048")},
					},
				},
			},
			expected: true,
		},
		{
			name: "PodWithoutPIDLimit",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
			},
			expected: false,
		},
		{
			name: "PodWithNilResources",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{},
			},
			expected: false,
		},
		{
			name: "PodWithEmptyLimits",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{},
					},
				},
			},
			expected: false,
		},
		{
			name: "PodWithPIDAndOtherLimits",
			podInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("2"),
							v1.ResourcePID: resource.MustParse("4096"),
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Feature.InferForScheduling(tt.podInfo); got != tt.expected {
				t.Fatalf("InferForScheduling() = %v, want %v", got, tt.expected)
			}
		})
	}
}

func TestPerPodPIDLimitFeature_InferForUpdate(t *testing.T) {
	oldPodInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	newPodInfo := &types.PodInfo{
		Spec: &v1.PodSpec{
			Resources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{v1.ResourcePID: resource.MustParse("2048")},
			},
		},
	}
	if Feature.InferForUpdate(oldPodInfo, newPodInfo) {
		t.Fatal("InferForUpdate should always return false (PID limits are immutable)")
	}
}
