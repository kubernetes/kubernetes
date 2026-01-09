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

package inplacepodresize

import (
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestPodLevelResourcesResizeFeatureDiscover(t *testing.T) {
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
			cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{IPPRPodLevelResourcesFeatureGate: tt.featureGate}}
			enabled := PodLevelResourcesResizeFeature.Discover(cfg)
			assert.Equal(t, tt.expected, enabled)
		})
	}
}

func TestPodLevelResourcesResizeFeatureInferForScheduling(t *testing.T) {
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}, Status: &v1.PodStatus{}}
	assert.False(t, PodLevelResourcesResizeFeature.InferForScheduling(podInfo), "InferForScheduling should always be false")
}

func TestPodLevelResourcesResizeFeatureInferForUpdate(t *testing.T) {
	tests := []struct {
		name       string
		oldPodInfo *types.PodInfo
		newPodInfo *types.PodInfo
		expected   bool
	}{
		{
			name: "NoResourcesChanged",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			expected: false,
		},
		{
			name: "ResourcesAdded",
			oldPodInfo: &types.PodInfo{
				Spec:   &v1.PodSpec{},
				Status: &v1.PodStatus{},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			expected: true,
		},
		{
			name: "ResourcesRemoved",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			newPodInfo: &types.PodInfo{
				Spec:   &v1.PodSpec{},
				Status: &v1.PodStatus{},
			},
			expected: true,
		},
		{
			name: "ResourcesChanged",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("2")},
						Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				},
				Status: &v1.PodStatus{},
			},
			expected: true,
		},
		{
			name: "NilResources",
			oldPodInfo: &types.PodInfo{
				Spec:   &v1.PodSpec{},
				Status: &v1.PodStatus{},
			},
			newPodInfo: &types.PodInfo{
				Spec:   &v1.PodSpec{},
				Status: &v1.PodStatus{},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, PodLevelResourcesResizeFeature.InferForUpdate(tt.oldPodInfo, tt.newPodInfo))
		})
	}
}
