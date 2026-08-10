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

package downwardapiassignedresources

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestDownwardAPIAssignedResourcesFeature_Name(t *testing.T) {
	if Feature.Name() != DownwardAPIAssignedResources {
		t.Errorf("Expected name %s, got %s", DownwardAPIAssignedResources, Feature.Name())
	}
}

func TestDownwardAPIAssignedResourcesFeature_Discover(t *testing.T) {
	tests := []struct {
		name           string
		featureEnabled bool
		expectDiscovered bool
	}{
		{
			name:           "feature gate enabled",
			featureEnabled: true,
			expectDiscovered: true,
		},
		{
			name:           "feature gate disabled",
			featureEnabled: false,
			expectDiscovered: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &types.NodeConfiguration{
				FeatureGates: types.FeatureGateMap{DownwardAPIAssignedResources: tt.featureEnabled},
			}
			result := Feature.Discover(cfg)
			if result != tt.expectDiscovered {
				t.Errorf("Discover() = %v, want %v", result, tt.expectDiscovered)
			}
		})
	}
}

func TestDownwardAPIAssignedResourcesFeature_Requirements(t *testing.T) {
	reqs := Feature.Requirements()
	if reqs == nil {
		t.Fatal("Requirements() returned nil")
	}
	if len(reqs.EnabledFeatureGates) != 1 {
		t.Errorf("Expected 1 feature gate, got %d", len(reqs.EnabledFeatureGates))
	}
	if reqs.EnabledFeatureGates[0] != DownwardAPIAssignedResources {
		t.Errorf("Expected feature gate %s, got %s", DownwardAPIAssignedResources, reqs.EnabledFeatureGates[0])
	}
}

func TestDownwardAPIAssignedResourcesFeature_InferForScheduling(t *testing.T) {
	tests := []struct {
		name     string
		pod      *types.PodInfo
		expected bool
	}{
		{
			name: "pod with assigned.cpuset in downwardAPI volume",
			pod: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								DownwardAPI: &v1.DownwardAPIVolumeSource{
									Items: []v1.DownwardAPIVolumeFile{
										{
											Path: "cpuset",
											ResourceFieldRef: &v1.ResourceFieldSelector{
												ContainerName: "test-container",
												Resource:      "assigned.cpuset",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod with assigned.cpuset in projected downwardAPI volume",
			pod: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								Projected: &v1.ProjectedVolumeSource{
									Sources: []v1.VolumeProjection{
										{
											DownwardAPI: &v1.DownwardAPIProjection{
												Items: []v1.DownwardAPIVolumeFile{
													{
														Path: "cpuset",
														ResourceFieldRef: &v1.ResourceFieldSelector{
															ContainerName: "test-container",
															Resource:      "assigned.cpuset",
														},
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod without assigned.cpuset",
			pod: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								DownwardAPI: &v1.DownwardAPIVolumeSource{
									Items: []v1.DownwardAPIVolumeFile{
										{
											Path: "cpu_limit",
											ResourceFieldRef: &v1.ResourceFieldSelector{
												ContainerName: "test-container",
												Resource:      "limits.cpu",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "pod with nil spec",
			pod:  &types.PodInfo{Spec: nil},
			expected: false,
		},
		{
			name: "pod with no volumes",
			pod: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Feature.InferForScheduling(tt.pod)
			if result != tt.expected {
				t.Errorf("InferForScheduling() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDownwardAPIAssignedResourcesFeature_InferForUpdate(t *testing.T) {
	// Since pod volumes are immutable, InferForUpdate should always return false.
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	if Feature.InferForUpdate(nil, podInfo) {
		t.Fatalf("expect InferForUpdate to be false")
	}
}

