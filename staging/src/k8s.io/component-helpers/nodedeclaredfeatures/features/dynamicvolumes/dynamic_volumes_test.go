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

package dynamicvolumes

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestDynamicNodeLocalEphemeralVolumesFeatureDiscover(t *testing.T) {
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
			cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{DynamicNodeLocalEphemeralVolumesFeatureGate: tt.featureGate}}
			enabled := Feature.Discover(cfg)
			if want, got := tt.expected, enabled; want != got {
				t.Fatalf("want=%v,got=%v", want, got)
			}
		})
	}
}

func TestDynamicNodeLocalEphemeralVolumesFeatureInferForScheduling(t *testing.T) {
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	if Feature.InferForScheduling(podInfo) {
		t.Fatalf("InferForScheduling should always be false")
	}
}

func TestDynamicNodeLocalEphemeralVolumesFeatureInferForUpdate(t *testing.T) {
	tests := []struct {
		name       string
		oldPodInfo *types.PodInfo
		newPodInfo *types.PodInfo
		expected   bool
	}{
		{
			name: "NoChanges",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}},
					Containers: []v1.Container{
						{
							Name:         "c1",
							VolumeMounts: []v1.VolumeMount{{Name: "vol-1", MountPath: "/mnt/vol-1"}},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}},
					Containers: []v1.Container{
						{
							Name:         "c1",
							VolumeMounts: []v1.VolumeMount{{Name: "vol-1", MountPath: "/mnt/vol-1"}},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "VolumeAdded",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}, {Name: "vol-2"}},
				},
			},
			expected: true,
		},
		{
			name: "VolumeRemoved",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}, {Name: "vol-2"}},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}},
				},
			},
			expected: true,
		},
		{
			name: "VolumeReplacedWithSameCount",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-2"}},
				},
			},
			expected: true,
		},
		{
			name: "ContainerVolumeMountAdded",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}, {Name: "vol-2"}},
					Containers: []v1.Container{
						{
							Name:         "c1",
							VolumeMounts: []v1.VolumeMount{{Name: "vol-1", MountPath: "/mnt/vol-1"}},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{{Name: "vol-1"}, {Name: "vol-2"}},
					Containers: []v1.Container{
						{
							Name: "c1",
							VolumeMounts: []v1.VolumeMount{
								{Name: "vol-1", MountPath: "/mnt/vol-1"},
								{Name: "vol-2", MountPath: "/mnt/vol-2"},
							},
						},
					},
				},
			},
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Feature.InferForUpdate(tt.oldPodInfo, tt.newPodInfo); got != tt.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tt.expected)
			}
		})
	}
}
