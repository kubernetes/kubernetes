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

package inplacepodresize

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/component-helpers/nodedeclaredfeatures/types"
)

func TestInitContainerResizeFeatureDiscover(t *testing.T) {
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
			cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{IPPRInitContainersFeatureGate: tt.featureGate}}
			enabled := NonSidecarInitContainerResizeFeature.Discover(cfg)
			if want, got := tt.expected, enabled; want != got {
				t.Fatalf("want=%v,got=%v", want, got)
			}
		})
	}
}

func TestNonSidecarInitContainerResizeFeatureInferForScheduling(t *testing.T) {
	f := &nonSidecarInitContainerResizeFeature{}
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	if f.InferForScheduling(podInfo) {
		t.Fatalf("InferForScheduling should always be false")
	}
}

func TestNonSidecarInitContainerResizeFeatureInferForUpdate(t *testing.T) {
	restartAlways := v1.ContainerRestartPolicyAlways

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
					InitContainers: []v1.Container{
						{
							Name: "init-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "init-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "InitContainerResourceChanged",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "init-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name: "init-1",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "SidecarContainerResourceChanged_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "MixedContainers_OnlyInitChanged",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
						},
						{
							Name: "regular-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
						},
						{
							Name: "regular-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "MixedContainers_OnlySidecarChanged_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
						{
							Name: "regular-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:          "sidecar",
							RestartPolicy: &restartAlways,
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("200m")},
							},
						},
						{
							Name: "regular-init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("100m")},
							},
						},
					},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &nonSidecarInitContainerResizeFeature{}
			got := f.InferForUpdate(tt.oldPodInfo, tt.newPodInfo)
			if got != tt.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tt.expected)
			}
		})
	}
}
