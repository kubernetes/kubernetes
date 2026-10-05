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

func TestMemoryBackedVolumesResizeFeatureDiscover(t *testing.T) {
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
			cfg := &types.NodeConfiguration{FeatureGates: types.FeatureGateMap{IPPRMemoryBackedVolumesFeatureGate: tt.featureGate}}
			enabled := MemoryBackedVolumesResizeFeature.Discover(cfg)
			if want, got := tt.expected, enabled; want != got {
				t.Fatalf("want=%v,got=%v", want, got)
			}
		})
	}
}

func TestMemoryBackedVolumesResizeFeatureInferForScheduling(t *testing.T) {
	f := &memoryBackedVolumesResizeFeature{}
	podInfo := &types.PodInfo{Spec: &v1.PodSpec{}}
	if f.InferForScheduling(podInfo) {
		t.Fatalf("InferForScheduling should always be false")
	}
}

func TestMemoryBackedVolumesResizeFeatureInferForUpdate(t *testing.T) {
	quantity100Mi := resource.MustParse("100Mi")
	quantity200Mi := resource.MustParse("200Mi")

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
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "SizeLimitIncreased",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity200Mi,
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "SizeLimitDecreased",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity200Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "DefaultMedium_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumDefault,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumDefault,
									SizeLimit: &quantity200Mi,
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "NonEmptyDirVolume_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: "/tmp",
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: "/tmp2",
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "VolumeCountMismatch_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
						{
							Name: "vol-2",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "VolumeNameMismatch_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-2",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity200Mi,
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "NilSizeLimit_Ignored",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: nil,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-1",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity200Mi,
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "MultipleVolumes_OneResizableMemoryBackedVolumeResized",
			oldPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-non-resizable",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: "/tmp",
								},
							},
						},
						{
							Name: "vol-resizable",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity100Mi,
								},
							},
						},
					},
				},
			},
			newPodInfo: &types.PodInfo{
				Spec: &v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name: "vol-non-resizable",
							VolumeSource: v1.VolumeSource{
								HostPath: &v1.HostPathVolumeSource{
									Path: "/tmp",
								},
							},
						},
						{
							Name: "vol-resizable",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									Medium:    v1.StorageMediumMemory,
									SizeLimit: &quantity200Mi,
								},
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
			f := &memoryBackedVolumesResizeFeature{}
			got := f.InferForUpdate(tt.oldPodInfo, tt.newPodInfo)
			if got != tt.expected {
				t.Errorf("InferForUpdate() = %v, want %v", got, tt.expected)
			}
		})
	}
}
