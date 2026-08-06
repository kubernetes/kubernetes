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

package state

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
)

func TestPodResourceInfoMap_Clone(t *testing.T) {
	tests := []struct {
		name     string
		original PodResourceInfoMap
		expected PodResourceInfoMap
	}{
		{
			name:     "nil map clone returns empty non-nil map",
			original: nil,
			expected: make(PodResourceInfoMap),
		},
		{
			name:     "empty map clone returns empty non-nil map",
			original: make(PodResourceInfoMap),
			expected: make(PodResourceInfoMap),
		},
		{
			name: "basic cloning with all fields populated",
			original: PodResourceInfoMap{
				types.UID("pod"): {
					PodSpec: &v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "container-a",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("256Mi"),
									},
									Limits: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("200m"),
										v1.ResourceMemory: resource.MustParse("512Mi"),
									},
								},
							},
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("256Mi"),
							},
						},
						Volumes: []v1.Volume{
							{
								Name: "vol-x",
								VolumeSource: v1.VolumeSource{
									EmptyDir: &v1.EmptyDirVolumeSource{
										SizeLimit: resource.NewQuantity(2, resource.BinarySI),
									},
								},
							},
						},
					},
				},
			},
			expected: PodResourceInfoMap{
				types.UID("pod"): {
					PodSpec: &v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "container-a",
								Resources: v1.ResourceRequirements{
									Requests: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("100m"),
										v1.ResourceMemory: resource.MustParse("256Mi"),
									},
									Limits: v1.ResourceList{
										v1.ResourceCPU:    resource.MustParse("200m"),
										v1.ResourceMemory: resource.MustParse("512Mi"),
									},
								},
							},
						},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("256Mi"),
							},
						},
						Volumes: []v1.Volume{
							{
								Name: "vol-x",
								VolumeSource: v1.VolumeSource{
									EmptyDir: &v1.EmptyDirVolumeSource{
										SizeLimit: resource.NewQuantity(2, resource.BinarySI),
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "cloning with missing or partially nil fields",
			original: PodResourceInfoMap{
				types.UID("pod"): {
					PodSpec: &v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "container-c",
							},
						},
					},
				},
			},
			expected: PodResourceInfoMap{
				types.UID("pod"): {
					PodSpec: &v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "container-c",
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var cloned PodResourceInfoMap
			require.NotPanics(t, func() {
				cloned = test.original.Clone()
			})

			assert.NotNil(t, cloned)
			diff := cmp.Diff(test.expected, cloned, cmp.Comparer(func(x, y resource.Quantity) bool {
				return x.Equal(y)
			}))
			if diff != "" {
				t.Errorf("PodResourceInfoMap mismatch (-want +got):\n%s", diff)
			}
		})
	}

}

func TestPodResourceInfoMap_Clone_DeepCopyIsolation(t *testing.T) {
	newBaseMap := func() PodResourceInfoMap {
		return PodResourceInfoMap{
			types.UID("pod"): {
				PodSpec: &v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "container-a",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: resource.MustParse("256Mi"),
								},
							},
						},
					},
					Resources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("256Mi"),
						},
					},
					Volumes: []v1.Volume{
						{
							Name: "vol-y",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{
									SizeLimit: resource.NewQuantity(1024*1024*100, resource.BinarySI),
								},
							},
						},
					},
				},
			},
		}
	}

	tests := []struct {
		name   string
		mutate func(cloned PodResourceInfoMap)
	}{
		{
			name: "modifying a resource quantity in cloned ContainerResources",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodSpec.Containers[0].Resources.Requests[v1.ResourceMemory] = resource.MustParse("512Mi")
			},
		},
		{
			name: "adding a new container to cloned Containers",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodSpec.Containers = append(pod.PodSpec.Containers, v1.Container{
					Name: "container-new",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
					},
				})
			},
		},
		{
			name: "modifying a resource quantity in cloned Pod Resources",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodSpec.Resources.Requests[v1.ResourceMemory] = resource.MustParse("512Mi")
			},
		},
		{
			name: "modifying a dynamic limit in cloned EmptyDirVolumeLimits",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodSpec.Volumes[0].EmptyDir.SizeLimit.Set(1024 * 1024 * 500)
			},
		},
		{
			name: "adding a new volume key to cloned Volumes",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodSpec.Volumes = append(pod.PodSpec.Volumes, v1.Volume{
					Name: "vol-new",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: resource.NewQuantity(1024, resource.DecimalSI),
						},
					},
				})
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			original := newBaseMap()
			cloned := original.Clone()

			// Perform the mutation on the clone and verify the original is unaffected.
			test.mutate(cloned)

			diff := cmp.Diff(newBaseMap(), original, cmp.Comparer(func(x, y resource.Quantity) bool {
				return x.Equal(y)
			}))
			if diff != "" {
				t.Errorf("original PodResourceInfoMap changed (-want +got):\n%s", diff)
			}
		})
	}
}
