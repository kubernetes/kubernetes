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
					ContainerResources: map[string]v1.ResourceRequirements{
						"container-a": {
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
					PodLevelResources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("256Mi"),
						},
					},
					EmptyDirVolumeLimits: map[string]*resource.Quantity{
						"vol-x": resource.NewQuantity(2, resource.BinarySI),
					},
				},
			},
			expected: PodResourceInfoMap{
				types.UID("pod"): {
					ContainerResources: map[string]v1.ResourceRequirements{
						"container-a": {
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
					PodLevelResources: &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("100m"),
							v1.ResourceMemory: resource.MustParse("256Mi"),
						},
					},
					EmptyDirVolumeLimits: map[string]*resource.Quantity{
						"vol-x": resource.NewQuantity(2, resource.BinarySI),
					},
				},
			},
		},
		{
			name: "cloning with missing or partially nil fields",
			original: PodResourceInfoMap{
				types.UID("pod"): {
					ContainerResources: map[string]v1.ResourceRequirements{
						"container-c": {},
					},
					PodLevelResources:    nil,
					EmptyDirVolumeLimits: nil,
				},
			},
			expected: PodResourceInfoMap{
				types.UID("pod"): {
					ContainerResources: map[string]v1.ResourceRequirements{
						"container-c": {},
					},
					PodLevelResources:    nil,
					EmptyDirVolumeLimits: nil,
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
				ContainerResources: map[string]v1.ResourceRequirements{
					"container-a": {
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("256Mi"),
						},
					},
				},
				PodLevelResources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("256Mi"),
					},
				},
				EmptyDirVolumeLimits: map[string]*resource.Quantity{
					"vol-y": resource.NewQuantity(1024*1024*100, resource.BinarySI),
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
				pod.ContainerResources["container-a"].Requests[v1.ResourceMemory] = resource.MustParse("512Mi")
			},
		},
		{
			name: "adding a new container to cloned ContainerResources",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.ContainerResources["container-new"] = v1.ResourceRequirements{
					Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")},
				}
			},
		},
		{
			name: "modifying a resource quantity in cloned PodLevelResources",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.PodLevelResources.Requests[v1.ResourceMemory] = resource.MustParse("512Mi")
			},
		},
		{
			name: "modifying a dynamic limit in cloned EmptyDirVolumeLimits",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.EmptyDirVolumeLimits["vol-y"].Set(1024 * 1024 * 500)
			},
		},
		{
			name: "adding a new volume key to cloned EmptyDirVolumeLimits",
			mutate: func(cloned PodResourceInfoMap) {
				pod := cloned[types.UID("pod")]
				pod.EmptyDirVolumeLimits["vol-new"] = resource.NewQuantity(1024, resource.DecimalSI)
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
