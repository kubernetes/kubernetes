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

package podresize

import (
	"encoding/json"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMakeResizePatch(t *testing.T) {
	tests := []struct {
		name string
		old  []ResizableContainerInfo
		new  []ResizableContainerInfo
		want string
	}{
		{
			name: "no change",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi"}},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi"}},
			},
			want: `{}`,
		},
		{
			name: "increase cpu",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", CPULim: "15m", MemReq: "10Mi"}},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "20m", CPULim: "25m", MemReq: "10Mi"}},
			},
			want: `{"spec":{"containers":[{"name":"c1","resources":{"limits":{"cpu":"25m"},"requests":{"cpu":"20m"}}}]}}`,
		},
		{
			name: "add cpu",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{MemReq: "10Mi"}},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "20m", CPULim: "25m", MemReq: "10Mi"}},
			},
			want: `{"spec":{"containers":[{"name":"c1","resources":{"limits":{"cpu":"25m"},"requests":{"cpu":"20m"}}}]}}`,
		},
		{
			name: "decrease memory",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", CPULim: "15m", MemReq: "20Mi", MemLim: "25Mi"}},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", CPULim: "15m", MemReq: "10Mi", MemLim: "15Mi"}},
			},
			want: `{"spec":{"containers":[{"name":"c1","resources":{"limits":{"memory":"15Mi"},"requests":{"memory":"10Mi"}}}]}}`,
		},
		{
			name: "change multiple contaniers",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi", CPULim: "15m", MemLim: "15Mi"}},
				{Name: "c2", Resources: &cgroups.ContainerResources{CPUReq: "20m", MemReq: "20Mi"}},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "15m", MemReq: "10Mi", CPULim: "25m", MemLim: "25Mi"}},
				{Name: "c2", Resources: &cgroups.ContainerResources{CPUReq: "20m", MemReq: "25Mi"}},
			},
			want: `{"spec":{"containers":[{"name":"c1","resources":{"limits":{"cpu":"25m","memory":"25Mi"},"requests":{"cpu":"15m"}}},{"name":"c2","resources":{"requests":{"memory":"25Mi"}}}]}}`,
		},
		{
			name: "change init container",
			old: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi"}},
				{Name: "ic1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi"}, InitCtr: true},
			},
			new: []ResizableContainerInfo{
				{Name: "c1", Resources: &cgroups.ContainerResources{CPUReq: "10m", MemReq: "10Mi"}},
				{Name: "ic1", Resources: &cgroups.ContainerResources{CPUReq: "20m", MemReq: "10Mi"}, InitCtr: true},
			},
			want: `{"spec":{"initContainers":[{"name":"ic1","resources":{"requests":{"cpu":"20m"}}}]}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := MakeResizePatch(tt.old, tt.new, nil, nil)

			// Ignore the "$setElementOrder" directive for patch comparison.
			patchMap := strategicpatch.JSONMap{}
			err := json.Unmarshal(patch, &patchMap)
			require.NoError(t, err)
			if patchMap, ok := patchMap["spec"].(map[string]interface{}); ok {
				delete(patchMap, "$setElementOrder/containers")
				delete(patchMap, "$setElementOrder/initContainers")
			}
			patch, err = json.Marshal(patchMap)
			require.NoError(t, err)

			assert.Equal(t, tt.want, string(patch))
		})
	}

}

func TestApplyPodLevelLimitsToContainer(t *testing.T) {
	tests := []struct {
		name         string
		container    v1.Container
		podResources *v1.ResourceRequirements
		wantLimits   v1.ResourceList
	}{
		{
			name: "nil podResources leaves container limits unchanged",
			container: v1.Container{
				Name: "c1",
			},
			podResources: nil,
			wantLimits:   nil,
		},
		{
			name: "podResources without limits leaves container limits unchanged",
			container: v1.Container{
				Name: "c1",
			},
			podResources: &v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
			},
			wantLimits: nil,
		},
		{
			name: "container with no limits inherits CPU and memory limits from pod",
			container: v1.Container{
				Name: "c1",
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("200m"),
					v1.ResourceMemory: resource.MustParse("256Mi"),
				},
			},
			wantLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("256Mi"),
			},
		},
		{
			name: "container with explicit CPU limit keeps CPU limit and inherits memory limit from pod",
			container: v1.Container{
				Name: "c1",
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("100m"),
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("200m"),
					v1.ResourceMemory: resource.MustParse("256Mi"),
				},
			},
			wantLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("256Mi"),
			},
		},
		{
			name: "container with explicit memory limit keeps memory limit and inherits CPU limit from pod",
			container: v1.Container{
				Name: "c1",
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("200m"),
					v1.ResourceMemory: resource.MustParse("256Mi"),
				},
			},
			wantLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("200m"),
				v1.ResourceMemory: resource.MustParse("128Mi"),
			},
		},
		{
			name: "container with both limits set preserves its limits",
			container: v1.Container{
				Name: "c1",
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("200m"),
					v1.ResourceMemory: resource.MustParse("256Mi"),
				},
			},
			wantLimits: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("100m"),
				v1.ResourceMemory: resource.MustParse("128Mi"),
			},
		},
		{
			name: "container with no limits inherits only CPU limit when pod has only CPU limit",
			container: v1.Container{
				Name: "c1",
			},
			podResources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("200m"),
				},
			},
			wantLimits: v1.ResourceList{
				v1.ResourceCPU: resource.MustParse("200m"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctr := tt.container
			ApplyPodLevelLimitsToContainer(&ctr, tt.podResources)
			assert.Equal(t, tt.wantLimits, ctr.Resources.Limits)
		})
	}
}
