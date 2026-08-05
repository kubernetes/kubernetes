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

package stats

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/utils/ptr"
)

func TestAdjustForHugePages(t *testing.T) {
	tests := []struct {
		name              string
		memory            *statsapi.MemoryStats
		node              *v1.Node
		expectedAvailable *uint64
	}{
		{
			name:              "nil memory stats",
			memory:            nil,
			node:              &v1.Node{},
			expectedAvailable: nil,
		},
		{
			name: "nil AvailableBytes",
			memory: &statsapi.MemoryStats{
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			node:              &v1.Node{},
			expectedAvailable: nil,
		},
		{
			name: "nil node",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10000),
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			node:              nil,
			expectedAvailable: ptr.To[uint64](10000),
		},
		{
			name: "no hugepages in capacity",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10000),
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			node: &v1.Node{
				Status: v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourceMemory: *resource.NewQuantity(20000, resource.BinarySI),
					},
				},
			},
			expectedAvailable: ptr.To[uint64](10000),
		},
		{
			name: "hugepages subtracted from available",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10 * 1024 * 1024 * 1024), // 10Gi
				WorkingSetBytes: ptr.To[uint64](50 * 1024 * 1024 * 1024), // 50Gi
			},
			node: &v1.Node{
				Status: v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourceMemory:                *resource.NewQuantity(64*1024*1024*1024, resource.BinarySI),
						v1.ResourceName("hugepages-1Gi"): *resource.NewQuantity(4*1024*1024*1024, resource.BinarySI),
					},
				},
			},
			expectedAvailable: ptr.To[uint64](6 * 1024 * 1024 * 1024), // 10Gi - 4Gi = 6Gi
		},
		{
			name: "multiple hugepage sizes",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10 * 1024 * 1024 * 1024),
				WorkingSetBytes: ptr.To[uint64](50 * 1024 * 1024 * 1024),
			},
			node: &v1.Node{
				Status: v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourceMemory:                *resource.NewQuantity(64*1024*1024*1024, resource.BinarySI),
						v1.ResourceName("hugepages-1Gi"): *resource.NewQuantity(2*1024*1024*1024, resource.BinarySI),
						v1.ResourceName("hugepages-2Mi"): *resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
					},
				},
			},
			expectedAvailable: ptr.To[uint64](7 * 1024 * 1024 * 1024), // 10Gi - 3Gi = 7Gi
		},
		{
			name: "hugepages exceed available clamps to zero",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](2 * 1024 * 1024 * 1024), // 2Gi
				WorkingSetBytes: ptr.To[uint64](60 * 1024 * 1024 * 1024),
			},
			node: &v1.Node{
				Status: v1.NodeStatus{
					Capacity: v1.ResourceList{
						v1.ResourceMemory:                *resource.NewQuantity(64*1024*1024*1024, resource.BinarySI),
						v1.ResourceName("hugepages-1Gi"): *resource.NewQuantity(4*1024*1024*1024, resource.BinarySI),
					},
				},
			},
			expectedAvailable: ptr.To[uint64](0),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := adjustForHugePages(tc.memory, tc.node)
			if tc.expectedAvailable == nil {
				if result != nil && result.AvailableBytes != nil {
					if tc.memory == nil || tc.memory.AvailableBytes == nil {
						t.Errorf("expected nil AvailableBytes, got %d", *result.AvailableBytes)
					}
				}
				return
			}
			if result == nil || result.AvailableBytes == nil {
				t.Fatalf("expected AvailableBytes=%d, got nil", *tc.expectedAvailable)
			}
			if *result.AvailableBytes != *tc.expectedAvailable {
				t.Errorf("expected AvailableBytes=%d, got %d", *tc.expectedAvailable, *result.AvailableBytes)
			}
		})
	}
}

func TestAdjustForHugePagesDoesNotMutateOriginal(t *testing.T) {
	original := &statsapi.MemoryStats{
		AvailableBytes:  ptr.To[uint64](10 * 1024 * 1024 * 1024),
		WorkingSetBytes: ptr.To[uint64](50 * 1024 * 1024 * 1024),
	}
	originalAvailable := *original.AvailableBytes

	node := &v1.Node{
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourceName("hugepages-1Gi"): *resource.NewQuantity(4*1024*1024*1024, resource.BinarySI),
			},
		},
	}

	result := adjustForHugePages(original, node)

	if *original.AvailableBytes != originalAvailable {
		t.Errorf("original AvailableBytes was mutated: expected %d, got %d", originalAvailable, *original.AvailableBytes)
	}
	if result == original {
		t.Error("adjustForHugePages should return a new copy, not the original pointer")
	}
}
