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

	cadvisorapi "github.com/google/cadvisor/lib/model"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	statstest "k8s.io/kubernetes/pkg/kubelet/server/stats/testing"
	"k8s.io/utils/ptr"
)

// cadvisor reports hugepage sizes in KiB.
const (
	hugePageSize2Mi = 2 * 1024
	hugePageSize1Gi = 1024 * 1024
)

func TestAdjustForHugePages(t *testing.T) {
	tests := []struct {
		name              string
		memory            *statsapi.MemoryStats
		machineInfo       *cadvisorapi.MachineInfo
		expectedAvailable *uint64
	}{
		{
			name:              "nil memory stats",
			memory:            nil,
			machineInfo:       &cadvisorapi.MachineInfo{},
			expectedAvailable: nil,
		},
		{
			name: "nil AvailableBytes",
			memory: &statsapi.MemoryStats{
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			machineInfo:       &cadvisorapi.MachineInfo{},
			expectedAvailable: nil,
		},
		{
			name: "nil machine info",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10000),
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			machineInfo:       nil,
			expectedAvailable: ptr.To[uint64](10000),
		},
		{
			name: "no hugepages reserved",
			memory: &statsapi.MemoryStats{
				AvailableBytes:  ptr.To[uint64](10000),
				WorkingSetBytes: ptr.To[uint64](1000),
			},
			machineInfo: &cadvisorapi.MachineInfo{
				HugePages: []cadvisorapi.HugePagesInfo{
					{PageSize: hugePageSize1Gi, NumPages: 0},
					{PageSize: hugePageSize2Mi, NumPages: 0},
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
			machineInfo: &cadvisorapi.MachineInfo{
				HugePages: []cadvisorapi.HugePagesInfo{
					{PageSize: hugePageSize1Gi, NumPages: 4},
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
			machineInfo: &cadvisorapi.MachineInfo{
				HugePages: []cadvisorapi.HugePagesInfo{
					{PageSize: hugePageSize1Gi, NumPages: 2},
					{PageSize: hugePageSize2Mi, NumPages: 512},
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
			machineInfo: &cadvisorapi.MachineInfo{
				HugePages: []cadvisorapi.HugePagesInfo{
					{PageSize: hugePageSize1Gi, NumPages: 4},
				},
			},
			expectedAvailable: ptr.To[uint64](0),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := adjustForHugePages(tc.memory, tc.machineInfo)
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

	machineInfo := &cadvisorapi.MachineInfo{
		HugePages: []cadvisorapi.HugePagesInfo{
			{PageSize: hugePageSize1Gi, NumPages: 4},
		},
	}

	result := adjustForHugePages(original, machineInfo)

	if *original.AvailableBytes != originalAvailable {
		t.Errorf("original AvailableBytes was mutated: expected %d, got %d", originalAvailable, *original.AvailableBytes)
	}
	if result == original {
		t.Error("adjustForHugePages should return a new copy, not the original pointer")
	}
}

// After a kubelet restart the node object still has the previous kubelet's
// hugepage capacity until the first node status update. The summary must
// follow the machine info, not that stale capacity.
func TestSummaryProviderHugePagesIgnoreStaleNodeCapacity(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.HugepageAwareEviction, true)

	const (
		rootAvailable = 1024 * 1024 * 1024 // 1Gi
		hugePages64Mi = 64 * 1024 * 1024
	)
	nodeWithHugePages := func(bytes int64) *v1.Node {
		return &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceName("hugepages-2Mi"): *resource.NewQuantity(bytes, resource.BinarySI),
				},
			},
		}
	}
	machineWith2MiPages := func(numPages uint64) *cadvisorapi.MachineInfo {
		return &cadvisorapi.MachineInfo{
			HugePages: []cadvisorapi.HugePagesInfo{
				{PageSize: hugePageSize2Mi, NumPages: numPages},
			},
		}
	}

	tests := []struct {
		name              string
		node              *v1.Node
		machineInfo       *cadvisorapi.MachineInfo
		expectedAvailable uint64
	}{
		{
			name:              "hugepages added since the last status update",
			node:              nodeWithHugePages(0),
			machineInfo:       machineWith2MiPages(32),
			expectedAvailable: rootAvailable - hugePages64Mi,
		},
		{
			name:              "hugepages removed since the last status update",
			node:              nodeWithHugePages(hugePages64Mi),
			machineInfo:       machineWith2MiPages(0),
			expectedAvailable: rootAvailable,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := t.Context()
			rootStats := &statsapi.ContainerStats{
				Memory: &statsapi.MemoryStats{
					AvailableBytes:  ptr.To[uint64](rootAvailable),
					WorkingSetBytes: ptr.To[uint64](rootAvailable),
				},
			}

			mockStatsProvider := statstest.NewMockProvider(t)
			mockStatsProvider.EXPECT().GetNode(ctx).Return(tc.node, nil)
			mockStatsProvider.EXPECT().GetCachedMachineInfo().Return(tc.machineInfo, nil)
			// Empty cgroup names skip system container stats.
			mockStatsProvider.EXPECT().GetNodeConfig().Return(cm.NodeConfig{})
			mockStatsProvider.EXPECT().GetPodCgroupRoot().Return("")
			mockStatsProvider.EXPECT().GetCgroupStats(ctx, "/", true).Return(rootStats, nil, nil).Maybe()
			mockStatsProvider.EXPECT().GetCgroupCPUAndMemoryStats("/", false).Return(rootStats, nil).Maybe()
			mockStatsProvider.EXPECT().RootFsStats().Return(nil, nil).Maybe()
			mockStatsProvider.EXPECT().ImageFsStats(ctx).Return(nil, nil, nil).Maybe()
			mockStatsProvider.EXPECT().ListPodStatsAndUpdateCPUNanoCoreUsage(ctx).Return(nil, nil).Maybe()
			mockStatsProvider.EXPECT().ListPodCPUAndMemoryStats(ctx).Return(nil, nil).Maybe()
			mockStatsProvider.EXPECT().RlimitStats().Return(nil, nil).Maybe()

			provider := NewSummaryProvider(ctx, mockStatsProvider)
			for method, get := range map[string]func() (*statsapi.Summary, error){
				"Get":                  func() (*statsapi.Summary, error) { return provider.Get(ctx, true) },
				"GetCPUAndMemoryStats": func() (*statsapi.Summary, error) { return provider.GetCPUAndMemoryStats(ctx) },
			} {
				summary, err := get()
				if err != nil {
					t.Fatalf("%s: unexpected error: %v", method, err)
				}
				if got := *summary.Node.Memory.AvailableBytes; got != tc.expectedAvailable {
					t.Errorf("%s: expected AvailableBytes=%d, got %d", method, tc.expectedAvailable, got)
				}
			}
		})
	}
}
