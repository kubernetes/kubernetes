/*
Copyright 2016 The Kubernetes Authors.

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

package eviction

import (
	"context"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	statsapi "k8s.io/kubernetes/pkg/apis/stats/v1alpha1"
)

// TestEvictionMemoryPressure verifies that eviction occurs when memory pressure is high
func TestEvictionMemoryPressure(t *testing.T) {
	// Simulate memory pressure condition
	memoryStats := &statsapi.MemoryStats{
		WorkingSetBytes: uint64(5 * 1024 * 1024 * 1024), // 5GB
	}
	summaryProvider := mockSummaryProvider(memoryStats, nil)
	config := Config{MemoryEvictionThreshold: 3 * 1024 * 1024 * 1024} // 3GB eviction threshold

	manager := NewManager(summaryProvider, config, mockKillPodFunc, nil, nil, nil, nil, nil, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go manager.StartManager(ctx)

	time.Sleep(2 * time.Second) // Let the eviction manager run

	// Check if eviction happened due to memory pressure
	if len(manager.thresholdsMet) == 0 {
		t.Fatalf("Expected eviction due to memory pressure, but none occurred")
	}
}

// TestEvictionDiskPressure verifies that eviction occurs when disk pressure is high
func TestEvictionDiskPressure(t *testing.T) {
	// Simulate disk pressure condition
	diskStats := &statsapi.FsStats{
		AvailableBytes: uint64(1 * 1024 * 1024 * 1024), // 1GB
	}
	summaryProvider := mockSummaryProvider(nil, diskStats)
	config := Config{DiskEvictionThreshold: 2 * 1024 * 1024 * 1024} // 2GB eviction threshold

	manager := NewManager(summaryProvider, config, mockKillPodFunc, nil, nil, nil, nil, nil, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go manager.StartManager(ctx)

	time.Sleep(2 * time.Second) // Let the eviction manager run

	// Check if eviction happened due to disk pressure
	if len(manager.thresholdsMet) == 0 {
		t.Fatalf("Expected eviction due to disk pressure, but none occurred")
	}
}

// mockSummaryProvider simulates memory and disk stats for testing
func mockSummaryProvider(memoryStats *statsapi.MemoryStats, diskStats *statsapi.FsStats) stats.SummaryProvider {
	return &mockSummaryProviderImpl{
		memoryStats: memoryStats,
		diskStats:   diskStats,
	}
}

type mockSummaryProviderImpl struct {
	memoryStats *statsapi.MemoryStats
	diskStats   *statsapi.FsStats
}

func (m *mockSummaryProviderImpl) GetStats() *statsapi.Summary {
	return &statsapi.Summary{
		Node: statsapi.NodeStats{
			Memory: m.memoryStats,
			Fs:     m.diskStats,
		},
	}
}

// mockKillPodFunc simulates killing a pod during eviction
func mockKillPodFunc(pod *v1.Pod, isEviction bool, gracePeriodOverride *int64, statusFn lifecycle.PodStatusFn) error {
	// Mock pod kill logic
	return nil
}
