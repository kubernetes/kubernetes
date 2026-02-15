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

package nodeinfocache

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// generatePods creates n pods with realistic resource requests.
// Each pod has 2 containers to simulate typical workloads.
func generatePods(count int) []*v1.Pod {
	pods := make([]*v1.Pod, count)
	for i := range count {
		pods[i] = &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod-%d", i),
				Namespace: "default",
				UID:       types.UID(fmt.Sprintf("uid-%d", i)),
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name: "main",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:              resource.MustParse("100m"),
								v1.ResourceMemory:           resource.MustParse("256Mi"),
								v1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "sidecar",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("50m"),
								v1.ResourceMemory: resource.MustParse("64Mi"),
							},
						},
					},
				},
			},
		}
	}
	return pods
}

// generatePodsWithPrefix creates pods with a custom name prefix.
func generatePodsWithPrefix(prefix string, count int) []*v1.Pod {
	pods := generatePods(count)
	for i, pod := range pods {
		pod.Name = fmt.Sprintf("%s-%d", prefix, i)
		pod.UID = types.UID(fmt.Sprintf("%s-uid-%d", prefix, i))
	}
	return pods
}

// makeNode creates a node with specified resources.
func makeNode(name, cpu, memory string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse(cpu),
				v1.ResourceMemory: resource.MustParse(memory),
				v1.ResourcePods:   resource.MustParse("110"),
			},
		},
	}
}

// BenchmarkNewNodeInfo measures the baseline: rebuilding NodeInfo from scratch.
// This is the OLD approach used before caching.
func BenchmarkNewNodeInfo(b *testing.B) {
	pods := generatePods(100)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = framework.NewNodeInfo(pods...)
	}
}

// BenchmarkCacheSnapshot measures the NEW approach: getting a snapshot from cache.
func BenchmarkCacheSnapshot(b *testing.B) {
	cache := New()
	for _, pod := range generatePods(100) {
		cache.AddPod(pod)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		_ = cache.Snapshot()
	}
}

// BenchmarkCacheAddPod measures the cost of adding a pod to the cache.
func BenchmarkCacheAddPod(b *testing.B) {
	pods := generatePods(b.N)
	cache := New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.AddPod(pods[i])
	}
}

// BenchmarkCacheRemovePod measures the cost of removing a pod from the cache.
func BenchmarkCacheRemovePod(b *testing.B) {
	logger, _ := ktesting.NewTestContext(b)
	pods := generatePods(b.N)
	cache := New()
	for _, pod := range pods {
		cache.AddPod(pod)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.RemovePod(logger, pods[i])
	}
}

// BenchmarkAdmissionE2E compares the full admission path: NodeInfo construction + SetNode.
// This simulates a single pod admission.
func BenchmarkAdmissionE2E(b *testing.B) {
	existingPods := generatePods(100)
	node := makeNode("test-node", "8", "32Gi")

	b.Run("Current-NewNodeInfo", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// Current approach: rebuild NodeInfo every admission
			nodeInfo := framework.NewNodeInfo(existingPods...)
			nodeInfo.SetNode(node)
		}
	})

	b.Run("Cached-Snapshot", func(b *testing.B) {
		// Setup cache once
		cache := New()
		cache.SetNode(node)
		for _, pod := range existingPods {
			cache.AddPod(pod)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// New approach: snapshot from pre-built cache
			nodeInfo := cache.Snapshot()
			nodeInfo.SetNode(node)
		}
	})
}

// BenchmarkResizeRetryScenario simulates pending resize retries.
// This is where caching provides the most benefit - the same pod set
// is evaluated repeatedly while waiting for resources to become available.
func BenchmarkResizeRetryScenario(b *testing.B) {
	existingPods := generatePods(100)
	node := makeNode("test-node", "8", "32Gi")
	retryCount := 10 // Typical retries before resources free up

	b.Run("Current-RebuildEachRetry", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			// Current: rebuild NodeInfo on every retry
			for range retryCount {
				nodeInfo := framework.NewNodeInfo(existingPods...)
				nodeInfo.SetNode(node)
			}
		}
	})

	b.Run("Cached-SnapshotEachRetry", func(b *testing.B) {
		cache := New()
		cache.SetNode(node)
		for _, pod := range existingPods {
			cache.AddPod(pod)
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// New: snapshot from cache on each retry
			for range retryCount {
				nodeInfo := cache.Snapshot()
				nodeInfo.SetNode(node)
			}
		}
	})
}

// BenchmarkSequentialAdmissions simulates burst pod creation.
// Tests the incremental update benefit when pods are admitted one after another.
func BenchmarkSequentialAdmissions(b *testing.B) {
	node := makeNode("test-node", "8", "32Gi")
	basePodCount := 50
	burstSize := 20

	b.Run("Current-RebuildGrowingSet", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			pods := generatePods(basePodCount)
			newPods := generatePodsWithPrefix("burst", burstSize)
			// Admit burst pods sequentially
			for _, newPod := range newPods {
				nodeInfo := framework.NewNodeInfo(pods...)
				nodeInfo.SetNode(node)
				// After admission, pod joins the set
				pods = append(pods, newPod)
			}
		}
	})

	b.Run("Cached-IncrementalAdd", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			cache := New()
			cache.SetNode(node)
			for _, pod := range generatePods(basePodCount) {
				cache.AddPod(pod)
			}
			newPods := generatePodsWithPrefix("burst", burstSize)
			// Admit burst pods sequentially
			for _, newPod := range newPods {
				_ = cache.Snapshot()
				// After admission, add to cache (O(1))
				cache.AddPod(newPod)
			}
		}
	})
}

// BenchmarkScalingComparison tests performance across different node sizes.
func BenchmarkScalingComparison(b *testing.B) {
	node := makeNode("test-node", "96", "384Gi") // Large node

	for _, podCount := range []int{10, 50, 100, 200} {
		pods := generatePods(podCount)

		b.Run(fmt.Sprintf("NewNodeInfo-%dPods", podCount), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				nodeInfo := framework.NewNodeInfo(pods...)
				nodeInfo.SetNode(node)
			}
		})

		b.Run(fmt.Sprintf("CacheSnapshot-%dPods", podCount), func(b *testing.B) {
			cache := New()
			cache.SetNode(node)
			for _, pod := range pods {
				cache.AddPod(pod)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nodeInfo := cache.Snapshot()
				nodeInfo.SetNode(node)
			}
		})
	}
}

// BenchmarkConcurrentSnapshots tests snapshot performance under concurrent access.
func BenchmarkConcurrentSnapshots(b *testing.B) {
	cache := New()
	cache.SetNode(makeNode("test-node", "8", "32Gi"))
	for _, pod := range generatePods(100) {
		cache.AddPod(pod)
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = cache.Snapshot()
		}
	})
}
