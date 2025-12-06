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

package lifecycle

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
)

// BenchmarkAdmitWithCacheHit measures admission performance when NodeInfo cache hits
func BenchmarkAdmitWithCacheHit(b *testing.B) {
	benchmarks := []struct {
		name     string
		podCount int
	}{
		{"10pods", 10},
		{"50pods", 50},
		{"100pods", 100},
		{"250pods", 250},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Setup
			node := makeBenchNode("test-node", 10000, 100000000000)
			pods := makeBenchPods(bm.podCount, 100, 1000000000)
			testPod := makeBenchPod("test-pod", 100, 1000000000)

			handler := NewPredicateAdmitHandler(
				func(ctx context.Context, useCache bool) (*v1.Node, error) { return node, nil },
				nil,
				func(_ *schedulerframework.NodeInfo, attrs *PodAdmitAttributes) error { return nil },
			)

			attrs := &PodAdmitAttributes{
				Pod:       testPod,
				OtherPods: pods,
			}

			// Warm up the cache
			handler.Admit(attrs)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Cache should hit every time since nothing changes
				result := handler.Admit(attrs)
				if !result.Admit {
					b.Fatalf("admission should succeed")
				}
			}
		})
	}
}

// BenchmarkAdmitWithCacheMiss measures admission performance when NodeInfo must be reconstructed
func BenchmarkAdmitWithCacheMiss(b *testing.B) {
	benchmarks := []struct {
		name     string
		podCount int
	}{
		{"10pods", 10},
		{"50pods", 50},
		{"100pods", 100},
		{"250pods", 250},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Setup
			node := makeBenchNode("test-node", 10000, 100000000000)
			testPod := makeBenchPod("test-pod", 100, 1000000000)

			handler := NewPredicateAdmitHandler(
				func(ctx context.Context, useCache bool) (*v1.Node, error) { return node, nil },
				nil,
				func(_ *schedulerframework.NodeInfo, attrs *PodAdmitAttributes) error { return nil },
			)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Generate new pods each iteration to force cache miss
				pods := makeBenchPodsWithGeneration(bm.podCount, 100, 1000000000, int64(i))
				attrs := &PodAdmitAttributes{
					Pod:       testPod,
					OtherPods: pods,
				}

				result := handler.Admit(attrs)
				if !result.Admit {
					b.Fatalf("admission should succeed")
				}
			}
		})
	}
}

// BenchmarkAdmitNoCache measures admission performance without any caching (theoretical baseline)
// This helps quantify the benefit of the cache by showing worst-case scenario
func BenchmarkAdmitNoCacheBaseline(b *testing.B) {
	benchmarks := []struct {
		name     string
		podCount int
	}{
		{"10pods", 10},
		{"50pods", 50},
		{"100pods", 100},
		{"250pods", 250},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			// Setup
			node := makeBenchNode("test-node", 10000, 100000000000)
			pods := makeBenchPods(bm.podCount, 100, 1000000000)
			testPod := makeBenchPod("test-pod", 100, 1000000000)

			handler := NewPredicateAdmitHandler(
				func(ctx context.Context, useCache bool) (*v1.Node, error) { return node, nil },
				nil,
				func(_ *schedulerframework.NodeInfo, attrs *PodAdmitAttributes) error { return nil },
			)

			attrs := &PodAdmitAttributes{
				Pod:       testPod,
				OtherPods: pods,
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Invalidate cache to simulate no caching
				handler.(*predicateAdmitHandler).cache.invalidate()
				result := handler.Admit(attrs)
				if !result.Admit {
					b.Fatalf("admission should succeed")
				}
			}
		})
	}
}

// Helper functions to create test objects for benchmarks
func makeBenchNode(name string, milliCPU int64, memory int64) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: "1",
		},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
			},
			Capacity: v1.ResourceList{
				v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
			},
		},
	}
}

func makeBenchPod(name string, milliCPU int64, memory int64) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:       name,
			UID:        types.UID(name),
			Generation: 1,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "container1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
							v1.ResourceMemory: *resource.NewQuantity(memory, resource.BinarySI),
						},
					},
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

func makeBenchPods(count int, milliCPU int64, memory int64) []*v1.Pod {
	pods := make([]*v1.Pod, count)
	for i := 0; i < count; i++ {
		pods[i] = makeBenchPod(fmt.Sprintf("pod-%d", i), milliCPU, memory)
	}
	return pods
}

func makeBenchPodsWithGeneration(count int, milliCPU int64, memory int64, generation int64) []*v1.Pod {
	pods := make([]*v1.Pod, count)
	for i := 0; i < count; i++ {
		pod := makeBenchPod(fmt.Sprintf("pod-%d-%d", i, generation), milliCPU, memory)
		pod.Generation = generation
		pods[i] = pod
	}
	return pods
}
