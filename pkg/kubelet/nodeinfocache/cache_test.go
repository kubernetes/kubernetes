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
)

func TestCacheAddRemovePod(t *testing.T) {
	cache := New()
	logger, _ := ktesting.NewTestContext(t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-pod", Namespace: "default", UID: types.UID("test-uid"),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("256Mi"),
					},
				},
			}},
		},
	}

	// Add pod
	cache.AddPod(pod)
	if cache.PodCount() != 1 {
		t.Errorf("expected 1 pod, got %d", cache.PodCount())
	}

	// Verify resources in snapshot
	snapshot := cache.Snapshot()
	if snapshot.Requested.MilliCPU != 100 {
		t.Errorf("expected 100 milliCPU, got %d", snapshot.Requested.MilliCPU)
	}

	// Remove pod
	cache.RemovePod(logger, pod)
	if cache.PodCount() != 0 {
		t.Errorf("expected 0 pods, got %d", cache.PodCount())
	}
}

func TestCacheSnapshotIsolation(t *testing.T) {
	cache := New()

	pod1 := makePod("pod1", "100m", "256Mi")
	cache.AddPod(pod1)

	// Take snapshot
	snapshot := cache.Snapshot()
	originalCPU := snapshot.Requested.MilliCPU

	// Add another pod to cache
	pod2 := makePod("pod2", "200m", "512Mi")
	cache.AddPod(pod2)

	// Snapshot should be unchanged (deep copy)
	if snapshot.Requested.MilliCPU != originalCPU {
		t.Error("snapshot was mutated after cache modification")
	}
}

func TestCacheConcurrentAccess(t *testing.T) {
	cache := New()
	logger, _ := ktesting.NewTestContext(t)

	done := make(chan bool)

	// Writer goroutine
	go func() {
		for i := range 100 {
			pod := makePod(fmt.Sprintf("pod-%d", i), "10m", "10Mi")
			cache.AddPod(pod)
			cache.RemovePod(logger, pod)
		}
		done <- true
	}()

	// Reader goroutine
	go func() {
		for range 100 {
			_ = cache.Snapshot()
			_ = cache.PodCount()
		}
		done <- true
	}()

	<-done
	<-done
}

func TestCacheSetNode(t *testing.T) {
	cache := New()

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
			},
		},
	}

	cache.SetNode(node)

	snapshot := cache.Snapshot()
	if snapshot.Node() == nil {
		t.Error("expected node to be set")
	}
	if snapshot.Node().Name != "test-node" {
		t.Errorf("expected node name 'test-node', got '%s'", snapshot.Node().Name)
	}
}

func TestCacheUpdatePod(t *testing.T) {
	cache := New()
	logger, _ := ktesting.NewTestContext(t)

	oldPod := makePod("test-pod", "100m", "256Mi")
	cache.AddPod(oldPod)

	// Verify initial state
	snapshot := cache.Snapshot()
	if snapshot.Requested.MilliCPU != 100 {
		t.Errorf("expected 100 milliCPU, got %d", snapshot.Requested.MilliCPU)
	}

	// Update pod with new resources
	newPod := makePod("test-pod", "200m", "512Mi")
	cache.UpdatePod(logger, oldPod, newPod)

	// Verify updated state
	snapshot = cache.Snapshot()
	if snapshot.Requested.MilliCPU != 200 {
		t.Errorf("expected 200 milliCPU after update, got %d", snapshot.Requested.MilliCPU)
	}
	if cache.PodCount() != 1 {
		t.Errorf("expected 1 pod after update, got %d", cache.PodCount())
	}
}

func makePod(name, cpu, memory string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: "default", UID: types.UID(name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse(cpu),
						v1.ResourceMemory: resource.MustParse(memory),
					},
				},
			}},
		},
	}
}
