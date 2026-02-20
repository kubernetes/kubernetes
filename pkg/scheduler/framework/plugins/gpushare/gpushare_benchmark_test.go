/*
Copyright 2024 The Kubernetes Authors.

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

package gpushare

import (
	"context"
	"fmt"
	"sync"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// setupScaleNode creates a node with a given capacity and a specified number of existing pods
// each consuming a fractional amount of GPU.
func setupScaleNode(capacity int64, numPods int, podRequest int64) (*v1.Node, []*v1.Pod) {
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "scale-node"},
		Status: v1.NodeStatus{
			Allocatable: v1.ResourceList{
				GPUShareResourceName: resource.MustParse(fmt.Sprintf("%d", capacity)),
			},
		},
	}

	var pods []*v1.Pod
	for i := 0; i < numPods; i++ {
		pods = append(pods, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i)},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								GPUShareResourceName: resource.MustParse(fmt.Sprintf("%d", podRequest)),
							},
						},
					},
				},
			},
		})
	}

	return node, pods
}

func BenchmarkGPUShareFilter(b *testing.B) {
	p := &GPUShare{}

	// Test scaling from 10 to 1000 pods on a single node
	for _, numPods := range []int{10, 100, 500, 1000} {
		b.Run(fmt.Sprintf("Pods-%d", numPods), func(b *testing.B) {
			node, existingPods := setupScaleNode(100000, numPods, 10) // 100,000 capacity, each pod uses 10

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "incoming-pod"},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									GPUShareResourceName: resource.MustParse("50"),
								},
							},
						},
					},
				},
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				nodeInfo := framework.NewNodeInfo(existingPods...)
				nodeInfo.SetNode(node)
				status := p.Filter(context.Background(), nil, pod, nodeInfo)
				if status != nil && status.Code() != fwk.Success {
					b.Fatalf("expected success, got %v", status)
				}
			}
		})
	}
}

func TestGPUShareConcurrency(t *testing.T) {
	// Simulate multiple scheduler threads calling Filter and Score concurrently
	// to test for any race conditions in the plugin. Notice that the kubernetes
	// scheduler framework manages state, but we want to ensure our plugin is statless
	// and thread-safe.

	p := &GPUShare{}
	node, existingPods := setupScaleNode(100000, 500, 10)
	nodeInfo := framework.NewNodeInfo(existingPods...)
	nodeInfo.SetNode(node)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "incoming-pod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							GPUShareResourceName: resource.MustParse("50"),
						},
					},
				},
			},
		},
	}

	var wg sync.WaitGroup
	numGoroutines := 100
	wg.Add(numGoroutines * 2) // 100 for Filter, 100 for Score

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			status := p.Filter(context.Background(), nil, pod, nodeInfo)
			if status != nil && status.Code() != fwk.Success {
				t.Errorf("Concurrent Filter failed: %v", status)
			}
		}()

		go func() {
			defer wg.Done()
			_, status := p.Score(context.Background(), nil, pod, nodeInfo)
			if status != nil && status.Code() != fwk.Success {
				t.Errorf("Concurrent Score failed: %v", status)
			}
		}()
	}

	wg.Wait()
}
