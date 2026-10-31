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

package node

import (
	"context"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestNodeStatusCapacityMutation verifies that the API server accepts live
// mutations to Node.Status.Capacity and Node.Status.Allocatable via
// UpdateStatus (PUT) and merge patches on /status, that the new values are
// durable, and that keys not named in a partial patch are left unchanged. It also pins
// the node resource validation that rejects malformed capacity values.
func TestNodeStatusCapacityMutation(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "node-status-mutation", nil)
	cs := testCtx.ClientSet

	const gpu = v1.ResourceName("example.com/gpu")
	baseResources := func() v1.ResourceList {
		return v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("4"),
			v1.ResourceMemory: resource.MustParse("8Gi"),
			v1.ResourcePods:   resource.MustParse("110"),
			gpu:               resource.MustParse("2"),
		}
	}
	memoryPatch := []byte(`{"status":{"capacity":{"memory":"16Gi"},"allocatable":{"memory":"16Gi"}}}`)

	tests := []struct {
		name    string
		mutate  func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error)
		want    v1.ResourceList
		wantErr bool
	}{
		{
			name: "upscale: cpu and memory increase via UpdateStatus is persisted",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				for _, rl := range []v1.ResourceList{node.Status.Capacity, node.Status.Allocatable} {
					rl[v1.ResourceCPU] = resource.MustParse("8")
					rl[v1.ResourceMemory] = resource.MustParse("16Gi")
				}
				return cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
			},
			want: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("8"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
				gpu:               resource.MustParse("2"),
			},
		},
		{
			name: "downscale: cpu and memory decrease via UpdateStatus is persisted",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				for _, rl := range []v1.ResourceList{node.Status.Capacity, node.Status.Allocatable} {
					rl[v1.ResourceCPU] = resource.MustParse("2")
					rl[v1.ResourceMemory] = resource.MustParse("4Gi")
				}
				return cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
			},
			want: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("2"),
				v1.ResourceMemory: resource.MustParse("4Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
				gpu:               resource.MustParse("2"),
			},
		},
		{
			name: "json merge patch: only memory is patched, other keys unchanged",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				return cs.CoreV1().Nodes().Patch(ctx, node.Name, types.MergePatchType, memoryPatch, metav1.PatchOptions{}, "status")
			},
			want: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
				gpu:               resource.MustParse("2"),
			},
		},
		{
			name: "strategic merge patch: only memory is patched, other keys unchanged",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				return cs.CoreV1().Nodes().Patch(ctx, node.Name, types.StrategicMergePatchType, memoryPatch, metav1.PatchOptions{}, "status")
			},
			want: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
				gpu:               resource.MustParse("2"),
			},
		},
		{
			name: "validation: fractional pods is rejected",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				node.Status.Capacity[v1.ResourcePods] = resource.MustParse("1500m")
				return cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
			},
			wantErr: true,
		},
		{
			name: "validation: negative cpu is rejected",
			mutate: func(ctx context.Context, cs clientset.Interface, node *v1.Node) (*v1.Node, error) {
				node.Status.Capacity[v1.ResourceCPU] = resource.MustParse("-1")
				return cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{})
			},
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{GenerateName: "resize-node-"},
				Status: v1.NodeStatus{
					Capacity:    baseResources(),
					Allocatable: baseResources(),
					Conditions: []v1.NodeCondition{
						{Type: v1.NodeReady, Status: v1.ConditionTrue},
					},
				},
			}
			created, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create node: %v", err)
			}

			_, err = tc.mutate(testCtx.Ctx, cs, created)
			if tc.wantErr {
				if err == nil {
					t.Fatal("expected mutation to be rejected by validation, but it succeeded")
				}
				return
			}
			if err != nil {
				t.Fatalf("mutation rejected unexpectedly: %v", err)
			}

			got, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, created.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get node after mutation: %v", err)
			}

			if len(got.Status.Capacity) != len(tc.want) {
				t.Errorf("Capacity has %d keys, want %d: %v", len(got.Status.Capacity), len(tc.want), got.Status.Capacity)
			}
			if len(got.Status.Allocatable) != len(tc.want) {
				t.Errorf("Allocatable has %d keys, want %d: %v", len(got.Status.Allocatable), len(tc.want), got.Status.Allocatable)
			}
			for res, want := range tc.want {
				gotCap, ok := got.Status.Capacity[res]
				if !ok || want.Cmp(gotCap) != 0 {
					t.Errorf("Capacity[%s]: want %s, got %s (present=%v)", res, want.String(), gotCap.String(), ok)
				}
				gotAlloc, ok := got.Status.Allocatable[res]
				if !ok || want.Cmp(gotAlloc) != 0 {
					t.Errorf("Allocatable[%s]: want %s, got %s (present=%v)", res, want.String(), gotAlloc.String(), ok)
				}
			}
		})
	}
}
