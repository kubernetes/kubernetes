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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestNodeStatusCapacityMutationAccepted verifies that the API server accepts
// live mutations to Node.Status.Capacity and Node.Status.Allocatable via
// UpdateStatus, and that the new values are durable — a subsequent Get returns
// exactly what was written.
//
// This is the API-layer contract that KEP-3953 (In-Place Node Resource Resize)
// depends on: the control plane must not reject or silently discard changes to
// node capacity that a hardware-resize agent writes after boot.
func TestNodeStatusCapacityMutationAccepted(t *testing.T) {
	testCtx := testutils.InitTestAPIServer(t, "node-status-mutation", nil)
	cs := testCtx.ClientSet
	t.Cleanup(func() { testutils.CleanupNodes(cs, t) })

	tests := []struct {
		name    string
		initial v1.ResourceList
		updated v1.ResourceList
	}{
		{
			name: "upscale: memory and CPU increase is persisted",
			initial: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("8Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
			updated: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("8"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
		},
		{
			name: "downscale: memory and CPU decrease is persisted",
			initial: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("8"),
				v1.ResourceMemory: resource.MustParse("16Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
			updated: v1.ResourceList{
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("8Gi"),
				v1.ResourcePods:   resource.MustParse("110"),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					GenerateName: "resize-node-",
				},
				Status: v1.NodeStatus{
					Capacity:    tc.initial,
					Allocatable: tc.initial,
					Conditions: []v1.NodeCondition{
						{Type: v1.NodeReady, Status: v1.ConditionTrue},
					},
				},
			}
			created, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create node: %v", err)
			}

			created.Status.Capacity = tc.updated
			created.Status.Allocatable = tc.updated
			if _, err := cs.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, created, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("UpdateStatus rejected the capacity mutation: %v", err)
			}

			got, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, created.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get node after update: %v", err)
			}
			for _, res := range []v1.ResourceName{v1.ResourceCPU, v1.ResourceMemory} {
				wantCap := tc.updated[res]
				wantAlloc := tc.updated[res]
				gotCap := got.Status.Capacity[res]
				gotAlloc := got.Status.Allocatable[res]
				if wantCap.Cmp(gotCap) != 0 {
					t.Errorf("Capacity[%s]: want %s, got %s", res, wantCap.String(), gotCap.String())
				}
				if wantAlloc.Cmp(gotAlloc) != 0 {
					t.Errorf("Allocatable[%s]: want %s, got %s", res, wantAlloc.String(), gotAlloc.String())
				}
			}
		})
	}
}
