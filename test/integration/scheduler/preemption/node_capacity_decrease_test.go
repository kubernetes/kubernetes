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

package preemption

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	configv1 "k8s.io/kube-scheduler/config/v1"
	"k8s.io/kubernetes/pkg/scheduler"
	configtesting "k8s.io/kubernetes/pkg/scheduler/apis/config/testing"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestNominatedPodRequeuesOnNodeAllocatableDecrease verifies that a pod sitting
// in the unschedulable queue with NominatedNodeName set is re-queued for a fresh
// scheduling cycle when the nominated node's allocatable decreases.
//
// This covers the preemption grace-period race: the scheduler has already chosen
// a preemption victim and written NominatedNodeName, but the node capacity drops
// before the victim finishes terminating.  Without the queueing hint the pod
// stays parked until the victim finally terminates and then fails Kubelet
// admission — the correct behaviour is an immediate re-evaluation.
func TestNominatedPodRequeuesOnNodeAllocatableDecrease(t *testing.T) {
	schedulerName := v1.DefaultSchedulerName
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: &schedulerName,
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "node-allocatable-decrease", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	tests := []struct {
		name            string
		expectIncrement bool
		trigger         func(ctx context.Context, t *testing.T, cs clientset.Interface, nominatedNode, otherNode string)
	}{
		{
			name:            "memory decrease on nominated node re-queues waiting pod",
			expectIncrement: true,
			trigger: func(ctx context.Context, t *testing.T, cs clientset.Interface, nominatedNode, _ string) {
				t.Helper()
				node, err := cs.CoreV1().Nodes().Get(ctx, nominatedNode, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get node: %v", err)
				}
				// Shrink below what the target requests (4Gi → 1Gi).
				node.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("1Gi")
				if _, err := cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("failed to update node status: %v", err)
				}
			},
		},
		{
			name:            "memory decrease on a different node does not re-queue waiting pod",
			expectIncrement: false,
			trigger: func(ctx context.Context, t *testing.T, cs clientset.Interface, _, otherNode string) {
				t.Helper()
				node, err := cs.CoreV1().Nodes().Get(ctx, otherNode, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get node: %v", err)
				}
				node.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("1Gi")
				if _, err := cs.CoreV1().Nodes().UpdateStatus(ctx, node, metav1.UpdateOptions{}); err != nil {
					t.Fatalf("failed to update node status: %v", err)
				}
			},
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nominatedNode := fmt.Sprintf("nominated-node-%d", idx)
			otherNode := fmt.Sprintf("other-node-%d", idx)

			// Create Nodes with 8Gi enough to eventually schedule the target.
			for _, name := range []string{nominatedNode, otherNode} {
				node := st.MakeNode().Name(name).Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "10",
					v1.ResourceMemory: "8Gi",
				}).Obj()
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create node %s: %v", name, err)
				}
			}

			// High-priority filler pods consume both nodes (7Gi each).  Because the
			// target has lower priority it cannot preempt them, so it parks in
			// unschedulableEntities without any preemption side-effects between subtests.
			for _, name := range []string{nominatedNode, otherNode} {
				filler := initPausePod(&testutils.PausePodConfig{
					Name:      fmt.Sprintf("filler-%s-%d", name, idx),
					Namespace: ns,
					Priority:  &asyncframework.HighPriority,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("7Gi"),
					}},
				})
				if _, err := runPausePod(cs, filler); err != nil {
					t.Fatalf("failed to run filler pod on node %s: %v", name, err)
				}
			}

			// Low-priority target requests 4Gi — can't fit anywhere and can't preempt the high-priority fillers.
			target := initPausePod(&testutils.PausePodConfig{
				Name:      fmt.Sprintf("target-%d", idx),
				Namespace: ns,
				Priority:  &asyncframework.LowPriority,
				Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("4Gi"),
				}},
			})
			target, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, target, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("failed to create target pod: %v", err)
			}

			// Wait for the target to land in unschedulableEntities.
			queue := testCtx.Scheduler.SchedulingQueue
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(_ context.Context) (bool, error) {
				for _, p := range queue.UnschedulablePods() {
					if p.Name == target.Name {
						return true, nil
					}
				}
				return false, nil
			}); err != nil {
				t.Fatalf("target pod not found in unschedulable queue: %v", err)
			}

			// Patch NominatedNodeName to reproduce the post-preemption window.  This is
			// exactly the value the scheduler writes after it selects a preemption victim
			// but before the victim terminates.
			patch := fmt.Sprintf(`{"status":{"nominatedNodeName":%q}}`, nominatedNode)
			if _, err := cs.CoreV1().Pods(ns).Patch(
				testCtx.Ctx, target.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{}, "status",
			); err != nil {
				t.Fatalf("failed to patch NominatedNodeName: %v", err)
			}

			// The patch triggers a generic pod-update event that briefly moves the pod to activeQ.
			// Wait until it has cycled back to unschedulableEntities with NominatedNodeName set
			// only then is the baseline stable for counting further scheduling attempts.
			var initialAttempts int
			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
				for _, p := range queue.UnschedulablePods() {
					if p.Name == target.Name && p.Status.NominatedNodeName == nominatedNode {
						qp, found := queue.GetPod(ctx, p.Name, p.Namespace, nil)
						if found {
							initialAttempts = qp.Attempts
						}
						return found, nil
					}
				}
				return false, nil
			}); err != nil {
				t.Fatalf("pod did not return to unschedulable queue with NominatedNodeName set: %v", err)
			}

			// Trigger the node allocatable change.
			tc.trigger(testCtx.Ctx, t, cs, nominatedNode, otherNode)

			if tc.expectIncrement {
				// The queueing hint must fire and move the target to activeQ, causing the scheduler to attempt it.
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					p, found := queue.GetPod(ctx, target.Name, ns, nil)
					return found && p.Attempts > initialAttempts, nil
				}); err != nil {
					t.Fatalf("expected scheduling attempts to increment after node allocatable decrease, but they did not")
				}
			} else {
				// The allocatable change is on a node the target pod was not nominated to, so increament is not expected.
				_ = wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 300*time.Millisecond, false, func(ctx context.Context) (bool, error) {
					p, found := queue.GetPod(ctx, target.Name, ns, nil)
					if found && p.Attempts > initialAttempts {
						t.Errorf("attempts incremented from %d to %d unexpectedly", initialAttempts, p.Attempts)
						return true, nil
					}
					return false, nil
				})
			}

			testutils.CleanupPods(testCtx.Ctx, cs, t, []*v1.Pod{target})
		})
	}
}
