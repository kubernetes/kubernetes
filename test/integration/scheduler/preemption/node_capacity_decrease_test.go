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
// scheduling cycle when the nominated node's allocatable drops below what the
// pod requests.
//
// When a node shrinks to the point where the pod can no longer fit even on an
// otherwise empty node, Fit.Filter marks it UnschedulableAndUnresolvable.  That
// makes the pod eligible to preempt on a different node, but only if the
// scheduler runs a fresh cycle.  The queueing hint in isSchedulableAfterNodeChange
// fires Queue in this case so the pod is not stuck waiting for the original
// victims to finish terminating on a node that can never host it.
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
				// Shrink below what the target requests (8Gi → 1Gi).
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

			// Create nodes with 8Gi, enough to host the target after eviction.
			for _, name := range []string{nominatedNode, otherNode} {
				node := st.MakeNode().Name(name).Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "10",
					v1.ResourceMemory: "8Gi",
				}).Obj()
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create node %s: %v", name, err)
				}
			}
			// Register cleanup here so it runs even if the subtest fails, keeping
			// the shared scheduler clean for the next subtest.
			t.Cleanup(func() {
				for _, name := range []string{nominatedNode, otherNode} {
					_ = cs.CoreV1().Pods(ns).DeleteCollection(testCtx.Ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{
						FieldSelector: "spec.nodeName=" + name,
					})
					_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, name, *metav1.NewDeleteOptions(0))
				}
			})

			// High-priority filler pods consume both nodes (7Gi each). The target has lower
			// priority so it cannot preempt them and parks in unschedulableEntities.
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

			// Patch NominatedNodeName to reproduce the post-preemption window.
			patch := fmt.Sprintf(`{"status":{"nominatedNodeName":%q}}`, nominatedNode)
			if _, err := cs.CoreV1().Pods(ns).Patch(
				testCtx.Ctx, target.Name, types.MergePatchType, []byte(patch), metav1.PatchOptions{}, "status",
			); err != nil {
				t.Fatalf("failed to patch NominatedNodeName: %v", err)
			}

			// Wait until the target is back in unschedulableEntities with NominatedNodeName set,
			// then snapshot the attempt count as the baseline.
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
				t.Fatalf("pod did not reach unschedulable queue with NominatedNodeName=%s: %v", nominatedNode, err)
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
				// The allocatable change is on a node the target pod was not nominated to, so no increment is expected.
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

// TestPreemptsElsewhereWhenNominatedNodeShrinks is an end-to-end test of the full
// preemption-and-capacity-decrease flow:
//
//  1. Two nodes: node-a holds a low-priority victim, node-b holds a medium-priority victim.
//  2. A high-priority target pod preempts on node-a and gets NominatedNodeName=node-a.
//  3. While the victim on node-a is still Terminating, node-a's allocatable is shrunk
//     below the target's request.
//  4. The scheduler detects the shrink via the queueing hint, re-queues the target,
//     and preempts the victim on node-b instead.
//  5. The target eventually binds to node-b.
//
func TestPreemptsElsewhereWhenNominatedNodeShrinks(t *testing.T) {
	schedulerName := v1.DefaultSchedulerName
	cfg := configtesting.V1ToInternalWithDefaults(t, configv1.KubeSchedulerConfiguration{
		Profiles: []configv1.KubeSchedulerProfile{{
			SchedulerName: &schedulerName,
		}},
	})

	testCtx := testutils.InitTestSchedulerWithOptions(t,
		testutils.InitTestAPIServer(t, "nominated-node-shrink", nil),
		0,
		scheduler.WithProfiles(cfg.Profiles...),
	)
	defer testCtx.SchedulerCloseFn()
	testutils.SyncSchedulerInformerFactory(testCtx)
	go testCtx.Scheduler.Run(testCtx.Ctx)

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	for _, name := range []string{"node-a", "node-b"} {
		node := st.MakeNode().Name(name).Capacity(map[v1.ResourceName]string{
			v1.ResourcePods:   "10",
			v1.ResourceMemory: "8Gi",
		}).Obj()
		if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create node %s: %v", name, err)
		}
	}

	// victim-node-a has strictly lower priority than victim-node-b, so the scheduler
	// always picks node-a as the preemption target first, making the test deterministic.
	victimPriorities := map[string]int32{
		"node-a": asyncframework.LowPriority,
		"node-b": asyncframework.MediumPriority,
	}
	for _, nodeName := range []string{"node-a", "node-b"} {
		p := victimPriorities[nodeName]
		victim := initPausePod(&testutils.PausePodConfig{
			Name:      "victim-" + nodeName,
			Namespace: ns,
			Priority:  &p,
			NodeName:  nodeName,
			Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("6Gi"),
			}},
		})
		if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victim, metav1.CreateOptions{}); err != nil {
			t.Fatalf("failed to create victim on %s: %v", nodeName, err)
		}
	}

	// High-priority target requests 4Gi, fits after evicting either victim.
	target := initPausePod(&testutils.PausePodConfig{
		Name:      "target",
		Namespace: ns,
		Priority:  &asyncframework.HighPriority,
		Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
			v1.ResourceMemory: resource.MustParse("4Gi"),
		}},
	})
	target, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, target, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create target pod: %v", err)
	}

	// The scheduler always nominates node-a first (lowest-priority victim).
	if err := testutils.WaitForNominatedNodeName(testCtx.Ctx, cs, target); err != nil {
		t.Fatalf("target pod did not get NominatedNodeName: %v", err)
	}
	target, err = cs.CoreV1().Pods(ns).Get(testCtx.Ctx, target.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to re-fetch target: %v", err)
	}
	if target.Status.NominatedNodeName != "node-a" {
		t.Fatalf("expected target to be nominated to node-a, got %s", target.Status.NominatedNodeName)
	}

	// Wait until the victim on node-a is Terminating (preemption has acted).
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
		testutils.PodIsGettingEvicted(cs, ns, "victim-node-a")); err != nil {
		t.Fatalf("victim on node-a did not enter Terminating state: %v", err)
	}

	// Shrink node-a's allocatable below the target's request while the victim is still
	// Terminating. The queueing hint should fire and re-queue the target.
	node, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, "node-a", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get node-a: %v", err)
	}
	node.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("1Gi")
	if _, err := cs.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, node, metav1.UpdateOptions{}); err != nil {
		t.Fatalf("failed to shrink node-a: %v", err)
	}

	// The target must be re-nominated to node-b while the node-a victim is still Terminating.
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false, func(ctx context.Context) (bool, error) {
		p, err := cs.CoreV1().Pods(ns).Get(ctx, target.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return p.Status.NominatedNodeName == "node-b", nil
	}); err != nil {
		t.Fatalf("target was not re-nominated to node-b after node-a shrank: %v", err)
	}

	// Confirm node-a's victim is still Terminating.
	firstVictim, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, "victim-node-a", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to fetch victim-node-a: %v", err)
	}
	if firstVictim.DeletionTimestamp == nil {
		t.Errorf("expected victim-node-a to still be Terminating, but DeletionTimestamp is nil")
	}

	// Assert that the victim on node-b was preempted before manually cleaning up.
	if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
		testutils.PodIsGettingEvicted(cs, ns, "victim-node-b")); err != nil {
		t.Fatalf("victim on node-b was not preempted after re-nomination: %v", err)
	}

	// Force-delete the victims so the target can actually bind, then wait for it to schedule.
	for _, nodeName := range []string{"node-a", "node-b"} {
		if err := cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, "victim-"+nodeName, *metav1.NewDeleteOptions(0)); err != nil {
			t.Logf("warning: failed to force-delete victim-%s: %v", nodeName, err)
		}
	}
	if err := testutils.WaitForPodToSchedule(testCtx.Ctx, cs, target); err != nil {
		t.Fatalf("target pod did not schedule after victims were removed: %v", err)
	}
	finalTarget, err := cs.CoreV1().Pods(ns).Get(testCtx.Ctx, target.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to fetch final target state: %v", err)
	}
	if finalTarget.Spec.NodeName != "node-b" {
		t.Errorf("target scheduled on %s, expected node-b", finalTarget.Spec.NodeName)
	}
}
