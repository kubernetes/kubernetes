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
	"k8s.io/apimachinery/pkg/util/wait"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	schedulerutils "k8s.io/kubernetes/test/integration/scheduler"
	"k8s.io/kubernetes/test/integration/scheduler/preemption/asyncframework"
	testutils "k8s.io/kubernetes/test/integration/util"
)

// TestNominatedPodRequeuesOnNodeAllocatableDecrease verifies that a pod sitting
// in the unschedulable queue with NominatedNodeName set is re-queued for a fresh
// scheduling cycle when the nominated node's allocatable drops below what the
// pod requests.
//
// The nominated state is produced by real preemption: each of two nodes holds a
// victim, the high-priority target preempts the lower-priority one, and the test
// shrinks a node while that victim is still Terminating. Requeueing is observed
// through the target's scheduling attempt count.
//
// When a node shrinks to the point where the pod can no longer fit even on an
// otherwise empty node, Fit.Filter marks it UnschedulableAndUnresolvable.  That
// makes the pod eligible to preempt on a different node, but only if the
// scheduler runs a fresh cycle.  The queueing hint in isSchedulableAfterNodeChange
// fires Queue in this case so the pod is not stuck waiting for the original
// victims to finish terminating on a node that can never host it.
func TestNominatedPodRequeuesOnNodeAllocatableDecrease(t *testing.T) {
	testCtx := schedulerutils.InitTestSchedulerWithNS(t, "node-allocatable-decrease")
	defer testCtx.SchedulerCloseFn()

	cs := testCtx.ClientSet
	ns := testCtx.NS.Name

	tests := []struct {
		name            string
		shrinkNominated bool
		expectIncrement bool
	}{
		{
			name:            "memory decrease on nominated node re-queues waiting pod",
			shrinkNominated: true,
			expectIncrement: true,
		},
		{
			name:            "memory decrease on a different node does not re-queue waiting pod",
			shrinkNominated: false,
			expectIncrement: false,
		},
	}

	for idx, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			nominatedNode := fmt.Sprintf("nominated-node-%d", idx)
			otherNode := fmt.Sprintf("other-node-%d", idx)

			for _, name := range []string{nominatedNode, otherNode} {
				node := st.MakeNode().Name(name).Capacity(map[v1.ResourceName]string{
					v1.ResourcePods:   "10",
					v1.ResourceMemory: "8Gi",
				}).Obj()
				if _, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create node %s: %v", name, err)
				}
			}
			t.Cleanup(func() {
				_ = cs.CoreV1().Pods(ns).Delete(testCtx.Ctx, fmt.Sprintf("target-%d", idx), *metav1.NewDeleteOptions(0))
				for _, name := range []string{nominatedNode, otherNode} {
					_ = cs.CoreV1().Pods(ns).DeleteCollection(testCtx.Ctx, *metav1.NewDeleteOptions(0), metav1.ListOptions{
						FieldSelector: "spec.nodeName=" + name,
					})
					_ = cs.CoreV1().Nodes().Delete(testCtx.Ctx, name, *metav1.NewDeleteOptions(0))
				}
			})

			// One victim bound to each node. The victim on nominatedNode has strictly
			// lower priority, so preemption deterministically picks that node first.
			victimPriorities := map[string]int32{
				nominatedNode: asyncframework.LowPriority,
				otherNode:     asyncframework.MediumPriority,
			}
			for _, name := range []string{nominatedNode, otherNode} {
				priority := victimPriorities[name]
				victim := initPausePod(&testutils.PausePodConfig{
					Name:      "victim-" + name,
					Namespace: ns,
					Priority:  &priority,
					NodeName:  name,
					Resources: &v1.ResourceRequirements{Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("6Gi"),
					}},
				})
				if _, err := cs.CoreV1().Pods(ns).Create(testCtx.Ctx, victim, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create victim on node %s: %v", name, err)
				}
			}

			// High-priority target requests 4Gi: it fits on either node once that node's victim is gone.
			target := initPausePod(&testutils.PausePodConfig{
				Name:      fmt.Sprintf("target-%d", idx),
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

			if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 10*time.Second, false,
				testutils.PodIsGettingEvicted(cs, ns, "victim-"+nominatedNode)); err != nil {
				t.Fatalf("victim on %s did not enter Terminating state: %v", nominatedNode, err)
			}

			// Wait until the target is parked in unschedulableEntities with NominatedNodeName set,
			// then snapshot the attempt count as the baseline.
			queue := testCtx.Scheduler.SchedulingQueue
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

			// Shrink memory below what the target requests (8Gi -> 1Gi).
			shrinkNode := otherNode
			if tc.shrinkNominated {
				shrinkNode = nominatedNode
			}
			node, err := cs.CoreV1().Nodes().Get(testCtx.Ctx, shrinkNode, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("failed to get node %s: %v", shrinkNode, err)
			}
			node.Status.Allocatable[v1.ResourceMemory] = resource.MustParse("1Gi")
			if _, err := cs.CoreV1().Nodes().UpdateStatus(testCtx.Ctx, node, metav1.UpdateOptions{}); err != nil {
				t.Fatalf("failed to update status of node %s: %v", shrinkNode, err)
			}

			if tc.expectIncrement {
				// The queueing hint must fire and move the target to activeQ, causing the scheduler to attempt it.
				currentAttempts := initialAttempts
				if err := wait.PollUntilContextTimeout(testCtx.Ctx, 50*time.Millisecond, 5*time.Second, false, func(ctx context.Context) (bool, error) {
					p, found := queue.GetPod(ctx, target.Name, ns, nil)
					if found {
						currentAttempts = p.Attempts
					}
					return currentAttempts > initialAttempts, nil
				}); err != nil {
					t.Fatalf("expected scheduling attempts to increment after node allocatable decrease, but they did not: initialAttempts=%d, last observed attempts=%d", initialAttempts, currentAttempts)
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
func TestPreemptsElsewhereWhenNominatedNodeShrinks(t *testing.T) {
	testCtx := schedulerutils.InitTestSchedulerWithNS(t, "nominated-node-shrink")
	defer testCtx.SchedulerCloseFn()

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
	if err := schedulerutils.WaitForNominatedNodeName(testCtx.Ctx, cs, target); err != nil {
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
			t.Fatalf("failed to force-delete victim-%s: %v", nodeName, err)
		}
	}
	if err := schedulerutils.WaitForPodToSchedule(testCtx.Ctx, cs, target); err != nil {
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
