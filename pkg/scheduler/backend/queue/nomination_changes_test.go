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

package queue

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// TestNominationChanges checks that the queue lets pods retry which an ended
// nomination blocked, and only then.
func TestNominationChanges(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	withNomination := func(nodeName string) *v1.Pod {
		p := pod.DeepCopy()
		p.Status.NominatedNodeName = nodeName
		return p
	}
	assumedOn := func(nodeName string) *v1.Pod {
		p := pod.DeepCopy()
		p.Spec.NodeName = nodeName
		return p
	}
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()
	highPod := st.MakePod().Name("high").Namespace("ns").UID("high").Priority(highPriority + 1).Obj()

	for name, tc := range map[string]struct {
		change    func(t *testing.T, q *PriorityQueue)
		wantMoved bool
	}{
		"re-added-unchanged": {
			change: func(t *testing.T, q *PriorityQueue) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustPodInfo(t, pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeNoop})
			},
		},
		"refreshed-by-update": {
			change: func(t *testing.T, q *PriorityQueue) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNomination("node1"))
			},
		},
		"changed-by-update": {
			change: func(t *testing.T, q *PriorityQueue) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNomination("node2"))
			},
			wantMoved: true,
		},
		"cleared-in-memory": {
			change: func(t *testing.T, q *PriorityQueue) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustPodInfo(t, pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride})
			},
			wantMoved: true,
		},
		"deleted": {
			change: func(t *testing.T, q *PriorityQueue) {
				logger, _ := ktesting.NewTestContext(t)
				q.Delete(logger, pod)
			},
			wantMoved: true,
		},
		// The pod uses the resources on its nominated node now.
		"assumed-on-nominated-node": {
			change: func(t *testing.T, q *PriorityQueue) {
				q.DeleteNominatedPodIfExists(assumedOn("node1"))
			},
		},
		"assumed-on-other-node": {
			change: func(t *testing.T, q *PriorityQueue) {
				q.DeleteNominatedPodIfExists(assumedOn("node2"))
			},
			wantMoved: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			m := makeEmptyQueueingHintMapPerProfile()
			m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod, highPod},
				WithQueueingHintMapPerProfile(m))

			// lowPod and highPod were rejected while pod was nominated.
			for _, p := range []*v1.Pod{lowPod, highPod} {
				q.Add(ctx, p)
				if _, err := q.Pop(logger); err != nil {
					t.Fatalf("Pop: %v", err)
				}
				if err := q.AddUnschedulablePodIfNotPresent(ctx, q.newQueuedPodInfo(ctx, p, "fooPlugin"), q.SchedulingCycle()); err != nil {
					t.Fatalf("AddUnschedulablePodIfNotPresent: %v", err)
				}
			}
			// A new nomination does not unblock anything.
			q.Add(ctx, pod)
			if getUnschedulablePod(q, lowPod) == nil {
				t.Fatal("Lower priority pod moved when the pod got nominated")
			}

			tc.change(t, q)
			if moved := getUnschedulablePod(q, lowPod) == nil; moved != tc.wantMoved {
				t.Errorf("Lower priority pod moved: got %v, want %v", moved, tc.wantMoved)
			}
			if getUnschedulablePod(q, highPod) == nil {
				t.Error("Higher priority pod moved, it did not have to respect the nomination")
			}
		})
	}
}

func mustPodInfo(t *testing.T, pod *v1.Pod) *framework.PodInfo {
	t.Helper()
	podInfo, err := framework.NewPodInfo(pod)
	if err != nil {
		t.Fatal(err)
	}
	return podInfo
}

// TestNominationChangeHandler checks that the handler sees every change of a
// nominated node and that the events which it returns let pods with lower or
// equal priority retry, also when the pod got assumed on its nominated node.
func TestNominationChangeHandler(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	assumed := pod.DeepCopy()
	assumed.Spec.NodeName = "node1"
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()
	highPod := st.MakePod().Name("high").Namespace("ns").UID("high").Priority(highPriority + 1).Obj()
	claimUpdate := fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Update}

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][claimUpdate] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	var calls []string
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod, highPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, state fwk.CycleState, pod *v1.Pod, oldNodeName, newNodeName string) []framework.ResourceEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s assigned=%s state=%v", pod.Name, oldNodeName, newNodeName, pod.Spec.NodeName, state != nil))
			if oldNodeName == "" {
				return nil
			}
			return []framework.ResourceEvent{{Event: claimUpdate}}
		}))
	for _, p := range []*v1.Pod{lowPod, highPod} {
		q.Add(ctx, p)
		if _, err := q.Pop(logger); err != nil {
			t.Fatalf("Pop: %v", err)
		}
		if err := q.AddUnschedulablePodIfNotPresent(ctx, q.newQueuedPodInfo(ctx, p, "fooPlugin"), q.SchedulingCycle()); err != nil {
			t.Fatalf("AddUnschedulablePodIfNotPresent: %v", err)
		}
	}

	q.Add(ctx, pod)
	if getUnschedulablePod(q, lowPod) == nil {
		t.Fatal("Lower priority pod moved when the pod got nominated")
	}
	// Whether the pod got the held resources is up to the plugins.
	q.DeleteNominatedPodIfExists(assumed)
	if getUnschedulablePod(q, lowPod) != nil {
		t.Error("Lower priority pod not moved by the event of the handler")
	}
	if getUnschedulablePod(q, highPod) == nil {
		t.Error("Higher priority pod moved, it did not have to respect the nomination")
	}
	want := []string{"p ->node1 assigned= state=false", "p node1-> assigned=node1 state=false"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("Unexpected handler calls (-want, +got):\n%s", diff)
	}
}

// TestRenominationToSameNode checks that an explicit nomination to the node
// which the pod is nominated to already is reported as a change, so that the
// handler can replace the previous one, without the generic event: the
// pod keeps occupying the node.
func TestRenominationToSameNode(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	var calls []string
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, state fwk.CycleState, pod *v1.Pod, oldNodeName, newNodeName string) []framework.ResourceEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s state=%v", pod.Name, oldNodeName, newNodeName, state != nil))
			return nil
		}))
	q.Add(ctx, lowPod)
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop: %v", err)
	}
	if err := q.AddUnschedulablePodIfNotPresent(ctx, q.newQueuedPodInfo(ctx, lowPod, "fooPlugin"), q.SchedulingCycle()); err != nil {
		t.Fatalf("AddUnschedulablePodIfNotPresent: %v", err)
	}
	q.Add(ctx, pod)
	podInfo := mustNewPodInfo(pod)

	// Neither keeping the nomination after a failure nor an informer update
	// is a new nomination.
	q.AddNominatedPod(logger, podInfo, nil)
	updated := pod.DeepCopy()
	updated.Labels = map[string]string{"updated": "true"}
	q.Update(ctx, pod, updated)
	state := framework.NewCycleState()
	q.AddNominatedPodWithState(logger, state, podInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"})
	if getUnschedulablePod(q, lowPod) == nil {
		t.Error("Lower priority pod moved, but the pod keeps occupying its node")
	}
	want := []string{"p ->node1 state=false", "p node1->node1 state=true"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("Unexpected handler calls (-want, +got):\n%s", diff)
	}
}

// TestNominationChangeHandlerLockReentrancyAndWaking verifies:
//   - NominationChangeHandler runs outside PriorityQueue.lock so calling back
//     into q.NominatedPodsForNode does not deadlock.
//   - Clearing NominatedNodeName via external Update wakes the displaced pod itself.
//   - In-flight Update updates nominator state.
//   - In-flight events carry preCheck so higher-priority in-flight pods are not
//     spuriously bounced by lower-priority nomination release events.
func TestNominationChangeHandlerLockReentrancyAndWaking(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	nomPod := st.MakePod().Name("nom").Namespace("ns").UID("nom").Priority(highPriority).NominatedNodeName("node1").Obj()
	highPod := st.MakePod().Name("high").Namespace("ns").UID("high").Priority(highPriority + 1).Obj()
	claimUpdate := fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Update}

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	m[""][claimUpdate] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}

	var q *PriorityQueue
	var sawState bool
	q = NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{nomPod, highPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, state fwk.CycleState, pod *v1.Pod, oldNodeName, newNodeName string) []framework.ResourceEvent {
			// Must not deadlock by re-entering PriorityQueue methods.
			_ = q.NominatedPodsForNode(logger, "node1")
			_ = q.NominatedNodeForPod(pod.UID)
			if state != nil {
				sawState = true
			}
			if oldNodeName != "" {
				return []framework.ResourceEvent{{Event: claimUpdate}}
			}
			return nil
		}))

	// 1. Add nomPod, pop it (in-flight), and update its NominatedNodeName while in-flight.
	q.Add(ctx, nomPod)
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop nomPod: %v", err)
	}
	nomPodNode2 := nomPod.DeepCopy()
	nomPodNode2.Status.NominatedNodeName = "node2"
	q.Update(ctx, nomPod, nomPodNode2)
	if got := q.NominatedNodeForPod(nomPod.UID); got != "node2" {
		t.Fatalf("NominatedNodeForPod after in-flight Update = %q, want %q", got, "node2")
	}

	// Requeue nomPod via AddUnschedulablePodWithNomination with CycleState.
	cycleState := framework.NewCycleState()
	if err := q.AddUnschedulablePodWithNomination(ctx, cycleState, q.newQueuedPodInfo(ctx, nomPodNode2, "fooPlugin"), q.SchedulingCycle(),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"}); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination: %v", err)
	}
	if !sawState {
		t.Error("Expected NominationChangeHandler to receive non-nil CycleState from AddUnschedulablePodWithNomination")
	}

	// 2. Pop highPod so it is in-flight while nomPod's nomination is cleared externally.
	q.Add(ctx, highPod)
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop highPod: %v", err)
	}

	nomPodCleared := nomPod.DeepCopy()
	nomPodCleared.Status.NominatedNodeName = ""
	q.Update(ctx, nomPodNode2, nomPodCleared)

	// Displaced nomPod itself must wake up from unschedulableEntities.
	if getUnschedulablePod(q, nomPod) != nil {
		t.Error("Expected displaced nomPod to move out of unschedulable queue when NominatedNodeName was cleared by Update")
	}

	// highPod has higher priority than nomPod, so preCheck on in-flight events must ignore nomPod's release events.
	if err := q.AddUnschedulablePodWithNomination(ctx, framework.NewCycleState(), q.newQueuedPodInfo(ctx, highPod, "fooPlugin"), q.SchedulingCycle(),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: ""}); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination highPod: %v", err)
	}
	if getUnschedulablePod(q, highPod) == nil {
		t.Error("Expected higher-priority in-flight pod to stay in unschedulable queue despite lower-priority nomination release")
	}
}

// TestAddUnschedulablePodWithNominationModeNoopPreservesInMemoryNomination verifies
// that ModeNoop requeues do not overwrite an in-memory nomination (or in-memory clear)
// with a stale cachedPod.Status.NominatedNodeName while an async status patch is in-flight.
func TestAddUnschedulablePodWithNominationModeNoopPreservesInMemoryNomination(t *testing.T) {
	logger, ctx := ktesting.NewTestContext(t)
	// cachedPod in podLister still has empty NominatedNodeName while async status patch is in-flight.
	cachedPod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}

	var calls []string
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{cachedPod, lowPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, state fwk.CycleState, pod *v1.Pod, oldNodeName, newNodeName string) []framework.ResourceEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s state=%v", pod.Name, oldNodeName, newNodeName, state != nil))
			return nil
		}))

	// Park lowPod in unschedulable queue.
	q.Add(ctx, lowPod)
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop lowPod: %v", err)
	}
	if err := q.AddUnschedulablePodIfNotPresent(ctx, q.newQueuedPodInfo(ctx, lowPod, "fooPlugin"), q.SchedulingCycle()); err != nil {
		t.Fatalf("AddUnschedulablePodIfNotPresent lowPod: %v", err)
	}

	// Cycle 1: nominate cachedPod to "node1" in-memory via ModeOverride("node1").
	q.Add(ctx, cachedPod)
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop cachedPod cycle 1: %v", err)
	}
	if err := q.AddUnschedulablePodWithNomination(ctx, framework.NewCycleState(), q.newQueuedPodInfo(ctx, cachedPod, "fooPlugin"), q.SchedulingCycle(),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"}); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination cycle 1: %v", err)
	}
	if got := q.NominatedNodeForPod(cachedPod.UID); got != "node1" {
		t.Fatalf("NominatedNodeForPod after cycle 1 = %q, want %q", got, "node1")
	}

	// Cycle 2: wake cachedPod before async patch reaches podLister (cachedPod.Status.NominatedNodeName == ""),
	// then requeue with nil NominatingInfo (ModeNoop). In-memory "node1" nomination must remain intact.
	q.Activate(logger, map[string]*v1.Pod{cachedPod.Name: cachedPod})
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop cycle 2: %v", err)
	}
	if err := q.AddUnschedulablePodWithNomination(ctx, framework.NewCycleState(), q.newQueuedPodInfo(ctx, cachedPod.DeepCopy(), "fooPlugin"), q.SchedulingCycle(), nil); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination cycle 2 ModeNoop: %v", err)
	}
	if got := q.NominatedNodeForPod(cachedPod.UID); got != "node1" {
		t.Fatalf("NominatedNodeForPod after ModeNoop requeue = %q, want %q", got, "node1")
	}
	if getUnschedulablePod(q, lowPod) == nil {
		t.Fatal("Lower-priority pod woke up on ModeNoop requeue")
	}
	if diff := cmp.Diff([]string{"p ->node1 state=true"}, calls); diff != "" {
		t.Fatalf("Unexpected handler calls after ModeNoop requeue (-want, +got):\n%s", diff)
	}

	// Cycle 3 & 4: clear "node1" via ModeOverride(""), then requeue via ModeNoop with a stale Pod copy
	// still carrying NominatedNodeName="node1"; the cleared state ("") must not be overwritten.
	q.Activate(logger, map[string]*v1.Pod{cachedPod.Name: cachedPod})
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop cycle 3: %v", err)
	}
	if err := q.AddUnschedulablePodWithNomination(ctx, framework.NewCycleState(), q.newQueuedPodInfo(ctx, cachedPod.DeepCopy(), "fooPlugin"), q.SchedulingCycle(),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: ""}); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination clear: %v", err)
	}
	q.Activate(logger, map[string]*v1.Pod{cachedPod.Name: cachedPod})
	if _, err := q.Pop(logger); err != nil {
		t.Fatalf("Pop cycle 4: %v", err)
	}
	stalePodWithNode1 := cachedPod.DeepCopy()
	stalePodWithNode1.Status.NominatedNodeName = "node1"
	if err := q.AddUnschedulablePodWithNomination(ctx, framework.NewCycleState(), q.newQueuedPodInfo(ctx, stalePodWithNode1, "fooPlugin"), q.SchedulingCycle(),
		&fwk.NominatingInfo{NominatingMode: fwk.ModeNoop}); err != nil {
		t.Fatalf("AddUnschedulablePodWithNomination cycle 4 ModeNoop: %v", err)
	}
	if got := q.NominatedNodeForPod(cachedPod.UID); got != "" {
		t.Fatalf("NominatedNodeForPod after ModeNoop with stale pod status = %q, want empty", got)
	}
	if diff := cmp.Diff([]string{"p ->node1 state=true", "p node1-> state=true"}, calls); diff != "" {
		t.Fatalf("Unexpected handler calls after clear + ModeNoop (-want, +got):\n%s", diff)
	}
}
