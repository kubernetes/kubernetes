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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// TestNominationChanges checks that the queue lets pods retry which an ended
// nomination blocked, and only then. Without the DRAPreemption feature gate,
// the queue leaves that to the event handlers.
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
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
				testNominationChange(t, pod, lowPod, highPod, tc.change, tc.wantMoved && draPreemption)
			})
		}
	}
}

func testNominationChange(t *testing.T, pod, lowPod, highPod *v1.Pod, change func(t *testing.T, q *PriorityQueue), wantMoved bool) {
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

	change(t, q)
	if moved := getUnschedulablePod(q, lowPod) == nil; moved != wantMoved {
		t.Errorf("Lower priority pod moved: got %v, want %v", moved, wantMoved)
	}
	if getUnschedulablePod(q, highPod) == nil {
		t.Error("Higher priority pod moved, it did not have to respect the nomination")
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
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
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
		WithNominationChangeHandler(func(_ klog.Logger, pod *v1.Pod, oldNodeName, newNodeName string) []framework.NominationEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s assigned=%s", pod.Name, oldNodeName, newNodeName, pod.Spec.NodeName))
			if oldNodeName == "" {
				return nil
			}
			return []framework.NominationEvent{{Event: claimUpdate}}
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
	want := []string{"p ->node1 assigned=", "p node1-> assigned=node1"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("Unexpected handler calls (-want, +got):\n%s", diff)
	}
}

// TestRenominationToSameNode checks that an explicit nomination to the node
// which the pod is nominated to already is reported as a change, so that the
// handler can replace the previous one, without the generic event: the
// pod keeps occupying the node.
func TestRenominationToSameNode(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	logger, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	var calls []string
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, pod *v1.Pod, oldNodeName, newNodeName string) []framework.NominationEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s", pod.Name, oldNodeName, newNodeName))
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
	q.AddNominatedPod(logger, podInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"})
	if getUnschedulablePod(q, lowPod) == nil {
		t.Error("Lower priority pod moved, but the pod keeps occupying its node")
	}
	want := []string{"p ->node1", "p node1->node1"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("Unexpected handler calls (-want, +got):\n%s", diff)
	}
}
