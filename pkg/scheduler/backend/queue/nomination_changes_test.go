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
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
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

// TestNominationExpectations checks that pod objects which don't show the
// nominated node name which the scheduler set yet don't override it. Without
// the DRAPreemption feature gate, they do.
func TestNominationExpectations(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	withNomination := func(nodeName string) *v1.Pod {
		p := pod.DeepCopy()
		p.Status.NominatedNodeName = nodeName
		return p
	}
	withLabel := func(p *v1.Pod) *v1.Pod {
		p = p.DeepCopy()
		p.Labels = map[string]string{"updated": "true"}
		return p
	}
	type step struct {
		do func(t *testing.T, q *PriorityQueue)
		// want and wantWithoutDRAPreemption are the nominated node in memory
		// after the step.
		want, wantWithoutDRAPreemption string
	}
	nominate := func(p *v1.Pod, nodeName string) func(t *testing.T, q *PriorityQueue) {
		return func(t *testing.T, q *PriorityQueue) {
			logger, _ := ktesting.NewTestContext(t)
			q.AddNominatedPod(logger, mustPodInfo(t, p), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName})
		}
	}
	// keep is what handleSchedulingFailure does when PostFilter did not
	// nominate the pod.
	keep := func(p *v1.Pod) func(t *testing.T, q *PriorityQueue) {
		return func(t *testing.T, q *PriorityQueue) {
			logger, _ := ktesting.NewTestContext(t)
			q.AddNominatedPod(logger, mustPodInfo(t, p), nil)
		}
	}
	update := func(oldPod, newPod *v1.Pod) func(t *testing.T, q *PriorityQueue) {
		return func(t *testing.T, q *PriorityQueue) {
			_, ctx := ktesting.NewTestContext(t)
			q.Update(ctx, oldPod, newPod)
		}
	}
	notPersisted := func(nodeName string) func(t *testing.T, q *PriorityQueue) {
		return func(t *testing.T, q *PriorityQueue) {
			q.NominationNotPersisted(pod, nodeName)
		}
	}

	for name, steps := range map[string][]step{
		"kept-with-stale-pod": {
			{do: nominate(pod, "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: keep(pod), want: "node1", wantWithoutDRAPreemption: ""},
		},
		"updated-with-stale-pod": {
			{do: nominate(withNomination("node0"), "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: update(withNomination("node0"), withLabel(withNomination("node0"))), want: "node1", wantWithoutDRAPreemption: "node0"},
			{do: update(withNomination("node0"), withNomination("node1")), want: "node1", wantWithoutDRAPreemption: "node1"},
			// Once the pod object showed it, it is up-to-date again.
			{do: update(withNomination("node1"), withNomination("")), want: "", wantWithoutDRAPreemption: ""},
		},
		"cleared-with-stale-pod": {
			{do: nominate(withNomination("node1"), "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: nominate(withNomination("node1"), ""), want: "", wantWithoutDRAPreemption: ""},
			{do: update(withNomination("node1"), withLabel(withNomination("node1"))), want: "", wantWithoutDRAPreemption: "node1"},
			{do: update(withNomination("node1"), withNomination("")), want: "", wantWithoutDRAPreemption: ""},
		},
		"not-persisted": {
			{do: nominate(pod, "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: notPersisted("node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: keep(pod), want: "", wantWithoutDRAPreemption: ""},
		},
		"older-nomination-not-persisted": {
			{do: nominate(pod, "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
			{do: nominate(pod, "node2"), want: "node2", wantWithoutDRAPreemption: "node2"},
			{do: notPersisted("node1"), want: "node2", wantWithoutDRAPreemption: "node2"},
			{do: keep(pod), want: "node2", wantWithoutDRAPreemption: ""},
		},
	} {
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
				_, ctx := ktesting.NewTestContext(t)
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod})
				for i, step := range steps {
					step.do(t, q)
					want := step.want
					if !draPreemption {
						want = step.wantWithoutDRAPreemption
					}
					q.nominator.nLock.RLock()
					nodeName := q.nominator.nominatedPodToNode[pod.UID]
					q.nominator.nLock.RUnlock()
					if nodeName != want {
						t.Errorf("step %d: nominated node %q, want %q", i, nodeName, want)
					}
				}
				q.Delete(klog.FromContext(ctx), pod)
				if len(q.nominator.expected) > 0 {
					t.Errorf("expected nominations left after deleting the pod: %v", q.nominator.expected)
				}
			})
		}
	}
}

// TestNominatedNodeName checks that the nominated node of a pod is what the
// scheduler set as long as the pod object doesn't show it yet, and what the
// pod object shows otherwise. Without the DRAPreemption feature gate, it is
// always what the pod object shows.
func TestNominatedNodeName(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	withNomination := func(nodeName string) *v1.Pod {
		p := pod.DeepCopy()
		p.Status.NominatedNodeName = nodeName
		return p
	}
	type nomination struct {
		// pod is the pod object at the time of the nomination.
		pod      *v1.Pod
		nodeName string
	}

	for name, tc := range map[string]struct {
		// nominations is what the scheduler set, in this order.
		nominations []nomination
		pod         *v1.Pod
		// want and wantWithoutDRAPreemption are the nominated node for pod.
		want, wantWithoutDRAPreemption string
	}{
		"not-nominated": {
			pod: pod,
		},
		"nominated-through-api": {
			pod:                      withNomination("node1"),
			want:                     "node1",
			wantWithoutDRAPreemption: "node1",
		},
		"nominated-with-stale-pod": {
			nominations: []nomination{{pod, "node1"}},
			pod:         pod,
			want:        "node1",
		},
		"nominated-with-up-to-date-pod": {
			nominations:              []nomination{{pod, "node1"}},
			pod:                      withNomination("node1"),
			want:                     "node1",
			wantWithoutDRAPreemption: "node1",
		},
		"nominated-elsewhere-with-stale-pod": {
			nominations:              []nomination{{pod, "node1"}, {withNomination("node1"), "node2"}},
			pod:                      withNomination("node1"),
			want:                     "node2",
			wantWithoutDRAPreemption: "node1",
		},
		"cleared-with-stale-pod": {
			nominations:              []nomination{{withNomination("node1"), "node1"}, {withNomination("node1"), ""}},
			pod:                      withNomination("node1"),
			wantWithoutDRAPreemption: "node1",
		},
	} {
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
				logger, ctx := ktesting.NewTestContext(t)
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod})
				for _, nomination := range tc.nominations {
					q.AddNominatedPod(logger, mustPodInfo(t, nomination.pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nomination.nodeName})
				}
				want := tc.want
				if !draPreemption {
					want = tc.wantWithoutDRAPreemption
				}
				if nodeName := q.NominatedNodeName(tc.pod); nodeName != want {
					t.Errorf("Nominated node %q, want %q", nodeName, want)
				}
			})
		}
	}
}

// TestPatchPodStatusOverwritten checks that an update of the nominated node
// name which got overwritten by another one lets the nominator take it from
// pod objects again.
func TestPatchPodStatusOverwritten(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	logger, ctx := ktesting.NewTestContext(t)
	client := clientsetfake.NewClientset(pod)
	apiDispatcher := apidispatcher.New(client, 1, apicalls.Relevances)
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod}, WithAPIDispatcher(apiDispatcher))

	nominatingInfo := &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"}
	q.AddNominatedPod(logger, mustPodInfo(t, pod), nominatingInfo)
	onFinish, err := q.PatchPodStatus(pod, nil, nominatingInfo)
	if err != nil {
		t.Fatalf("PatchPodStatus: %v", err)
	}
	// Like preemption clearing the nomination of another pod, before the
	// dispatcher executed the first call.
	if _, err := q.PatchPodStatus(pod, nil, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride}); err != nil {
		t.Fatalf("PatchPodStatus: %v", err)
	}
	apiDispatcher.Run(logger)
	defer apiDispatcher.Close()
	select {
	case err := <-onFinish:
		if !errors.Is(err, fwk.ErrCallOverwritten) {
			t.Fatalf("Unexpected result of the call: %v", err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("Call did not finish")
	}
	if _, expected := q.nominator.expectedNodeName(pod.UID); expected {
		t.Error("Nomination still expected")
	}
}

// TestPatchPodStatusNotPersisted checks that a failed asynchronous update of
// the nominated node name lets the nominator take it from pod objects again.
func TestPatchPodStatusNotPersisted(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	nominatingInfo := &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"}

	for name, patchErr := range map[string]error{
		"succeeded": nil,
		"failed":    errors.New("injected patch error"),
	} {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset(pod)
			client.PrependReactor("patch", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				return patchErr != nil, nil, patchErr
			})
			apiDispatcher := apidispatcher.New(client, 1, apicalls.Relevances)
			apiDispatcher.Run(logger)
			defer apiDispatcher.Close()
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod}, WithAPIDispatcher(apiDispatcher))

			q.AddNominatedPod(logger, mustPodInfo(t, pod), nominatingInfo)
			onFinish, err := q.PatchPodStatus(pod, nil, nominatingInfo)
			if err != nil {
				t.Fatalf("PatchPodStatus: %v", err)
			}
			select {
			case err := <-onFinish:
				if (err != nil) != (patchErr != nil) {
					t.Fatalf("Unexpected result of the call: %v", err)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("Call did not finish")
			}
			if _, expected := q.nominator.expectedNodeName(pod.UID); expected != (patchErr == nil) {
				t.Errorf("Nomination still expected: %v", expected)
			}
		})
	}
}

// TestPatchPodStatusMerged checks that the nominator keeps waiting for the
// nominated node name when the call which sets it gets merged into a later
// one which only updates a condition: the merged call still sets it. Only a
// failure of the merged call lets the nominator take the nominated node name
// from pod objects again.
func TestPatchPodStatusMerged(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	nominatingInfo := &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node1"}
	condition := &v1.PodCondition{Type: v1.PodScheduled, Status: v1.ConditionFalse, Reason: v1.PodReasonUnschedulable}

	for name, patchErr := range map[string]error{
		"succeeded": nil,
		"failed":    errors.New("injected patch error"),
	} {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			client := clientsetfake.NewClientset(pod)
			client.PrependReactor("patch", "pods", func(clienttesting.Action) (bool, runtime.Object, error) {
				return patchErr != nil, nil, patchErr
			})
			apiDispatcher := apidispatcher.New(client, 1, apicalls.Relevances)
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod}, WithAPIDispatcher(apiDispatcher))

			q.AddNominatedPod(logger, mustPodInfo(t, pod), nominatingInfo)
			onFinish, err := q.PatchPodStatus(pod, nil, nominatingInfo)
			if err != nil {
				t.Fatalf("PatchPodStatus: %v", err)
			}
			// Like the next scheduling attempt ending without a
			// PostFilterResult, before the dispatcher executed the first call.
			q.AddNominatedPod(logger, mustPodInfo(t, pod), nil)
			mergedOnFinish, err := q.PatchPodStatus(pod, []*v1.PodCondition{condition}, nil)
			if err != nil {
				t.Fatalf("PatchPodStatus: %v", err)
			}
			select {
			case err := <-onFinish:
				if !errors.Is(err, fwk.ErrCallOverwritten) {
					t.Fatalf("Unexpected result of the first call: %v", err)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("First call did not finish")
			}
			// The pod object still does not show the nominated node name in
			// the attempt after that.
			q.AddNominatedPod(logger, mustPodInfo(t, pod), nil)
			q.nominator.nLock.RLock()
			nodeName := q.nominator.nominatedPodToNode[pod.UID]
			q.nominator.nLock.RUnlock()
			if nodeName != "node1" {
				t.Errorf("Nominated node %q before the merged call got executed, want node1", nodeName)
			}

			apiDispatcher.Run(logger)
			defer apiDispatcher.Close()
			select {
			case err := <-mergedOnFinish:
				if (err != nil) != (patchErr != nil) {
					t.Fatalf("Unexpected result of the merged call: %v", err)
				}
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatal("Merged call did not finish")
			}
			if _, expected := q.nominator.expectedNodeName(pod.UID); expected != (patchErr == nil) {
				t.Errorf("Nomination still expected: %v", expected)
			}
			if patchErr != nil {
				return
			}
			updatedPod, err := client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Get pod: %v", err)
			}
			if updatedPod.Status.NominatedNodeName != "node1" {
				t.Errorf("Nominated node name %q set by the merged call, want node1", updatedPod.Status.NominatedNodeName)
			}
		})
	}
}

// TestNominationChangeOfPodItself checks that a pod whose nomination got
// changed by something else than its own scheduling cycle, for example
// preemption for another pod, retries.
func TestNominationChangeOfPodItself(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	withNomination := func(nodeName string) *v1.Pod {
		p := pod.DeepCopy()
		p.Status.NominatedNodeName = nodeName
		return p
	}

	for name, tc := range map[string]struct {
		change    func(t *testing.T, q *PriorityQueue)
		wantMoved bool
	}{
		"cleared-by-update": {
			change: func(t *testing.T, q *PriorityQueue) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNomination(""))
			},
			wantMoved: true,
		},
		"changed-by-update": {
			change: func(t *testing.T, q *PriorityQueue) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNomination("node2"))
			},
			wantMoved: true,
		},
		"cleared-by-own-cycle": {
			change: func(t *testing.T, q *PriorityQueue) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustPodInfo(t, pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride})
			},
		},
		"nominated-elsewhere-by-own-cycle": {
			change: func(t *testing.T, q *PriorityQueue) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustPodInfo(t, pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: "node2"})
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			m := makeEmptyQueueingHintMapPerProfile()
			m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod}, WithQueueingHintMapPerProfile(m))
			q.Add(ctx, pod)
			if _, err := q.Pop(logger); err != nil {
				t.Fatalf("Pop: %v", err)
			}
			if err := q.AddUnschedulablePodIfNotPresent(ctx, q.newQueuedPodInfo(ctx, pod, "fooPlugin"), q.SchedulingCycle()); err != nil {
				t.Fatalf("AddUnschedulablePodIfNotPresent: %v", err)
			}

			tc.change(t, q)
			if moved := getUnschedulablePod(q, pod) == nil; moved != tc.wantMoved {
				t.Errorf("Pod moved: got %v, want %v", moved, tc.wantMoved)
			}
		})
	}
}
