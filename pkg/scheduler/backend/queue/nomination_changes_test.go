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
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// assumedOn returns a copy of pod which got assumed on nodeName.
func assumedOn(pod *v1.Pod, nodeName string) *v1.Pod {
	p := pod.DeepCopy()
	p.Spec.NodeName = nodeName
	return p
}

// TestNominationChanges checks that the queue lets pods retry which an ended
// nomination blocked, and only then. Without the DRAPreemption feature gate,
// and for a pod of a pod group, the queue leaves that to the event handlers.
// A pod with a scheduling group is only a pod of a pod group with the
// GenericWorkload feature gate.
func TestNominationChanges(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	podGroupPod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").PodGroupName("pg").Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()
	highPod := st.MakePod().Name("high").Namespace("ns").UID("high").Priority(highPriority + 1).Obj()

	for _, tc := range []struct {
		name      string
		change    func(t *testing.T, q *PriorityQueue, pod *v1.Pod)
		wantMoved bool
	}{
		{
			name: "re-added-unchanged",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustNewTestPodInfo(t, pod), &fwk.NominatingInfo{NominatingMode: fwk.ModeNoop})
			},
		},
		{
			name: "refreshed-by-update",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNominatedNodeName(pod, "node1"))
			},
		},
		// The pod objects don't show the nomination in memory yet.
		{
			name: "kept-by-update-of-stale-pod-object",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, withNominatedNodeName(pod, ""), withNominatedNodeName(pod, ""))
			},
		},
		{
			name: "changed-by-update",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				_, ctx := ktesting.NewTestContext(t)
				q.Update(ctx, pod, withNominatedNodeName(pod, "node2"))
			},
			wantMoved: true,
		},
		// Through fwk.PodNominator, without the rest of Update.
		{
			name: "cleared-by-nominated-pod-update",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				logger, _ := ktesting.NewTestContext(t)
				var nominator fwk.PodNominator = q
				nominator.UpdateNominatedPod(logger, pod, mustNewTestPodInfo(t, withNominatedNodeName(pod, "")))
			},
			wantMoved: true,
		},
		{
			name: "cleared-in-memory",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				logger, _ := ktesting.NewTestContext(t)
				q.AddNominatedPod(logger, mustNewTestPodInfo(t, pod), overrideNomination(""))
			},
			wantMoved: true,
		},
		{
			name: "deleted",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				logger, _ := ktesting.NewTestContext(t)
				q.Delete(logger, pod)
			},
			wantMoved: true,
		},
		// The pod uses the resources on its nominated node now.
		{
			name: "assumed-on-nominated-node",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				q.DeleteNominatedPodIfExists(assumedOn(pod, "node1"))
			},
		},
		{
			name: "assumed-on-other-node",
			change: func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
				q.DeleteNominatedPodIfExists(assumedOn(pod, "node2"))
			},
			wantMoved: true,
		},
	} {
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", tc.name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
				testNominationChange(t, pod, lowPod, highPod, tc.change, tc.wantMoved && draPreemption)
			})
		}
		for _, genericWorkload := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/scheduling-group/GenericWorkload=%v", tc.name, genericWorkload), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAPreemption:   true,
					features.GenericWorkload: genericWorkload,
				})
				testNominationChange(t, podGroupPod, lowPod, highPod, tc.change, tc.wantMoved && !genericWorkload)
			})
		}
	}
}

func testNominationChange(t *testing.T, pod, lowPod, highPod *v1.Pod, change func(t *testing.T, q *PriorityQueue, pod *v1.Pod), wantMoved bool) {
	_, ctx := ktesting.NewTestContext(t)
	q := newNominationChangesTestQueue(ctx, pod, lowPod, highPod)

	// lowPod and highPod were rejected while pod was nominated.
	for _, p := range []*v1.Pod{lowPod, highPod} {
		putPodToUnschedulable(t, ctx, q, p, "fooPlugin")
	}
	// A new nomination does not unblock anything.
	q.Add(ctx, pod)
	if getUnschedulablePod(q, lowPod) == nil {
		t.Fatal("Lower priority pod moved when the pod got nominated")
	}

	change(t, q, pod)
	if moved := getUnschedulablePod(q, lowPod) == nil; moved != wantMoved {
		t.Errorf("Lower priority pod moved: got %v, want %v", moved, wantMoved)
	}
	if getUnschedulablePod(q, highPod) == nil {
		t.Error("Higher priority pod moved, it did not have to respect the nomination")
	}
}

// newNominationChangesTestQueue returns a queue which knows the pods and
// where fooPlugin lets pods retry on any AssignedPod/Delete event.
func newNominationChangesTestQueue(ctx context.Context, pods ...*v1.Pod) *PriorityQueue {
	m := makeEmptyQueueingHintMapPerProfile()
	m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	objs := make([]runtime.Object, 0, len(pods))
	for _, pod := range pods {
		objs = append(objs, pod)
	}
	return NewTestQueueWithObjects(ctx, newDefaultQueueSort(), objs, WithQueueingHintMapPerProfile(m))
}

// TestNominationChangeOfPodItself checks that a pod whose nomination got
// changed retries, for example when the informer reports that preemption for
// another pod cleared it.
func TestNominationChangeOfPodItself(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()

	for _, tc := range []struct {
		name        string
		newNodeName string
	}{
		{name: "cleared-by-update", newNodeName: ""},
		{name: "changed-by-update", newNodeName: "node2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			q := newNominationChangesTestQueue(ctx, pod)
			putPodToUnschedulable(t, ctx, q, pod, "fooPlugin")

			q.Update(ctx, pod, withNominatedNodeName(pod, tc.newNodeName))
			if getUnschedulablePod(q, pod) != nil {
				t.Error("Pod does not retry although its nomination changed")
			}
		})
	}
}

// TestNominationChangeEvent checks which pod queueing hints get with the
// AssignedPod/Delete event for an ended nomination: an unassigned pod with
// the nominated node name of that nomination. That also holds when the pod
// got assumed on another node, where hints must not take it for a pod deleted
// from that node, and when the pod object shows another nominated node name.
func TestNominationChangeEvent(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()

	for _, tc := range []struct {
		name   string
		change func(ctx context.Context, q *PriorityQueue)
	}{
		{
			name: "assumed-on-other-node",
			change: func(ctx context.Context, q *PriorityQueue) {
				q.DeleteNominatedPodIfExists(assumedOn(pod, "node2"))
			},
		},
		{
			name: "cleared-by-update",
			change: func(ctx context.Context, q *PriorityQueue) {
				q.Update(ctx, pod, withNominatedNodeName(pod, ""))
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			var deletedPods []*v1.Pod
			m := makeEmptyQueueingHintMapPerProfile()
			m[""][framework.EventAssignedPodDelete] = []*QueueingHintFunction{{
				PluginName: "fooPlugin",
				QueueingHintFn: func(_ klog.Logger, _ *v1.Pod, oldObj, _ interface{}) (fwk.QueueingHint, error) {
					deletedPods = append(deletedPods, oldObj.(*v1.Pod))
					return fwk.Queue, nil
				},
			}}
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod}, WithQueueingHintMapPerProfile(m))
			putPodToUnschedulable(t, ctx, q, lowPod, "fooPlugin")
			q.Add(ctx, pod)

			tc.change(ctx, q)
			if len(deletedPods) != 1 {
				t.Fatalf("Queueing hint calls: got %d, want 1", len(deletedPods))
			}
			if nodeName := deletedPods[0].Spec.NodeName; nodeName != "" {
				t.Errorf("Node of the pod in the event %q, want none", nodeName)
			}
			if nodeName := deletedPods[0].Status.NominatedNodeName; nodeName != "node1" {
				t.Errorf("Nominated node of the pod in the event %q, want %q", nodeName, "node1")
			}
		})
	}
}

// TestDeleteNominatedPodIfExistsWithoutNomination checks that a pod which is
// not nominated, like most pods when they get assumed, does not have to wait
// for the lock of the scheduling queue.
func TestDeleteNominatedPodIfExistsWithoutNomination(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
	_, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod})

	q.lock.Lock()
	done := make(chan struct{})
	go func() {
		defer close(done)
		q.DeleteNominatedPodIfExists(pod)
	}()
	select {
	case <-done:
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Waited for the lock of the scheduling queue although the pod is not nominated")
	}
	q.lock.Unlock()
	<-done
}

// TestNominationChangeHandler checks that the handler sees every change of a
// nominated node and that the events which it returns let pods with lower or
// equal priority retry, also when the pod got assumed on its nominated node.
// It doesn't see the changes of a pod of a pod group.
func TestNominationChangeHandler(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAPreemption: true,
		// Makes podGroupPod a pod of a pod group.
		features.GenericWorkload: true,
	})
	logger, ctx := ktesting.NewTestContext(t)
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()
	assumed := pod.DeepCopy()
	assumed.Spec.NodeName = "node1"
	podGroupPod := st.MakePod().Name("pg-pod").Namespace("ns").UID("pg-pod").Priority(highPriority).NominatedNodeName("node1").PodGroupName("pg").Obj()
	lowPod := st.MakePod().Name("low").Namespace("ns").UID("low").Priority(lowPriority).Obj()
	highPod := st.MakePod().Name("high").Namespace("ns").UID("high").Priority(highPriority + 1).Obj()
	claimUpdate := fwk.ClusterEvent{Resource: fwk.ResourceClaim, ActionType: fwk.Update}

	m := makeEmptyQueueingHintMapPerProfile()
	m[""][claimUpdate] = []*QueueingHintFunction{{PluginName: "fooPlugin", QueueingHintFn: queueHintReturnQueue}}
	var calls []string
	q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod, lowPod, highPod, podGroupPod},
		WithQueueingHintMapPerProfile(m),
		WithNominationChangeHandler(func(_ klog.Logger, pod *v1.Pod, oldNodeName, newNodeName string) []framework.NominationEvent {
			calls = append(calls, fmt.Sprintf("%s %s->%s assigned=%s", pod.Name, oldNodeName, newNodeName, pod.Spec.NodeName))
			if oldNodeName == "" {
				return nil
			}
			return []framework.NominationEvent{{Event: claimUpdate}}
		}))
	for _, p := range []*v1.Pod{lowPod, highPod} {
		putPodToUnschedulable(t, ctx, q, p, "fooPlugin")
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

	q.Add(ctx, podGroupPod)
	q.AddNominatedPod(logger, mustNewTestPodInfo(t, podGroupPod), overrideNomination("node2"))
	q.DeleteNominatedPodIfExists(assumedOn(podGroupPod, "node3"))

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
	putPodToUnschedulable(t, ctx, q, lowPod, "fooPlugin")
	q.Add(ctx, pod)
	podInfo := mustNewTestPodInfo(t, pod)

	// Neither keeping the nomination after a failure nor an informer update
	// is a new nomination.
	q.AddNominatedPod(logger, podInfo, nil)
	updated := pod.DeepCopy()
	updated.Labels = map[string]string{"updated": "true"}
	q.Update(ctx, pod, updated)
	q.AddNominatedPod(logger, podInfo, overrideNomination("node1"))
	if getUnschedulablePod(q, lowPod) == nil {
		t.Error("Lower priority pod moved, but the pod keeps occupying its node")
	}
	want := []string{"p ->node1", "p node1->node1"}
	if diff := cmp.Diff(want, calls); diff != "" {
		t.Errorf("Unexpected handler calls (-want, +got):\n%s", diff)
	}
}
