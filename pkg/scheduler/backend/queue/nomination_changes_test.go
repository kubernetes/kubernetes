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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
