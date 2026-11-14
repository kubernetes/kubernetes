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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// withNominatedNodeName returns a copy of pod which shows nodeName as its
// nominated node name.
func withNominatedNodeName(pod *v1.Pod, nodeName string) *v1.Pod {
	p := pod.DeepCopy()
	p.Status.NominatedNodeName = nodeName
	return p
}

func overrideNomination(nodeName string) *fwk.NominatingInfo {
	return &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName}
}

// nominatedInMemory returns the node which the pod is nominated to in memory.
func nominatedInMemory(q *PriorityQueue, uid types.UID) string {
	q.nominator.nLock.RLock()
	defer q.nominator.nLock.RUnlock()
	return q.nominator.nominatedPodToNode[uid]
}

// TestNominationOfRecreatedPod checks that with the DRAPreemption feature
// gate, a pod does not get nominated if the pod lister has a pod with the
// same name and a different UID. Such a pod was deleted and created again,
// and its deletion does not remove a nomination that gets added afterwards.
func TestNominationOfRecreatedPod(t *testing.T) {
	oldPod := st.MakePod().Name("p").Namespace("ns").UID("old").Priority(highPriority).Obj()
	newPod := st.MakePod().Name("p").Namespace("ns").UID("new").Priority(highPriority).Obj()

	for _, draPreemption := range []bool{false, true} {
		t.Run(fmt.Sprintf("DRAPreemption=%v", draPreemption), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
			logger, ctx := ktesting.NewTestContext(t)
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{newPod})

			q.AddNominatedPod(logger, mustNewTestPodInfo(t, oldPod), overrideNomination("node1"))
			want := "node1"
			if draPreemption {
				want = ""
			}
			if nodeName := nominatedInMemory(q, oldPod.UID); nodeName != want {
				t.Errorf("Nominated node of the deleted pod %q, want %q", nodeName, want)
			}
		})
	}
}

// TestNominationOfInFlightPod checks that with the DRAPreemption feature
// gate, a changed nominated node name reaches the nominator while the pod is
// in a scheduling cycle, and that NominatedPodsForNode returns the pod object
// that the pod was popped with. Without the feature gate, updates of in-flight
// pods don't reach the nominator, and NominatedPodsForNode returns a pod with
// only the name, namespace and UID.
func TestNominationOfInFlightPod(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).NominatedNodeName("node1").Obj()

	for _, draPreemption := range []bool{false, true} {
		t.Run(fmt.Sprintf("DRAPreemption=%v", draPreemption), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAPreemption, draPreemption)
			logger, ctx := ktesting.NewTestContext(t)
			q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod})
			q.Add(ctx, pod)
			if _, err := q.Pop(logger); err != nil {
				t.Fatalf("Pop: %v", err)
			}

			nominatedPods := q.NominatedPodsForNode(logger, "node1")
			if len(nominatedPods) != 1 {
				t.Fatalf("Nominated pods of node1: got %d, want 1", len(nominatedPods))
			}
			wantPriority := int32(0)
			if draPreemption {
				wantPriority = highPriority
			}
			if priority := corev1helpers.PodPriority(nominatedPods[0].GetPod()); priority != wantPriority {
				t.Errorf("Priority of the nominated in-flight pod: got %d, want %d", priority, wantPriority)
			}

			// An update that does not change the nominated node name.
			labeled := pod.DeepCopy()
			labeled.Labels = map[string]string{"foo": "bar"}
			q.Update(ctx, pod, labeled)
			if nodeName := nominatedInMemory(q, pod.UID); nodeName != "node1" {
				t.Errorf("Nominated node after an unrelated update %q, want %q", nodeName, "node1")
			}

			// Preemption for another pod cleared the nominated node name.
			q.Update(ctx, labeled, withNominatedNodeName(labeled, ""))
			want := "node1"
			if draPreemption {
				want = ""
			}
			if nodeName := nominatedInMemory(q, pod.UID); nodeName != want {
				t.Errorf("Nominated node after the nominated node name got cleared %q, want %q", nodeName, want)
			}
		})
	}
}

// TestNominationsInMemory checks that with the DRAPreemption feature gate, a
// nomination in memory is kept while pod objects tell nothing new about the
// nominated node name, like those which don't show the nomination yet. A pod
// object whose nominated node name changed still wins, and so does any pod
// object when the pod is not nominated in memory. Without the feature gate,
// and for pods of pod groups, pod objects win unless neither the old nor the
// new one has a nominated node name. A pod with a scheduling group is only a
// pod of a pod group with the GenericWorkload feature gate.
func TestNominationsInMemory(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	podGroupPod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).PodGroupName("pg").Obj()
	type step struct {
		do func(t *testing.T, q *PriorityQueue, pod *v1.Pod)
		// want and wantWithoutDRAPreemption are the nominated node in memory
		// after the step.
		want, wantWithoutDRAPreemption string
	}
	// nominate is what handleSchedulingFailure does with a nomination from
	// PostFilter, or without one when it clears the nomination. podNodeName
	// is what the pod object shows.
	nominate := func(podNodeName, nodeName string) func(*testing.T, *PriorityQueue, *v1.Pod) {
		return func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
			logger, _ := ktesting.NewTestContext(t)
			q.AddNominatedPod(logger, mustNewTestPodInfo(t, withNominatedNodeName(pod, podNodeName)), overrideNomination(nodeName))
		}
	}
	// keep is what handleSchedulingFailure does when PostFilter did not run
	// or left the nomination alone.
	keep := func(podNodeName string) func(*testing.T, *PriorityQueue, *v1.Pod) {
		return func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
			logger, _ := ktesting.NewTestContext(t)
			q.AddNominatedPod(logger, mustNewTestPodInfo(t, withNominatedNodeName(pod, podNodeName)), nil)
		}
	}
	update := func(oldNodeName, newNodeName string) func(*testing.T, *PriorityQueue, *v1.Pod) {
		return func(t *testing.T, q *PriorityQueue, pod *v1.Pod) {
			logger, _ := ktesting.NewTestContext(t)
			q.UpdateNominatedPod(logger, withNominatedNodeName(pod, oldNodeName), mustNewTestPodInfo(t, withNominatedNodeName(pod, newNodeName)))
		}
	}

	for _, tc := range []struct {
		name string
		pod  *v1.Pod
		// genericWorkload enables the GenericWorkload feature gate.
		genericWorkload bool
		steps           []step
	}{
		{
			name: "kept-without-new-nomination",
			pod:  pod,
			steps: []step{
				{do: nominate("", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: keep(""), want: "node1", wantWithoutDRAPreemption: ""},
			},
		},
		{
			name: "kept-without-new-nomination-after-renomination",
			pod:  pod,
			steps: []step{
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: keep("node0"), want: "node1", wantWithoutDRAPreemption: "node0"},
			},
		},
		{
			name: "kept-on-update-without-nominated-node-name",
			pod:  pod,
			steps: []step{
				{do: nominate("", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: update("", ""), want: "node1", wantWithoutDRAPreemption: "node1"},
			},
		},
		{
			name: "kept-on-update-with-unchanged-nominated-node-name",
			pod:  pod,
			steps: []step{
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: update("node0", "node0"), want: "node1", wantWithoutDRAPreemption: "node0"},
			},
		},
		{
			name: "changed-nominated-node-name-wins",
			pod:  pod,
			steps: []step{
				{do: nominate("", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				// The status patch of the scheduler got applied.
				{do: update("", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				// Someone else changed and then cleared it.
				{do: update("node1", "node3"), want: "node3", wantWithoutDRAPreemption: "node3"},
				{do: update("node3", ""), want: "", wantWithoutDRAPreemption: ""},
			},
		},
		// Only a nomination can be kept: after the scheduler cleared it in
		// memory, a pod object which still shows it brings it back until the
		// pod object shows that the status patch got applied.
		{
			name: "pod-object-wins-without-nomination-in-memory",
			pod:  pod,
			steps: []step{
				{do: nominate("node1", ""), want: "", wantWithoutDRAPreemption: ""},
				{do: keep("node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: nominate("node1", ""), want: "", wantWithoutDRAPreemption: ""},
				{do: update("node1", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: update("node1", ""), want: "", wantWithoutDRAPreemption: ""},
			},
		},
		{
			name:            "pod-group-pod-follows-pod-objects",
			pod:             podGroupPod,
			genericWorkload: true,
			steps: []step{
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: keep("node0"), want: "node0", wantWithoutDRAPreemption: "node0"},
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: update("node0", "node0"), want: "node0", wantWithoutDRAPreemption: "node0"},
			},
		},
		// Without GenericWorkload, the pod gets scheduled like any other pod.
		{
			name: "scheduling-group-without-GenericWorkload",
			pod:  podGroupPod,
			steps: []step{
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: keep("node0"), want: "node1", wantWithoutDRAPreemption: "node0"},
				{do: nominate("node0", "node1"), want: "node1", wantWithoutDRAPreemption: "node1"},
				{do: update("node0", "node0"), want: "node1", wantWithoutDRAPreemption: "node0"},
			},
		},
	} {
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", tc.name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAPreemption:   draPreemption,
					features.GenericWorkload: tc.genericWorkload,
				})
				_, ctx := ktesting.NewTestContext(t)
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{tc.pod})
				for i, step := range tc.steps {
					step.do(t, q, tc.pod)
					want := step.want
					if !draPreemption {
						want = step.wantWithoutDRAPreemption
					}
					if nodeName := nominatedInMemory(q, tc.pod.UID); nodeName != want {
						t.Errorf("Step %d: nominated node %q, want %q", i, nodeName, want)
					}
				}
			})
		}
	}
}

// TestNominatedNodeName checks that with the DRAPreemption feature gate, the
// nominated node of a pod is its nomination in memory, also while the pod
// object doesn't show it yet. Without the gate, and for a pod of a pod group,
// it is what the pod object shows. A pod with a scheduling group is only a
// pod of a pod group with the GenericWorkload feature gate.
func TestNominatedNodeName(t *testing.T) {
	pod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).Obj()
	podGroupPod := st.MakePod().Name("p").Namespace("ns").UID("p").Priority(highPriority).PodGroupName("pg").Obj()
	for _, tc := range []struct {
		name string
		// pod defaults to a pod without a scheduling group.
		pod *v1.Pod
		// genericWorkload enables the GenericWorkload feature gate.
		genericWorkload bool
		// podNodeName is what the pod object shows.
		podNodeName string
		// added is true if the queue got the pod object from the informer.
		added bool
		// nominations is what the scheduler set in memory, in this order.
		nominations []string
		// want and wantWithoutDRAPreemption are the nominated node of the pod.
		want, wantWithoutDRAPreemption string
	}{
		{name: "not-nominated"},
		{
			name:                     "nominated-through-api",
			podNodeName:              "node1",
			added:                    true,
			want:                     "node1",
			wantWithoutDRAPreemption: "node1",
		},
		{
			name:        "nominated-with-stale-pod-object",
			nominations: []string{"node1"},
			want:        "node1",
		},
		{
			name:                     "nominated-elsewhere-with-stale-pod-object",
			nominations:              []string{"node1", "node2"},
			podNodeName:              "node1",
			want:                     "node2",
			wantWithoutDRAPreemption: "node1",
		},
		{
			name:                     "cleared-with-stale-pod-object",
			nominations:              []string{"node1", ""},
			podNodeName:              "node1",
			wantWithoutDRAPreemption: "node1",
		},
		{
			name:                     "pod-group-pod-nominated-elsewhere-with-stale-pod-object",
			pod:                      podGroupPod,
			genericWorkload:          true,
			nominations:              []string{"node1", "node2"},
			podNodeName:              "node1",
			want:                     "node1",
			wantWithoutDRAPreemption: "node1",
		},
		// Without GenericWorkload, the pod gets scheduled like any other pod.
		{
			name:                     "scheduling-group-without-GenericWorkload-nominated-elsewhere-with-stale-pod-object",
			pod:                      podGroupPod,
			nominations:              []string{"node1", "node2"},
			podNodeName:              "node1",
			want:                     "node2",
			wantWithoutDRAPreemption: "node1",
		},
	} {
		for _, draPreemption := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/DRAPreemption=%v", tc.name, draPreemption), func(t *testing.T) {
				featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
					features.DRAPreemption:   draPreemption,
					features.GenericWorkload: tc.genericWorkload,
				})
				logger, ctx := ktesting.NewTestContext(t)
				pod := pod
				if tc.pod != nil {
					pod = tc.pod
				}
				q := NewTestQueueWithObjects(ctx, newDefaultQueueSort(), []runtime.Object{pod})
				if tc.added {
					q.Add(ctx, withNominatedNodeName(pod, tc.podNodeName))
				}
				for _, nodeName := range tc.nominations {
					q.AddNominatedPod(logger, mustNewTestPodInfo(t, pod), overrideNomination(nodeName))
				}
				want := tc.want
				if !draPreemption {
					want = tc.wantWithoutDRAPreemption
				}
				if nodeName := q.NominatedNodeName(withNominatedNodeName(pod, tc.podNodeName)); nodeName != want {
					t.Errorf("Nominated node %q, want %q", nodeName, want)
				}
			})
		}
	}
}
