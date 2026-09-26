/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"sort"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPodGroupState_AssumeForget(t *testing.T) {
	pgs := newPodGroupState()
	pod := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()

	pgs.addPod(pod)
	if pgs.AssumedPods().Has(pod.UID) {
		t.Fatal("AssumedPods should be initially empty")
	}
	if !pgs.unscheduledPods.Has(pod.UID) {
		t.Fatal("Pod should be initially in UnscheduledPods")
	}

	pgs.assumePod(pod)
	if !pgs.AssumedPods().Has(pod.UID) {
		t.Fatal("Pod should be in AssumedPods after AssumePod")
	}
	if pgs.unscheduledPods.Has(pod.UID) {
		t.Fatal("UnscheduledPods should be empty after AssumePod")
	}

	pgs.forgetPod(pod.UID)
	if pgs.AssumedPods().Has(pod.UID) {
		t.Fatal("Pod should not be in AssumedPods after ForgetPod")
	}
	if !pgs.unscheduledPods.Has(pod.UID) {
		t.Fatal("Pod should be in UnscheduledPods after ForgetPod")
	}
}

func TestPodGroupState_Clone(t *testing.T) {
	pgs := newPodGroupState()

	pod1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		PodGroupName("pg").Obj()
	pod2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").
		PodGroupName("pg").Obj()

	pgs.addPod(pod1)
	pgs.addPod(pod2)
	pgs.assumePod(pod2)

	snap := pgs.snapshot()

	// Clone has the same generation.
	if snap.generation != pgs.generation {
		t.Errorf("expected clone generation %d, got %d", pgs.generation, snap.generation)
	}

	// Clone contains both pods.
	if !snap.AllPods().Has(pod1.UID) || !snap.AllPods().Has(pod2.UID) {
		t.Error("expected both pods in clone's AllPods")
	}

	// Clone preserves pod1 as unscheduled.
	if _, ok := snap.UnscheduledPods()[pod1.Name]; !ok {
		t.Error("expected pod1 in clone's UnscheduledPods")
	}

	// Clone preserves pod2 as assumed.
	if !snap.AssumedPods().Has(pod2.UID) {
		t.Error("expected pod2 in clone's AssumedPods")
	}

	// Mutating the clone does not affect the original.
	snap.assumePod(pod1)
	if _, ok := pgs.assumedPods[pod1.UID]; ok {
		t.Error("mutation to clone should not affect original's assumedPods")
	}

	// Mutating the original does not affect the clone.
	pod3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").
		PodGroupName("pg").Obj()
	pgs.addPod(pod3)
	if snap.AllPods().Has(pod3.UID) {
		t.Error("mutation to original should not affect clone's AllPods")
	}
}

func TestPodGroupState_PodCounts(t *testing.T) {
	pgs := newPodGroupState()
	pod1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").
		PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").
		PodGroupName("pg1").Obj()
	pod3 := st.MakePod().Namespace("ns1").Name("p3").UID("p3").Node("node1").
		PodGroupName("pg1").Obj()
	pod4 := st.MakePod().Namespace("ns1").Name("p4").UID("p4").
		PodGroupName("pg1").Obj()

	if count := pgs.AllPodsCount(); count != 0 {
		t.Errorf("Expected AllPodsCount to be 0, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 0 {
		t.Errorf("Expected ScheduledPodsCount to be 0, got %d", count)
	}

	pgs.addPod(pod1)
	pgs.addPod(pod2)
	pgs.addPod(pod3)

	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 1 {
		t.Errorf("Expected ScheduledPodsCount to be 1, got %d", count)
	}

	// Assuming a pod should move it from unscheduled to assumed, increasing the count of scheduled pods.
	pgs.assumePod(pod1)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}

	// Assuming a pod that is already scheduled should not change the counts.
	pgs.assumePod(pod3)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}

	// Assuming a pod that is not in the state should not change the counts.
	pgs.assumePod(pod4)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}

	// Forgetting a pod that is already scheduled should not change the counts.
	pgs.forgetPod(pod3.UID)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}

	// Forgetting a pod that is in the assumed state should move it back to unscheduled,
	// decreasing the count of scheduled pods.
	pgs.forgetPod(pod1.UID)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 1 {
		t.Errorf("Expected ScheduledPodsCount to be 1, got %d", count)
	}

	// Forgetting a pod that is not assumed should not change the counts.
	pgs.forgetPod(pod1.UID)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 1 {
		t.Errorf("Expected ScheduledPodsCount to be 1, got %d", count)
	}

	// Assuming a pod again should move it back to assumed, increasing the count of scheduled pods.
	pgs.assumePod(pod2)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}

	// Forgetting a pod that is not in the state should not change the counts.
	pgs.forgetPod(pod4.UID)
	if count := pgs.AllPodsCount(); count != 3 {
		t.Errorf("Expected AllPodsCount to be 3, got %d", count)
	}
	if count := pgs.ScheduledPodsCount(); count != 2 {
		t.Errorf("Expected ScheduledPodsCount to be 2, got %d", count)
	}
}

// TestPodGroupState_ScheduledPods tests that ScheduledPods returns pods that
// are currently either assumed or assigned altogether.
func TestPodGroupState_ScheduledPods(t *testing.T) {

	pgs := newPodGroupState()
	unscheduledPod := st.MakePod().Namespace("ns").Name("p1").UID("p1").
		PodGroupName("pg").Obj()
	assumedPod := st.MakePod().Namespace("ns").Name("p2").UID("p2").
		PodGroupName("pg").Obj()
	assignedPod := st.MakePod().Namespace("ns").Name("p3").UID("p3").Node("node1").
		PodGroupName("pg").Obj()

	pgs.addPod(assignedPod)
	pgs.addPod(unscheduledPod)
	pgs.addPod(assumedPod)

	// Simulate the scheduler assuming the pod on a node.
	assumedPodWithNodeName := assumedPod.DeepCopy()
	assumedPodWithNodeName.Spec.NodeName = "node2"

	pgs.assumePod(assumedPodWithNodeName)
	scheduledPods := pgs.ScheduledPods()

	snapshot := pgs.snapshot()
	pgs.assumePod(unscheduledPod)
	snapshotScheduledPods := snapshot.ScheduledPods()

	expectedScheduledPods := []*v1.Pod{assignedPod, assumedPodWithNodeName}

	if diff := cmp.Diff(expectedScheduledPods, scheduledPods); diff != "" {
		t.Errorf("unexpected ScheduledPods result (-want,+got):\n%s", diff)
	}

	if diff := cmp.Diff(expectedScheduledPods, snapshotScheduledPods); diff != "" {
		t.Errorf("unexpected snapshot ScheduledPods result (-want,+got):\n%s", diff)
	}
}

func TestCompositePodGroupState_Children(t *testing.T) {
	cpgs := newCompositePodGroupState()

	// 1. Initial state (children should be empty)
	if children := cpgs.GetChildren(); len(children) != 0 {
		t.Errorf("Expected no children initially, got %v", children)
	}

	// 2. Set children and test GetChildren
	childKey1 := fwk.PodGroupKey("ns1", "child1")
	childKey2 := fwk.CompositePodGroupKey("ns1", "child2")
	cpgs.addChild(childKey1)
	cpgs.addChild(childKey2)

	children := cpgs.GetChildren()
	var childrenStrs []string
	for _, child := range children {
		childrenStrs = append(childrenStrs, child.String())
	}
	sort.Strings(childrenStrs)
	expectedChildren := []string{"compositepodgroup/ns1/child2", "podgroup/ns1/child1"}
	if diff := cmp.Diff(expectedChildren, childrenStrs); diff != "" {
		t.Errorf("Unexpected children result (-want,+got):\n%s", diff)
	}
}

func TestPodGroupState_AssumedInThisCycleCount(t *testing.T) {
	pod1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj()
	unknownPod := st.MakePod().Namespace("ns1").Name("unknown").UID("unknown").PodGroupName("pg1").Obj()

	tests := []struct {
		name              string
		action            func(snap *podGroupStateSnapshot)
		expectedCount     int
		testSnapshotClone bool
	}{
		{
			name:          "initial snapshot has count 0",
			action:        func(snap *podGroupStateSnapshot) {},
			expectedCount: 0,
		},
		{
			name:          "assuming a pod in the snapshot increments count",
			action:        func(snap *podGroupStateSnapshot) { snap.assumePod(pod2) },
			expectedCount: 1,
		},
		{
			name: "forgetting the assumed pod decrements count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				snap.forgetPod(pod2.UID)
			},
			expectedCount: 0,
		},
		{
			name:          "assuming an unknown pod does not increment count (desync scenario)",
			action:        func(snap *podGroupStateSnapshot) { snap.assumePod(unknownPod) },
			expectedCount: 0,
		},
		{
			name:          "re-assuming an already assumed pod does not increment count (spurious assume)",
			action:        func(snap *podGroupStateSnapshot) { snap.assumePod(pod1) },
			expectedCount: 0,
		},
		{
			name:          "forgetting a pod assumed before the snapshot does not affect count",
			action:        func(snap *podGroupStateSnapshot) { snap.forgetPod(pod1.UID) },
			expectedCount: 0,
		},
		{
			name: "re-assuming a pod assumed during the snapshot does not increment count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				snap.assumePod(pod2)
			},
			expectedCount: 1,
		},
		{
			name:          "forgetting a pod that was never assumed does not affect count",
			action:        func(snap *podGroupStateSnapshot) { snap.forgetPod(pod2.UID) },
			expectedCount: 0,
		},
		{
			name:              "cloning a snapshot preserves assumedThisCycle count",
			action:            func(snap *podGroupStateSnapshot) { snap.assumePod(pod2) },
			expectedCount:     1,
			testSnapshotClone: true,
		},
		{
			name: "deletePod on assumed pod in snapshot decrements count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				snap.deletePod(pod2.UID)
			},
			expectedCount: 0,
		},
		{
			name: "addPod with node name on assumed pod in snapshot decrements count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				boundPod2 := pod2.DeepCopy()
				boundPod2.Spec.NodeName = "node1"
				snap.addPod(boundPod2)
			},
			expectedCount: 0,
		},
		{
			name: "updatePod assigning an assumed pod in snapshot decrements count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				boundPod2 := pod2.DeepCopy()
				boundPod2.Spec.NodeName = "node1"
				snap.updatePod(pod2, boundPod2)
			},
			expectedCount: 0,
		},
		{
			name: "addPod without node name on assumed pod in snapshot preserves count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				snap.addPod(pod2)
			},
			expectedCount: 1,
		},
		{
			name: "updatePod without assigning an assumed pod in snapshot preserves count",
			action: func(snap *podGroupStateSnapshot) {
				snap.assumePod(pod2)
				updatedPod2 := pod2.DeepCopy()
				updatedPod2.Labels = map[string]string{"foo": "bar"}
				snap.updatePod(pod2, updatedPod2)
			},
			expectedCount: 1,
		},
		{
			name: "deletePod on initial snapshot with nil assumedThisCycle does not panic",
			action: func(snap *podGroupStateSnapshot) {
				snap.deletePod(pod2.UID)
			},
			expectedCount: 0,
		},
		{
			name: "addPod with node name on initial snapshot with nil assumedThisCycle does not panic",
			action: func(snap *podGroupStateSnapshot) {
				boundPod2 := pod2.DeepCopy()
				boundPod2.Spec.NodeName = "node1"
				snap.addPod(boundPod2)
			},
			expectedCount: 0,
		},
		{
			name: "updatePod assigning pod on initial snapshot with nil assumedThisCycle does not panic",
			action: func(snap *podGroupStateSnapshot) {
				boundPod2 := pod2.DeepCopy()
				boundPod2.Spec.NodeName = "node1"
				snap.updatePod(pod2, boundPod2)
			},
			expectedCount: 0,
		},
		{
			name:          "forgetting a pod when no pod was ever added",
			action:        func(snap *podGroupStateSnapshot) { snap.forgetPod(unknownPod.UID) },
			expectedCount: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Set up a fresh podGroupState for every subtest
			pgs := newPodGroupState()
			pgs.addPod(pod1)
			pgs.addPod(pod2)

			// Live state should always return 0 for AssumedInThisCycleCount.
			pgs.assumePod(pod1)
			if diff := cmp.Diff(0, pgs.AssumedInThisCycleCount()); diff != "" {
				t.Fatalf("unexpected live podGroupState AssumedInThisCycleCount result (-want,+got):\n%s", diff)
			}

			// Spawn an isolated snapshot for this scenario
			snap := pgs.snapshot()

			tc.action(snap)

			if diff := cmp.Diff(tc.expectedCount, snap.AssumedInThisCycleCount()); diff != "" {
				t.Errorf("unexpected AssumedInThisCycleCount result (-want,+got):\n%s", diff)
			}

			if tc.testSnapshotClone {
				cloned := snap.Clone()
				if diff := cmp.Diff(tc.expectedCount, cloned.AssumedInThisCycleCount()); diff != "" {
					t.Errorf("unexpected AssumedInThisCycleCount in cloned snapshot (-want,+got):\n%s", diff)
				}
			}
		})
	}
}

func TestPodGroupState_EmptyGroupState(t *testing.T) {
	pgs := newPodGroupState()
	snap := pgs.snapshot()

	if diff := cmp.Diff(0, snap.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("unexpected AssumedInThisCycleCount on empty group state (-want,+got):\n%s", diff)
	}

	snap.forgetPod("non-existent-uid")
	if diff := cmp.Diff(0, snap.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("unexpected AssumedInThisCycleCount after forgetting pod on empty group state (-want,+got):\n%s", diff)
	}
}

func TestPodGroupStateSnapshot_CloneIndependence(t *testing.T) {
	pgs := newPodGroupState()
	pod1 := st.MakePod().Namespace("ns1").Name("p1").UID("p1").PodGroupName("pg1").Obj()
	pod2 := st.MakePod().Namespace("ns1").Name("p2").UID("p2").PodGroupName("pg1").Obj()
	pgs.addPod(pod1)
	pgs.addPod(pod2)

	pgs1 := pgs.snapshot()
	pgs1.assumePod(pod1)

	pgs2 := pgs1.Clone()

	if diff := cmp.Diff(1, pgs1.AssumedInThisCycleCount()); diff != "" {
		t.Fatalf("unexpected pgs1 initial count (-want,+got):\n%s", diff)
	}
	if diff := cmp.Diff(1, pgs2.AssumedInThisCycleCount()); diff != "" {
		t.Fatalf("unexpected pgs2 initial count (-want,+got):\n%s", diff)
	}

	pgs1.assumePod(pod2)
	if diff := cmp.Diff(2, pgs1.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("unexpected pgs1 count after assuming pod2 (-want,+got):\n%s", diff)
	}
	if diff := cmp.Diff(1, pgs2.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("pgs2 count should be unaffected by pgs1 mutation (-want,+got):\n%s", diff)
	}

	pgs2.forgetPod(pod1.UID)
	if diff := cmp.Diff(2, pgs1.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("pgs1 count should be unaffected by pgs2 mutation (-want,+got):\n%s", diff)
	}
	if diff := cmp.Diff(0, pgs2.AssumedInThisCycleCount()); diff != "" {
		t.Errorf("unexpected pgs2 count after forgetting pod1 (-want,+got):\n%s", diff)
	}
}
