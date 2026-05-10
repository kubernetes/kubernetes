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
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
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
