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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func TestPodGroupPreemptionState(t *testing.T) {
	makeMultiNodeDV := func(name string, nodes ...string) *DomainVictim {
		var pods []fwk.PodInfo
		affected := make(map[string]fwk.NodeInfo)
		for _, n := range nodes {
			pod := st.MakePod().Name(name + "-" + n).UID(name + "-" + n).Priority(10).Node(n).Obj()
			pi, _ := framework.NewPodInfo(pod)
			pods = append(pods, pi)
			ni := framework.NewNodeInfo()
			ni.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
			affected[n] = ni
		}
		v, err := NewVictim(pods, 10, fwk.PodGroupKeyType)
		if err != nil {
			t.Fatalf("NewVictim failed: %v", err)
		}
		return &DomainVictim{
			Victim:        v,
			affectedNodes: affected,
		}
	}

	multiNodePG := makeMultiNodeDV("pg-multi", "node1", "node2")
	singleNodePod := makeMultiNodeDV("pod-node1", "node1")

	state := NewPodGroupPreemptionState([]fwk.PreemptionVictim{
		&victimWithPDBViolations{Victim: multiNodePG},
		&victimWithPDBViolations{Victim: singleNodePod},
	})

	if got := len(state.SurvivingVictimsOnNode("node1")); got != 2 {
		t.Fatalf("expected 2 surviving victims on node1 initially, got %d", got)
	}
	if got := len(state.SurvivingVictimsOnNode("node2")); got != 1 {
		t.Fatalf("expected 1 surviving victim on node2 initially, got %d", got)
	}

	// Condemn multiNodePG (e.g., pod1 placed on node1).
	v0 := state.SurvivingVictimsOnNode("node2")[0]
	state.AddCondemned([]fwk.PreemptionVictim{v0})
	if got := len(state.SurvivingVictimsOnNode("node1")); got != 1 {
		t.Fatalf("expected 1 surviving victim (singleNodePod) on node1 after condemning multiNodePG, got %d", got)
	}
	if got := len(state.SurvivingVictimsOnNode("node2")); got != 0 {
		t.Fatalf("expected 0 surviving victims on node2 after condemning multiNodePG on node1, got %d", got)
	}

	// Revert condemning -> multiNodePG should survive again on both nodes.
	state.RemoveCondemned([]fwk.PreemptionVictim{v0})
	if got := len(state.SurvivingVictimsOnNode("node1")); got != 2 {
		t.Fatalf("expected 2 surviving victims on node1 after rollback, got %d", got)
	}
	if got := len(state.SurvivingVictimsOnNode("node2")); got != 1 {
		t.Fatalf("expected 1 surviving victim on node2 after rollback, got %d", got)
	}

	// Verify GetPodGroupPreemptionState traverses nested CompositePodGroup -> Placement -> ChildPodGroup -> Pod hierarchy.
	rootPGState := framework.NewCycleState()
	state.WriteToPodGroupCycleState(rootPGState)

	placementState := framework.NewCycleState()
	placementState.SetPodGroupCycleState(rootPGState)

	childPGState := framework.NewCycleState()
	childPGState.SetPlacementCycleState(placementState)

	podState := framework.NewCycleState()
	podState.SetPodGroupCycleState(childPGState)

	got, err := GetPodGroupPreemptionState(podState)
	if err != nil {
		t.Fatalf("unexpected error from GetPodGroupPreemptionState: %v", err)
	}
	if got != state {
		t.Fatalf("expected GetPodGroupPreemptionState to traverse hierarchy and return root state, got %v", got)
	}
}
