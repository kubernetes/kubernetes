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
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type invalidStateData struct{}

func (invalidStateData) Clone() fwk.StateData { return invalidStateData{} }

func TestPodGroupPreemptionState(t *testing.T) {
	makeDV := func(name string, nodePods map[string]int) *DomainVictim {
		var pods []fwk.PodInfo
		affected := make(map[string]fwk.NodeInfo)
		for n, count := range nodePods {
			ni := framework.NewNodeInfo()
			ni.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: n}})
			affected[n] = ni
			for i := range count {
				podName := fmt.Sprintf("%s-%s-%d", name, n, i)
				pod := st.MakePod().Name(podName).UID(podName).Priority(10).Node(n).Obj()
				pi, err := framework.NewPodInfo(pod)
				if err != nil {
					t.Fatalf("NewPodInfo failed: %v", err)
				}
				pods = append(pods, pi)
			}
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

	multiNodePG := makeDV("pg-multi", map[string]int{"node1": 1, "node2": 1})
	singleNodePod := makeDV("pod-node1", map[string]int{"node1": 1})
	sameNodeMultiPodPG := makeDV("pg-same-node", map[string]int{"node1": 3})

	tests := []struct {
		name                   string
		potentialVictims       []fwk.PreemptionVictim
		wantInitialCounts      map[string]int
		wantInitialPodCounts   map[string][]int
		condemnVictims         []fwk.PreemptionVictim
		wantAfterCondemnCounts map[string]int
		revertVictims          []fwk.PreemptionVictim
		wantAfterRevertCounts  map[string]int
	}{
		{
			name: "multi-node and single-node victims sorted by importance: condemn and revert",
			// multiNodePG (2 pods) is more important than singleNodePod (1 pod) of equal priority,
			// so NewPodGroupPreemptionState sorts it first despite singleNodePod appearing later in potentialVictims.
			potentialVictims: []fwk.PreemptionVictim{
				&victimWithPDBViolations{Victim: multiNodePG},
				&victimWithPDBViolations{Victim: singleNodePod},
			},
			wantInitialCounts:      map[string]int{"node1": 2, "node2": 1, "node3": 0},
			wantInitialPodCounts:   map[string][]int{"node1": {2, 1}, "node2": {2}},
			condemnVictims:         []fwk.PreemptionVictim{&victimWithPDBViolations{Victim: multiNodePG}},
			wantAfterCondemnCounts: map[string]int{"node1": 1, "node2": 0},
			revertVictims:          []fwk.PreemptionVictim{&victimWithPDBViolations{Victim: multiNodePG}},
			wantAfterRevertCounts:  map[string]int{"node1": 2, "node2": 1},
		},
		{
			name: "multi-pod victim on a single node is deduplicated",
			potentialVictims: []fwk.PreemptionVictim{
				&victimWithPDBViolations{Victim: singleNodePod},
				&victimWithPDBViolations{Victim: sameNodeMultiPodPG},
			},
			wantInitialCounts:      map[string]int{"node1": 2},
			wantInitialPodCounts:   map[string][]int{"node1": {3, 1}},
			condemnVictims:         []fwk.PreemptionVictim{&victimWithPDBViolations{Victim: sameNodeMultiPodPG}},
			wantAfterCondemnCounts: map[string]int{"node1": 1},
			revertVictims:          []fwk.PreemptionVictim{&victimWithPDBViolations{Victim: sameNodeMultiPodPG}},
			wantAfterRevertCounts:  map[string]int{"node1": 2},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			state := NewPodGroupPreemptionState(tc.potentialVictims)
			// Map domain victims in tc.condemnVictims / tc.revertVictims to the actual PreemptionVictim instances in tc.potentialVictims.
			findInPotential := func(targets []fwk.PreemptionVictim) []fwk.PreemptionVictim {
				var res []fwk.PreemptionVictim
				for _, target := range targets {
					for _, pv := range tc.potentialVictims {
						if pv.(*victimWithPDBViolations).Victim == target.(*victimWithPDBViolations).Victim {
							res = append(res, pv)
						}
					}
				}
				return res
			}

			for node, want := range tc.wantInitialCounts {
				if got := len(state.SurvivingVictimsOnNode(node)); got != want {
					t.Fatalf("SurvivingVictimsOnNode(%q) initial count = %d, want %d", node, got, want)
				}
			}
			for node, wantPodCounts := range tc.wantInitialPodCounts {
				var gotPodCounts []int
				for _, v := range state.SurvivingVictimsOnNode(node) {
					gotPodCounts = append(gotPodCounts, len(v.Pods()))
				}
				if diff := cmp.Diff(wantPodCounts, gotPodCounts); diff != "" {
					t.Fatalf("SurvivingVictimsOnNode(%q) pod counts diff (-want +got):\n%s", node, diff)
				}
			}

			state.AddCondemned(findInPotential(tc.condemnVictims))
			for node, want := range tc.wantAfterCondemnCounts {
				if got := len(state.SurvivingVictimsOnNode(node)); got != want {
					t.Fatalf("SurvivingVictimsOnNode(%q) count after AddCondemned = %d, want %d", node, got, want)
				}
			}

			state.RemoveCondemned(findInPotential(tc.revertVictims))
			for node, want := range tc.wantAfterRevertCounts {
				if got := len(state.SurvivingVictimsOnNode(node)); got != want {
					t.Fatalf("SurvivingVictimsOnNode(%q) count after RemoveCondemned = %d, want %d", node, got, want)
				}
			}
		})
	}

	t.Run("Clone", func(t *testing.T) {
		var nilState *PodGroupPreemptionState
		if got := nilState.Clone(); got != nil {
			t.Fatalf("expected nil from cloning nil PodGroupPreemptionState, got %v", got)
		}

		state := NewPodGroupPreemptionState([]fwk.PreemptionVictim{
			&victimWithPDBViolations{Victim: multiNodePG},
			&victimWithPDBViolations{Victim: singleNodePod},
		})
		state.AddCondemned([]fwk.PreemptionVictim{state.SurvivingVictimsOnNode("node2")[0]})

		cloned := state.Clone().(*PodGroupPreemptionState)
		cloned.AddCondemned([]fwk.PreemptionVictim{cloned.SurvivingVictimsOnNode("node1")[0]})
		if got := len(cloned.SurvivingVictimsOnNode("node1")); got != 0 {
			t.Fatalf("expected 0 surviving victims on node1 in cloned state, got %d", got)
		}
		if got := len(state.SurvivingVictimsOnNode("node1")); got != 1 {
			t.Fatalf("expected 1 surviving victim on node1 in original state after mutating clone, got %d", got)
		}
	})

	t.Run("GetPodGroupPreemptionState", func(t *testing.T) {
		rootPreemptionState := NewPodGroupPreemptionState([]fwk.PreemptionVictim{
			&victimWithPDBViolations{Victim: multiNodePG},
		})
		// Ensure nil PodGroupCycleState is a no-op.
		rootPreemptionState.WriteToPodGroupCycleState(nil)

		getTests := []struct {
			name            string
			buildCycleState func() fwk.CycleState
			wantState       *PodGroupPreemptionState
			wantDistinct    bool
			wantErrSubstr   string
		}{
			{
				name: "nil CycleState returns nil",
				buildCycleState: func() fwk.CycleState {
					return nil
				},
				wantState: nil,
			},
			{
				name: "CycleState without PodGroupCycleState returns nil",
				buildCycleState: func() fwk.CycleState {
					return framework.NewCycleState()
				},
				wantState: nil,
			},
			{
				name: "nested hierarchy without PodGroupPreemptionState returns nil",
				buildCycleState: func() fwk.CycleState {
					rootPGState := framework.NewCycleState()
					placementState := framework.NewCycleState()
					placementState.SetPodGroupCycleState(rootPGState)
					childPGState := framework.NewCycleState()
					childPGState.SetPlacementCycleState(placementState)
					podState := framework.NewCycleState()
					podState.SetPodGroupCycleState(childPGState)
					return podState
				},
				wantState: nil,
			},
			{
				name: "nested CompositePodGroup -> Placement -> ChildPodGroup -> Pod hierarchy returns root state",
				buildCycleState: func() fwk.CycleState {
					rootPGState := framework.NewCycleState()
					rootPreemptionState.WriteToPodGroupCycleState(rootPGState)

					placementState := framework.NewCycleState()
					placementState.SetPodGroupCycleState(rootPGState)

					childPGState := framework.NewCycleState()
					childPGState.SetPlacementCycleState(placementState)

					podState := framework.NewCycleState()
					podState.SetPodGroupCycleState(childPGState)
					return podState
				},
				wantState: rootPreemptionState,
			},
			{
				name: "cloning rootPGState produces an isolated PodGroupPreemptionState clone",
				buildCycleState: func() fwk.CycleState {
					rootPGState := framework.NewCycleState()
					rootPreemptionState.WriteToPodGroupCycleState(rootPGState)
					clonedRootPGState := rootPGState.Clone().(*framework.CycleState)
					clonedPodState := framework.NewCycleState()
					clonedPodState.SetPodGroupCycleState(clonedRootPGState)
					return clonedPodState
				},
				wantState:    rootPreemptionState,
				wantDistinct: true,
			},
			{
				name: "invalid state type assertion in GetPodGroupPreemptionState returns error",
				buildCycleState: func() fwk.CycleState {
					pgState := framework.NewCycleState()
					pgState.Write(PodGroupPreemptionStateKey, invalidStateData{})
					podState := framework.NewCycleState()
					podState.SetPodGroupCycleState(pgState)
					return podState
				},
				wantErrSubstr: "failed to convert preemption.invalidStateData to *PodGroupPreemptionState",
			},
		}

		for _, tc := range getTests {
			t.Run(tc.name, func(t *testing.T) {
				got, err := GetPodGroupPreemptionState(tc.buildCycleState())
				if tc.wantErrSubstr != "" {
					if err == nil || !strings.Contains(err.Error(), tc.wantErrSubstr) {
						t.Fatalf("GetPodGroupPreemptionState() error = %v, want substring %q", err, tc.wantErrSubstr)
					}
					return
				}
				if err != nil {
					t.Fatalf("unexpected error from GetPodGroupPreemptionState: %v", err)
				}
				if tc.wantDistinct {
					if got == nil || got == tc.wantState {
						t.Fatalf("expected distinct non-nil PodGroupPreemptionState clone, got %v", got)
					}
					return
				}
				if got != tc.wantState {
					t.Fatalf("GetPodGroupPreemptionState() = %v, want %v", got, tc.wantState)
				}
			})
		}
	})
}
