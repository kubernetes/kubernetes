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
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

type invalidStateData struct{}

func (invalidStateData) Clone() fwk.StateData { return invalidStateData{} }

// customPreemptionVictim is a victim returned by a custom PreemptionManager: it reports its own
// PDB violations but does not expose the PDBs that protect its pods.
type customPreemptionVictim struct {
	Victim
	numPDBViolations int
}

func (v *customPreemptionVictim) NumPDBViolations() int { return v.numPDBViolations }

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
			// singleNodePod is more important than multiNodePG because it's appearing later in potential victims,
			// despite having fewer pods.
			potentialVictims: []fwk.PreemptionVictim{
				&victimWithPDBViolations{Victim: multiNodePG},
				&victimWithPDBViolations{Victim: singleNodePod},
			},
			wantInitialCounts:      map[string]int{"node1": 2, "node2": 1, "node3": 0},
			wantInitialPodCounts:   map[string][]int{"node1": {1, 2}, "node2": {2}},
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

func TestPodGroupPreemptionState_DynamicPDBTracking(t *testing.T) {
	appLabels := map[string]string{"app": "web"}
	dbLabels := map[string]string{"app": "db"}

	makePodVictim := func(name, node string, priority int32, lbls map[string]string) Victim {
		pod := st.MakePod().Name(name).UID(name).Namespace("default").Node(node).Priority(priority).Labels(lbls).Obj()
		pi, err := framework.NewPodInfo(pod)
		if err != nil {
			t.Fatalf("NewPodInfo failed: %v", err)
		}
		return NewPodVictim(pi)
	}

	makeGroupVictim := func(priority int32, pods ...*v1.Pod) Victim {
		podInfos := make([]fwk.PodInfo, len(pods))
		for i, p := range pods {
			pi, err := framework.NewPodInfo(p)
			if err != nil {
				t.Fatalf("NewPodInfo failed: %v", err)
			}
			podInfos[i] = pi
		}
		v, err := NewVictim(podInfos, priority, fwk.PodGroupKeyType)
		if err != nil {
			t.Fatalf("NewVictim failed: %v", err)
		}
		return v
	}

	// nodeVictimSet is the comparable form of the fwk.PreemptionVictimSet that VictimSet returns for a node:
	// Victims() as victimPods, each victim listed by the names of its pods, and NumPDBViolations().
	type nodeVictimSet struct {
		victimPods       [][]string
		numPDBViolations int
	}

	gotNodeVictimSets := func(state *PodGroupPreemptionState, nodes map[string]nodeVictimSet) map[string]nodeVictimSet {
		res := make(map[string]nodeVictimSet, len(nodes))
		for node := range nodes {
			set := state.VictimSet(state.SurvivingVictimsOnNode(node))
			got := nodeVictimSet{numPDBViolations: set.NumPDBViolations()}
			for _, v := range set.Victims() {
				var podNames []string
				for _, p := range v.Pods() {
					podNames = append(podNames, p.GetPod().Name)
				}
				got.victimPods = append(got.victimPods, podNames)
			}
			res[node] = got
		}
		return res
	}

	perVictimPDBViolations := func(victims []fwk.PreemptionVictim) []int {
		res := make([]int, len(victims))
		for i, v := range victims {
			res[i] = v.NumPDBViolations()
		}
		return res
	}

	tests := []struct {
		name             string
		victims          []Victim
		pdbs             []*policy.PodDisruptionBudget
		wantInitial      map[string]nodeVictimSet
		condemnNode      string
		wantAfterCondemn map[string]nodeVictimSet
	}{
		{
			name: "cross-node DisruptionsAllowed=1 updates dynamically on AddCondemned, Clone, and RemoveCondemned",
			victims: []Victim{
				makePodVictim("web-high", "node1", 50, appLabels),
				makePodVictim("web-low", "node2", 10, appLabels),
				makePodVictim("unprotected-mid", "node3", 30, nil),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("pdb-web").Namespace("default").MatchLabel("app", "web").DisruptionsAllowed(1).Obj(),
			},
			wantInitial: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"web-high"}}, numPDBViolations: 0},
				"node2": {victimPods: [][]string{{"web-low"}}, numPDBViolations: 0},
				"node3": {victimPods: [][]string{{"unprotected-mid"}}, numPDBViolations: 0},
			},
			condemnNode: "node2",
			wantAfterCondemn: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"web-high"}}, numPDBViolations: 1},
				"node2": {},
				"node3": {victimPods: [][]string{{"unprotected-mid"}}, numPDBViolations: 0},
			},
		},
		{
			name: "a node's PDB violations are computed for evicting all its victims against the remaining disruptions allowed",
			victims: []Victim{
				makePodVictim("high-unprotected", "node1", 50, nil),
				makePodVictim("mid-web", "node1", 30, appLabels),
				makePodVictim("low-web", "node1", 10, appLabels),
				makePodVictim("other-node-web", "node2", 20, appLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("pdb-web").Namespace("default").MatchLabel("app", "web").DisruptionsAllowed(1).Obj(),
			},
			wantInitial: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"high-unprotected"}, {"mid-web"}, {"low-web"}}, numPDBViolations: 1},
			},
			condemnNode: "node2",
			wantAfterCondemn: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"high-unprotected"}, {"mid-web"}, {"low-web"}}, numPDBViolations: 2},
			},
		},
		{
			name: "PodGroup victim with multiple pods consumes PDB disruptions equivalent to pod count when condemned",
			victims: []Victim{
				makeGroupVictim(20,
					st.MakePod().Name("pg-db-1").UID("pg-db-1").Namespace("default").Node("node1").Priority(20).Labels(dbLabels).Obj(),
					st.MakePod().Name("pg-db-2").UID("pg-db-2").Namespace("default").Node("node2").Priority(20).Labels(dbLabels).Obj(),
				),
				makePodVictim("single-db", "node3", 10, dbLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("pdb-db").Namespace("default").MatchLabel("app", "db").DisruptionsAllowed(2).Obj(),
			},
			wantInitial: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"pg-db-1", "pg-db-2"}}, numPDBViolations: 0},
				"node3": {victimPods: [][]string{{"single-db"}}, numPDBViolations: 0},
			},
			condemnNode: "node1",
			wantAfterCondemn: map[string]nodeVictimSet{
				"node1": {},
				"node2": {},
				"node3": {victimPods: [][]string{{"single-db"}}, numPDBViolations: 1},
			},
		},
		{
			name: "a multi-node PodGroup victim counts all its pods toward each node's PDB violations",
			victims: []Victim{
				makeGroupVictim(20,
					st.MakePod().Name("pg-web-1").UID("pg-web-1").Namespace("default").Node("node1").Priority(20).Labels(appLabels).Obj(),
					st.MakePod().Name("pg-web-2").UID("pg-web-2").Namespace("default").Node("node2").Priority(20).Labels(appLabels).Obj(),
				),
				makePodVictim("low-web-1", "node2", 10, appLabels),
				makePodVictim("low-web-2", "node2", 10, appLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("pdb-web").Namespace("default").MatchLabel("app", "web").DisruptionsAllowed(2).Obj(),
			},
			wantInitial: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"pg-web-1", "pg-web-2"}}, numPDBViolations: 0},
				"node2": {victimPods: [][]string{{"pg-web-1", "pg-web-2"}, {"low-web-1"}, {"low-web-2"}}, numPDBViolations: 2},
			},
			condemnNode: "node1",
			wantAfterCondemn: map[string]nodeVictimSet{
				"node1": {},
				"node2": {victimPods: [][]string{{"low-web-1"}, {"low-web-2"}}, numPDBViolations: 2},
			},
		},
		{
			name: "victims matching overlapping PDBs are evaluated from least to most important",
			victims: []Victim{
				makePodVictim("mid-both", "node1", 30, map[string]string{"app": "web", "tier": "frontend"}),
				makePodVictim("low-web", "node1", 10, appLabels),
				makePodVictim("other-web", "node2", 20, appLabels),
			},
			pdbs: []*policy.PodDisruptionBudget{
				st.MakePDB().Name("pdb-web").Namespace("default").MatchLabel("app", "web").DisruptionsAllowed(1).Obj(),
				st.MakePDB().Name("pdb-frontend").Namespace("default").MatchLabel("tier", "frontend").DisruptionsAllowed(0).Obj(),
			},
			// low-web consumes the only disruption of pdb-web, so only mid-both violates.
			wantInitial: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"mid-both"}, {"low-web"}}, numPDBViolations: 1},
				"node2": {victimPods: [][]string{{"other-web"}}, numPDBViolations: 0},
			},
			condemnNode: "node2",
			wantAfterCondemn: map[string]nodeVictimSet{
				"node1": {victimPods: [][]string{{"mid-both"}, {"low-web"}}, numPDBViolations: 2},
				"node2": {},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			domainVictims := prepareDomainVictims(tc.victims, 100, tc.pdbs)
			pdbViolationsBefore := perVictimPDBViolations(domainVictims)
			preemptionState := NewPodGroupPreemptionState(domainVictims)

			if diff := cmp.Diff(tc.wantInitial, gotNodeVictimSets(preemptionState, tc.wantInitial), cmp.AllowUnexported(nodeVictimSet{})); diff != "" {
				t.Fatalf("initial victim sets mismatch (-want +got):\n%s", diff)
			}

			condemned := preemptionState.SurvivingVictimsOnNode(tc.condemnNode)
			preemptionState.AddCondemned(condemned)
			if diff := cmp.Diff(tc.wantAfterCondemn, gotNodeVictimSets(preemptionState, tc.wantAfterCondemn), cmp.AllowUnexported(nodeVictimSet{})); diff != "" {
				t.Fatalf("after condemning %s, victim sets mismatch (-want +got):\n%s", tc.condemnNode, diff)
			}

			cloned := preemptionState.Clone().(*PodGroupPreemptionState)
			cloned.RemoveCondemned(condemned)
			if diff := cmp.Diff(tc.wantInitial, gotNodeVictimSets(cloned, tc.wantInitial), cmp.AllowUnexported(nodeVictimSet{})); diff != "" {
				t.Fatalf("cloned state after RemoveCondemned mismatch (-want +got):\n%s", diff)
			}
			if diff := cmp.Diff(tc.wantAfterCondemn, gotNodeVictimSets(preemptionState, tc.wantAfterCondemn), cmp.AllowUnexported(nodeVictimSet{})); diff != "" {
				t.Fatalf("original state after mutating clone mismatch (-want +got):\n%s", diff)
			}

			preemptionState.RemoveCondemned(condemned)
			if diff := cmp.Diff(tc.wantInitial, gotNodeVictimSets(preemptionState, tc.wantInitial), cmp.AllowUnexported(nodeVictimSet{})); diff != "" {
				t.Fatalf("after RemoveCondemned, victim sets mismatch (-want +got):\n%s", diff)
			}

			// Victims are shared by all nodes, so evaluating nodes must not write per-node counts into them.
			if diff := cmp.Diff(pdbViolationsBefore, perVictimPDBViolations(domainVictims)); diff != "" {
				t.Errorf("evaluating nodes changed NumPDBViolations of the domain victims (-before +after):\n%s", diff)
			}
		})
	}
}

func TestPodGroupPreemptionState_CustomPreemptionManagerVictims(t *testing.T) {
	makeVictim := func(name string, numPDBViolations int) fwk.PreemptionVictim {
		pi, err := framework.NewPodInfo(st.MakePod().Name(name).UID(name).Node("node1").Obj())
		if err != nil {
			t.Fatalf("NewPodInfo failed: %v", err)
		}
		return &customPreemptionVictim{Victim: NewPodVictim(pi), numPDBViolations: numPDBViolations}
	}
	victims := []fwk.PreemptionVictim{makeVictim("custom-1", 1), makeVictim("custom-2", 2)}
	preemptionState := NewPodGroupPreemptionState(victims)

	// Custom PreemptionManager victims don't expose their PDBs, so VictimSet and
	// countPDBViolations add up the NumPDBViolations() each victim reports.
	const want = 3
	if got := preemptionState.VictimSet(preemptionState.SurvivingVictimsOnNode("node1")).NumPDBViolations(); got != want {
		t.Errorf("VictimSet(node1).NumPDBViolations() = %d, want %d", got, want)
	}
	if got := countPDBViolations(victims); got != want {
		t.Errorf("countPDBViolations() = %d, want %d", got, want)
	}
}
