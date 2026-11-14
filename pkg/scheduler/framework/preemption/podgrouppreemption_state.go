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
	"errors"
	"fmt"
	"maps"
	"slices"
	"sort"
	"sync"

	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodGroupPreemptionStateKey is the PodGroupCycleState key used to store PodGroupPreemptionState.
const PodGroupPreemptionStateKey fwk.StateKey = "PodGroupPreemptionState"

// PodGroupPreemptionState tracks dynamic victim eviction state across sequential
// pod scheduling attempts within a single PodGroup preemption evaluation.
type PodGroupPreemptionState struct {
	lock sync.RWMutex

	// victimsByNode maps nodeName -> list of potential PreemptionVictims that have
	// at least one pod scheduled on that node, sorted from least to most preferred for preemption.
	victimsByNode map[string][]fwk.PreemptionVictim

	// condemned tracks the set of PreemptionVictims that assumed preemptor pods require to be evicted.
	condemned sets.Set[fwk.PreemptionVictim]

	// pdbDisruptionsAllowed tracks the remaining DisruptionsAllowed budget for each PodDisruptionBudget
	// as victims are condemned in Reserve and restored in Unreserve.
	pdbDisruptionsAllowed map[*policy.PodDisruptionBudget]int32
}

// Clone implements fwk.StateData.
// victimsByNode is read-only after initialization in NewPodGroupPreemptionState,
// so the clone shares the map and copies the mutable condemned set and PDB budgets.
func (s *PodGroupPreemptionState) Clone() fwk.StateData {
	if s == nil {
		return nil
	}
	s.lock.RLock()
	defer s.lock.RUnlock()
	return &PodGroupPreemptionState{
		victimsByNode:         s.victimsByNode,
		condemned:             s.condemned.Clone(),
		pdbDisruptionsAllowed: maps.Clone(s.pdbDisruptionsAllowed),
	}
}

// WriteToPodGroupCycleState stores PodGroupPreemptionState in PodGroupCycleState.
func (s *PodGroupPreemptionState) WriteToPodGroupCycleState(state fwk.PodGroupCycleState) {
	if state == nil {
		return
	}
	state.Write(PodGroupPreemptionStateKey, s)
}

// GetPodGroupPreemptionState retrieves PodGroupPreemptionState from a pod's CycleState
// by walking up its PodGroupCycleState / PlacementCycleState hierarchy to the root PodGroupCycleState.
// Returns (nil, nil) if not in a PodGroup preemption cycle.
func GetPodGroupPreemptionState(state fwk.CycleState) (*PodGroupPreemptionState, error) {
	if state == nil {
		return nil, nil
	}
	for pgState := state.GetPodGroupCycleState(); pgState != nil; {
		c, err := pgState.Read(PodGroupPreemptionStateKey)
		if err == nil {
			s, ok := c.(*PodGroupPreemptionState)
			if !ok {
				return nil, fmt.Errorf("failed to convert %T to *PodGroupPreemptionState", c)
			}
			return s, nil
		}
		if !errors.Is(err, fwk.ErrNotFound) {
			return nil, fmt.Errorf("error reading %q from podGroupCycleState: %w", PodGroupPreemptionStateKey, err)
		}
		parentPlacement := pgState.GetParentPlacementCycleState()
		if parentPlacement == nil {
			break
		}
		pgState = parentPlacement.GetPodGroupCycleState()
	}
	return nil, nil
}

// NewPodGroupPreemptionState initializes dynamic victim tracking state for a PodGroup preemption cycle.
// The provided potentialVictims slice is expected to be ordered from most to least preferred for preemption
// (as returned by PreemptionManager.GenerateVictims).
func NewPodGroupPreemptionState(potentialVictims []fwk.PreemptionVictim) *PodGroupPreemptionState {
	victimsByNode := make(map[string][]fwk.PreemptionVictim)
	pdbDisruptionsAllowed := make(map[*policy.PodDisruptionBudget]int32)
	for _, v := range slices.Backward(potentialVictims) {
		if vw, ok := v.(*victimWithPDBViolations); ok {
			for _, pdbsPerPod := range vw.pdbsPerPod {
				for _, pdb := range pdbsPerPod {
					pdbDisruptionsAllowed[pdb] = pdb.Status.DisruptionsAllowed
				}
			}
		}
		for _, pi := range v.Pods() {
			nodeName := pi.GetPod().Spec.NodeName
			nodeVictims := victimsByNode[nodeName]
			// A PodGroup or CompositePodGroup victim can have multiple pods on the same node,
			// and duplicating it in victimsByNode[nodeName] would double-count its PDB violations
			// and victim count in CompareVictimSets.
			if len(nodeVictims) == 0 || nodeVictims[len(nodeVictims)-1] != v {
				victimsByNode[nodeName] = append(nodeVictims, v)
			}
		}
	}
	// GenerateVictims places violating victims after non-violating victims regardless of priority.
	// re-sort by MoreImportantVictim so SurvivingVictimsOnNode can evaluate victims in priority order.
	if len(pdbDisruptionsAllowed) > 0 {
		for _, victims := range victimsByNode {
			sort.Slice(victims, func(i, j int) bool {
				return MoreImportantVictim(victims[i].(Victim), victims[j].(Victim))
			})
		}
	}
	return &PodGroupPreemptionState{
		victimsByNode:         victimsByNode,
		condemned:             sets.New[fwk.PreemptionVictim](),
		pdbDisruptionsAllowed: pdbDisruptionsAllowed,
	}
}

// SurvivingVictimsOnNode returns all potential victims on nodeName that have not yet been condemned,
// sorted from least to most preferred for preemption. The returned slice must not be mutated by callers.
func (s *PodGroupPreemptionState) SurvivingVictimsOnNode(nodeName string) []fwk.PreemptionVictim {
	s.lock.Lock()
	defer s.lock.Unlock()

	victims := s.victimsByNode[nodeName]
	if len(victims) == 0 {
		return victims
	}
	// If no victims are condemned and victims are not protected by any PDBs, then all victims
	// on this node survive with their original ordering and violation counts are unchanged.
	if len(s.condemned) == 0 && len(s.pdbDisruptionsAllowed) == 0 {
		return victims
	}

	remainingDisruptionsAllowed := maps.Clone(s.pdbDisruptionsAllowed)
	hasCondemned := false
	hasViolating := false
	// Evaluate surviving victims from least important to most important so lower-priority
	// victims on nodeName consume available PDB disruptions before higher-priority victims.
	for _, v := range slices.Backward(victims) {
		if s.condemned.Has(v) {
			hasCondemned = true
			continue
		}
		if vw, ok := v.(*victimWithPDBViolations); ok {
			vw.numPDBViolations = countVictimPDBViolations(vw, remainingDisruptionsAllowed)
		}
		if v.NumPDBViolations() > 0 {
			hasViolating = true
		}
	}

	// On most nodes no victim is condemned and none violate a PDB, so avoid allocating a new slice.
	if !hasCondemned && !hasViolating {
		return victims
	}

	var violating, nonViolating []fwk.PreemptionVictim
	for _, v := range victims {
		if s.condemned.Has(v) {
			continue
		}
		if v.NumPDBViolations() > 0 {
			violating = append(violating, v)
		} else {
			nonViolating = append(nonViolating, v)
		}
	}
	return append(violating, nonViolating...)
}

func countVictimPDBViolations(vw *victimWithPDBViolations, remainingDisruptionsAllowed map[*policy.PodDisruptionBudget]int32) int {
	violatingPods := 0
	for _, podPDBs := range vw.pdbsPerPod {
		podViolates := false
		for _, pdb := range podPDBs {
			remainingDisruptionsAllowed[pdb]--
			if remainingDisruptionsAllowed[pdb] < 0 {
				podViolates = true
			}
		}
		if podViolates {
			violatingPods++
		}
	}
	return violatingPods
}

// AddCondemned marks the given victims as condemned and decrements remaining
// DisruptionsAllowed budgets for any PDBs matched by those victims.
func (s *PodGroupPreemptionState) AddCondemned(victims []fwk.PreemptionVictim) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.condemned.Insert(victims...)
	s.updatePDBDisruptions(victims, -1)
}

// RemoveCondemned unmarks the given victims as condemned when reverting condemnation
// and restores DisruptionsAllowed budgets for any PDBs matched by those victims.
func (s *PodGroupPreemptionState) RemoveCondemned(victims []fwk.PreemptionVictim) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.condemned.Delete(victims...)
	s.updatePDBDisruptions(victims, +1)
}

func (s *PodGroupPreemptionState) updatePDBDisruptions(victims []fwk.PreemptionVictim, delta int32) {
	if len(s.pdbDisruptionsAllowed) == 0 {
		return
	}
	for _, v := range victims {
		if vw, ok := v.(*victimWithPDBViolations); ok {
			for _, podPDBs := range vw.pdbsPerPod {
				for _, pdb := range podPDBs {
					s.pdbDisruptionsAllowed[pdb] += delta
				}
			}
		}
	}
}
