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

	// pdbDisruptionsAllowed tracks the remaining DisruptionsAllowed for each PodDisruptionBudget
	// as victims are condemned in Reserve and restored in Unreserve.
	pdbDisruptionsAllowed map[*policy.PodDisruptionBudget]int32
}

// Clone implements fwk.StateData.
// victimsByNode is read-only after initialization in NewPodGroupPreemptionState,
// so the clone shares the map and copies the mutable condemned set and remaining DisruptionsAllowed.
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

// victimSet implements fwk.PreemptionVictimSet.
type victimSet struct {
	victims          []fwk.PreemptionVictim
	numPDBViolations int
}

var _ fwk.PreemptionVictimSet = &victimSet{}

func (s *victimSet) Victims() []fwk.PreemptionVictim {
	return s.victims
}

func (s *victimSet) NumPDBViolations() int {
	return s.numPDBViolations
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
			initPDBDisruptionsAllowed(pdbDisruptionsAllowed, vw)
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
	// GenerateVictims puts violating victims after non-violating ones regardless of the priority. VictimSet recounts
	// PDB violations per node, and the default CompareVictimSets compare victims from the most to the least important, so
	// each node's victims must be sorted from the most to the least important as well. However sorting is only needed if
	// there is at least one victim that is protected by a PDB, because otherwise there will be no violations.
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

// VictimSet returns victims as a PreemptionVictimSet, together with the PDB violations that evicting them
// would cause on top of the already condemned victims. Victims are ordered from the most to the least important.
func (s *PodGroupPreemptionState) VictimSet(victims []fwk.PreemptionVictim) fwk.PreemptionVictimSet {
	s.lock.RLock()
	defer s.lock.RUnlock()

	var remainingDisruptionsAllowed map[*policy.PodDisruptionBudget]int32
	numPDBViolations := 0
	// Evaluate victims from least to most important to have the least important victims on the node consume the remaining PDB disruptions first.
	for _, v := range slices.Backward(victims) {
		if vw, ok := v.(*victimWithPDBViolations); ok {
			// Check that at least one victim on this node is protected by a PDB to
			// avoid unnecessary allocations of remainingDisruptionsAllowed.
			if len(vw.pdbsPerPod) == 0 {
				continue
			}
			if remainingDisruptionsAllowed == nil {
				remainingDisruptionsAllowed = make(map[*policy.PodDisruptionBudget]int32)
			}
			// Default PreemptionManager victims carry their matched PDBs, so copy only their remaining
			// DisruptionsAllowed from state and recount violations against them.
			for _, podPDBs := range vw.pdbsPerPod {
				for _, pdb := range podPDBs {
					if _, exists := remainingDisruptionsAllowed[pdb]; !exists {
						remainingDisruptionsAllowed[pdb] = s.pdbDisruptionsAllowed[pdb]
					}
				}
			}
			numPDBViolations += countVictimPDBViolations(vw, remainingDisruptionsAllowed)
		} else {
			// Custom PreemptionManager victims do not expose their PDBs, so we use the static
			// NumPDBViolations() their manager computed in GenerateVictims.
			numPDBViolations += v.NumPDBViolations()
		}
	}
	return &victimSet{victims: victims, numPDBViolations: numPDBViolations}
}

// SurvivingVictimsOnNode returns all potential victims on nodeName that have not yet been condemned,
// sorted from the most to the least important victim. The returned slice must not be mutated by callers.
func (s *PodGroupPreemptionState) SurvivingVictimsOnNode(nodeName string) []fwk.PreemptionVictim {
	s.lock.RLock()
	defer s.lock.RUnlock()
	victims := s.victimsByNode[nodeName]
	if len(victims) == 0 || len(s.condemned) == 0 {
		return victims
	}
	// On most nodes no victim is condemned, so avoid allocating a new slice
	// unless at least one victim on this node is in s.condemned.
	for i, v := range victims {
		if s.condemned.Has(v) {
			surviving := make([]fwk.PreemptionVictim, i, len(victims)-1)
			copy(surviving, victims[:i])
			for _, v := range victims[i+1:] {
				if !s.condemned.Has(v) {
					surviving = append(surviving, v)
				}
			}
			return surviving
		}
	}
	return victims
}

// countPDBViolations returns the number of PDB violations that evicting all the given victims would cause.
func countPDBViolations(victims []fwk.PreemptionVictim) int {
	numPDBViolations := 0
	var pdbDisruptionsAllowed map[*policy.PodDisruptionBudget]int32
	var pdbProtectedVictims []*victimWithPDBViolations
	for _, v := range victims {
		if vw, ok := v.(*victimWithPDBViolations); ok {
			// Default PreemptionManager victims carry their matched PDBs, so violations are recounted against each PDB's full DisruptionsAllowed.
			if len(vw.pdbsPerPod) > 0 {
				if pdbDisruptionsAllowed == nil {
					pdbDisruptionsAllowed = make(map[*policy.PodDisruptionBudget]int32)
				}
				initPDBDisruptionsAllowed(pdbDisruptionsAllowed, vw)
				pdbProtectedVictims = append(pdbProtectedVictims, vw)
			}
		} else {
			// Custom PreemptionManager victims do not expose their PDBs, so we use the static
			// NumPDBViolations() their manager computed in GenerateVictims.
			numPDBViolations += v.NumPDBViolations()
		}
	}
	if len(pdbDisruptionsAllowed) == 0 {
		return numPDBViolations
	}

	// For a pod protected by a single PDB, order doesn't matter, but when a pod matches multiple PDBs,
	// evaluating it earlier can both violate one PDB and consume another PDB's DisruptionsAllowed,
	// causing a later pod to violate as well. Therefore, we evaluate victims from least to most important
	// to match GenerateVictims, ensuring we never report more violations than GenerateVictims did.
	sort.Slice(pdbProtectedVictims, func(i, j int) bool {
		return MoreImportantVictim(pdbProtectedVictims[i], pdbProtectedVictims[j])
	})

	for _, vw := range slices.Backward(pdbProtectedVictims) {
		numPDBViolations += countVictimPDBViolations(vw, pdbDisruptionsAllowed)
	}
	return numPDBViolations
}

// AddCondemned marks the given victims as condemned and decrements remaining
// DisruptionsAllowed for any PDBs matched by those victims.
func (s *PodGroupPreemptionState) AddCondemned(victims []fwk.PreemptionVictim) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.condemned.Insert(victims...)
	s.updatePDBDisruptions(victims, -1)
}

// RemoveCondemned unmarks the given victims as condemned when reverting condemnation
// and restores DisruptionsAllowed for any PDBs matched by those victims.
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
