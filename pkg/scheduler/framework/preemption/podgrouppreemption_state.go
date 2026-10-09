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
	"slices"
	"sort"
	"sync"

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
	// at least one pod scheduled on that node, sorted from most to least important.
	victimsByNode map[string][]fwk.PreemptionVictim

	// condemned tracks the set of PreemptionVictims that assumed preemptor pods require to be evicted.
	condemned sets.Set[fwk.PreemptionVictim]
}

// Clone implements fwk.StateData.
// victimsByNode is read-only after initialization in NewPodGroupPreemptionState,
// so the clone shares the map and copies the mutable condemned set.
func (s *PodGroupPreemptionState) Clone() fwk.StateData {
	if s == nil {
		return nil
	}
	s.lock.RLock()
	defer s.lock.RUnlock()
	return &PodGroupPreemptionState{
		victimsByNode: s.victimsByNode,
		condemned:     s.condemned.Clone(),
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
// The provided potentialVictims slice is expected to be ordered from least important to most important
// (as returned by PreemptionManager.GenerateVictims).
// TODO(#142718): Support custom PreemptionManager implementations that do not implement Victim
// or return non-comparable PreemptionVictim values by respecting GenerateVictims ordering and
// indexing victims by integer ID instead of storing interface values in sets.Set / slices.Contains.
func NewPodGroupPreemptionState(potentialVictims []fwk.PreemptionVictim) *PodGroupPreemptionState {
	victimsByNode := make(map[string][]fwk.PreemptionVictim)
	for _, v := range slices.Backward(potentialVictims) {
		for _, pi := range v.Pods() {
			nodeName := pi.GetPod().Spec.NodeName
			// A PodGroup or CompositePodGroup victim can have multiple pods on the same node,
			// and duplicating it in victimsByNode[nodeName] would double-count its PDB violations
			// and victim count in CompareVictimSets.
			if !slices.Contains(victimsByNode[nodeName], v) {
				victimsByNode[nodeName] = append(victimsByNode[nodeName], v)
			}
		}
	}
	for _, victims := range victimsByNode {
		sort.SliceStable(victims, func(i, j int) bool {
			vi1, ok1 := victims[i].(Victim)
			vi2, ok2 := victims[j].(Victim)
			if ok1 && ok2 {
				return MoreImportantVictim(vi1, vi2)
			}
			return false
		})
	}
	return &PodGroupPreemptionState{
		victimsByNode: victimsByNode,
		condemned:     sets.New[fwk.PreemptionVictim](),
	}
}

// SurvivingVictimsOnNode returns all potential victims on nodeName that have not yet been condemned,
// sorted from most to least important. The returned slice must not be mutated by callers.
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

// AddCondemned marks the given victims as condemned.
func (s *PodGroupPreemptionState) AddCondemned(victims []fwk.PreemptionVictim) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.condemned.Insert(victims...)
}

// RemoveCondemned unmarks the given victims as condemned when reverting condemnation.
func (s *PodGroupPreemptionState) RemoveCondemned(victims []fwk.PreemptionVictim) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.condemned.Delete(victims...)
}
