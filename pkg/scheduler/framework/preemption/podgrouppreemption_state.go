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

const podGroupPreemptionStateKey fwk.StateKey = "PodGroupPreemptionState"

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
	state.Write(podGroupPreemptionStateKey, s)
}

// GetPodGroupPreemptionState retrieves PodGroupPreemptionState from a pod's CycleState
// by walking up its PodGroupCycleState / PlacementCycleState hierarchy to the root PodGroupCycleState.
// Returns (nil, nil) if not in a PodGroup preemption cycle.
func GetPodGroupPreemptionState(state fwk.CycleState) (*PodGroupPreemptionState, error) {
	if state == nil {
		return nil, nil
	}
	for pgState := state.GetPodGroupCycleState(); pgState != nil; {
		c, err := pgState.Read(podGroupPreemptionStateKey)
		if err == nil {
			s, ok := c.(*PodGroupPreemptionState)
			if !ok {
				return nil, fmt.Errorf("failed to convert %T to *PodGroupPreemptionState", c)
			}
			return s, nil
		}
		if !errors.Is(err, fwk.ErrNotFound) {
			return nil, fmt.Errorf("error reading %q from podGroupCycleState: %w", podGroupPreemptionStateKey, err)
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
// sorted from most to least important.
func (s *PodGroupPreemptionState) SurvivingVictimsOnNode(nodeName string) []fwk.PreemptionVictim {
	s.lock.RLock()
	defer s.lock.RUnlock()
	var surviving []fwk.PreemptionVictim
	for _, v := range s.victimsByNode[nodeName] {
		if !s.condemned.Has(v) {
			surviving = append(surviving, v)
		}
	}
	return surviving
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
