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

package dynamicresources

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
)

// This file implements device holds for nominated preemptors (KEP-5690),
// keeping devices freed by preemption from going to pods with priority <= the
// preemptor before it is scheduled:
//
//  1. Preemption simulation's Filter stores allocations in simulatedResults;
//     PrepareNomination/AddNomination records the chosen node's hold in
//     preemptionTracker, and RemoveNomination drops it when nomination ends.
//  2. In another pod's cycle, pass 1 of RunFilterPluginsWithNominatedPods
//     calls AddPod for nominated pods with priority >= that pod into sim.holds.
//  3. Filter applies holdsOnNode after simulated victim releases; pass 2 reuses
//     pass 1's cached result so Reserve never allocates held devices.
//
// Holds cover only exclusively allocated devices; unioning them into
// AllocatedDevices needs no reconciliation with still-settling victim claims.

// nodeHolds aggregates the devices held by nominated pods on a single node for
// one Filter call.
type nodeHolds struct {
	devices sets.Set[structured.DeviceID]
}

// holdsOnNode collects the active device holds from sim.holds on nodeName, or
// returns nil if none apply.
func (pl *DynamicResources) holdsOnNode(sim *preemptionSimulationState, nodeName string) *nodeHolds {
	if sim == nil {
		return nil
	}
	var holds *nodeHolds
	for _, hold := range sim.holds {
		if hold.nodeName != nodeName {
			continue
		}
		if holds == nil {
			holds = &nodeHolds{devices: sets.New[structured.DeviceID]()}
		}
		pl.addHeldClaims(holds, hold)
	}
	return holds
}

// addHeldClaims inserts hold's active exclusively held devices into holds.
func (pl *DynamicResources) addHeldClaims(holds *nodeHolds, hold *podHold) {
	for uid, held := range hold.heldClaims {
		claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
		if err != nil || claim.UID != uid || pl.allocationOrPending(claim) != nil {
			// Skip claims that were deleted or recreated, or are already
			// allocated or pending and thus counted in AllocatedState.
			continue
		}
		for id := range exclusiveDevices(held.allocatedDevices) {
			holds.devices.Insert(id)
		}
	}
}

func (h *nodeHolds) applyTo(allocatedState *structured.AllocatedState) {
	allocatedState.AllocatedDevices = allocatedState.AllocatedDevices.Union(h.devices)
}
