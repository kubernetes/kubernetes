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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
)

// This file implements device holds for nominated preemptors (KEP-5690). When
// the framework runs pass 1 of RunFilterPluginsWithNominatedPods, it calls
// AddPod for nominated pods with priority >= the pod being scheduled; Filter
// then treats devices held for those nominated pods on the node as allocated.
//
// Holds only reserve devices and never free anything still in use:
//   - Exclusive and shared device IDs from active holds are added to the
//     allocated sets.
//   - Per-device consumed capacity becomes max(before, before - victims + held),
//     where victims is the capacity of still-settling claims released for those
//     holders. Subtracting victims avoids double-counting capacity that will be
//     deallocated before the holders bind, scoping victims to the holders' own
//     released claims prevents borrowing against unrelated preemptions, and
//     clamping at before keeps any surplus victim capacity unavailable until the
//     victim claim is actually deallocated.

// nodeHolds aggregates the devices and capacities held by nominated pods on a
// single node for one Filter call.
type nodeHolds struct {
	exclusive sets.Set[structured.DeviceID]
	shared    sets.Set[structured.DeviceID]
	// capacity is the consumed capacity held on shared devices.
	capacity structured.ConsumedCapacityCollection
	// victimClaims maps UIDs of still-settling claims released for the holders
	// to their consumed shared-device capacity. Filter also rejects pods that
	// try to reserve these claims so they can finish deallocating.
	victimClaims map[types.UID]structured.ConsumedCapacityCollection
}

// holdsOnNode collects the active device holds from sim.holds on nodeName, or
// returns nil if none apply.
func (pl *DynamicResources) holdsOnNode(state *stateData, sim *simulationState, nodeName string) *nodeHolds {
	if sim == nil || state.preemption == nil {
		return nil
	}
	var holds *nodeHolds
	for _, hold := range sim.holds {
		if hold.nodeName != nodeName {
			continue
		}
		if holds == nil {
			holds = &nodeHolds{
				exclusive:    sets.New[structured.DeviceID](),
				shared:       sets.New[structured.DeviceID](),
				capacity:     structured.NewConsumedCapacityCollection(),
				victimClaims: make(map[types.UID]structured.ConsumedCapacityCollection),
			}
		}
		for uid, held := range hold.claims {
			if state.preemption.podClaims.Has(uid) {
				// The pod being scheduled shares this unallocated claim with
				// the holder and will allocate it for both.
				continue
			}
			claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
			if err != nil || claim.UID != uid || claim.Status.Allocation != nil {
				// Skip if the claim was deleted/recreated (not part of the
				// preemption) or is already allocated (already counted in
				// AllocatedState).
				continue
			}
			structured.ForEachAllocatedDevice(held.allocation,
				func(id structured.DeviceID) {
					holds.exclusive.Insert(id)
				},
				pl.fts.EnableDRAConsumableCapacity,
				func(id structured.SharedDeviceID) {
					holds.shared.Insert(id.GetDeviceID())
				},
				func(capacity structured.DeviceConsumedCapacity) {
					holds.capacity.Insert(capacity)
				})
		}
		for uid, released := range hold.released {
			if _, ok := holds.victimClaims[uid]; ok {
				continue
			}
			if settling, _ := pl.victimClaimSettling(released); !settling {
				continue
			}
			capacity := structured.NewConsumedCapacityCollection()
			structured.ForEachAllocatedDevice(released.allocation,
				func(structured.DeviceID) {},
				pl.fts.EnableDRAConsumableCapacity,
				func(structured.SharedDeviceID) {},
				func(consumed structured.DeviceConsumedCapacity) {
					capacity.Insert(consumed)
				})
			holds.victimClaims[uid] = capacity
		}
	}
	return holds
}

// applyTo adds h's held devices and capacities to state after sim.applyTo has
// already removed sim's released claims.
func (h *nodeHolds) applyTo(state *structured.AllocatedState, released map[types.UID]*releasedClaim) {
	state.AllocatedDevices = state.AllocatedDevices.Union(h.exclusive)
	state.AllocatedSharedDeviceIDs = state.AllocatedSharedDeviceIDs.Union(h.shared)
	victims := structured.NewConsumedCapacityCollection()
	for uid, capacity := range h.victimClaims {
		if released[uid] != nil {
			// Already subtracted from state by sim.applyTo.
			continue
		}
		for id, consumed := range capacity {
			victims.Insert(structured.DeviceConsumedCapacity{DeviceID: id, ConsumedCapacity: consumed})
		}
	}
	for id, held := range h.capacity {
		before := state.AggregatedCapacity[id]
		after := before.Clone()
		for name, quantity := range held {
			current := before[name]
			value := *quantity
			if current != nil {
				value.Add(*current)
			}
			if victim, ok := victims[id][name]; ok {
				value.Sub(*victim)
			}
			if current == nil && value.Sign() > 0 || current != nil && value.Cmp(*current) > 0 {
				after[name] = new(value)
			}
		}
		state.AggregatedCapacity[id] = after
	}
}

// deallocating reports whether uid is a victim claim being deallocated for a
// holder on this node.
func (h *nodeHolds) deallocating(uid types.UID) bool {
	if h == nil {
		return false
	}
	_, ok := h.victimClaims[uid]
	return ok
}
