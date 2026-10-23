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
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
)

// This file implements holding devices for nominated preemptors (KEP-5690):
// when Filter checks a node, the devices which preemption freed for other
// pods nominated to the node with at least the priority of the pod being
// scheduled count as allocated, so that it does not take them. Those are the
// nominated pods which RunFilterPluginsWithNominatedPods adds in its first
// pass. The holds come from the allocations which AllocationExtensions
// confirmed for the nominations of the PodNominator instead of AddPod, so
// that its second pass, whose result gets stored for Reserve, sees them as
// well.
//
// Holds never free devices, they only add the held devices to the state
// without them: exclusively allocated devices and shared devices are added,
// consumed capacity becomes max(before, before - victims + held) per device,
// where victims is the capacity of the claims of preempted pods which the
// holds rely on and which are still allocated. Those get deallocated before
// the holders use the capacity, so it must not be counted twice. Counting
// only the claims which the holds rely on keeps capacity that a preemption
// freed from being taken by borrowing against claims of other preemptions.
// Because nothing gets freed, devices of a victim claim which the holders
// don't need stay in use until the claim gets deallocated.

// nodeHolds are the devices held on a node for nominated pods.
type nodeHolds struct {
	// released are the UIDs of the claims of preempted pods which the
	// holds rely on.
	released sets.Set[types.UID]
	// exclusive and shared are the devices held without and with sharing.
	exclusive sets.Set[structured.DeviceID]
	shared    sets.Set[structured.DeviceID]
	// capacity is the consumed capacity held for shared devices.
	capacity structured.ConsumedCapacityCollection
	// victimClaims are the claims of preempted pods which the holds rely on
	// and which are being deallocated, with the capacity that their shares
	// consume. No pod may start to use them, otherwise they would not get
	// deallocated.
	victimClaims map[types.UID]structured.ConsumedCapacityCollection
}

// holdsOnNode returns the devices held on the node which the pod of the
// preemption state must not get, nil if no nomination applies: those held
// for other pods nominated to the node with at least its priority.
func (pl *DynamicResources) holdsOnNode(logger klog.Logger, p *preemptionState, nodeName string) *nodeHolds {
	if p == nil || pl.nominations == nil {
		return nil
	}
	var holds *nodeHolds
	for holderUID, nomination := range pl.nominations.nominationsOnNode(nodeName) {
		// The hold of the pod itself is what it is going to get, or gets
		// replaced when it preempts again.
		if holderUID == p.root.podUID || nomination.priority < p.root.priority || len(nomination.claims) == 0 {
			continue
		}
		if holds == nil {
			holds = &nodeHolds{
				released:     sets.New[types.UID](),
				exclusive:    sets.New[structured.DeviceID](),
				shared:       sets.New[structured.DeviceID](),
				capacity:     structured.NewConsumedCapacityCollection(),
				victimClaims: make(map[types.UID]structured.ConsumedCapacityCollection),
			}
		}
		holds.released = holds.released.Union(nomination.released)
		for uid, held := range nomination.claims {
			if p.root.podClaims.Has(uid) {
				// The pod being scheduled shares the claim, so it is
				// going to allocate it for both.
				continue
			}
			claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
			if err != nil || claim.UID != uid || claim.Status.Allocation != nil {
				// Allocated devices are already in use, a different
				// claim was not part of the preemption.
				continue
			}
			foreachAllocatedDevice(&resourceapi.ResourceClaim{Status: resourceapi.ResourceClaimStatus{Allocation: held.allocation}},
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
		logger.V(5).Info("Devices held for nominated pod", "podUID", holderUID, "node", nodeName, "numClaims", len(nomination.claims))
	}
	if holds == nil {
		return nil
	}
	for _, victimClaim := range pl.nominations.victimClaimsOnNode(nodeName) {
		if !holds.released.Has(victimClaim.uid) || !pl.victimClaimSettling(victimClaim) {
			continue
		}
		capacity := structured.NewConsumedCapacityCollection()
		foreachAllocatedDevice(&resourceapi.ResourceClaim{Status: resourceapi.ResourceClaimStatus{Allocation: &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: victimClaim.results}}}},
			func(structured.DeviceID) {},
			pl.fts.EnableDRAConsumableCapacity,
			func(structured.SharedDeviceID) {},
			func(consumed structured.DeviceConsumedCapacity) {
				capacity.Insert(consumed)
			})
		holds.victimClaims[victimClaim.uid] = capacity
	}
	return holds
}

// applyTo adds the held devices to the allocated state, which already has
// the released devices removed.
func (h *nodeHolds) applyTo(state *structured.AllocatedState, released map[types.UID]*releasedClaim) {
	state.AllocatedDevices = state.AllocatedDevices.Union(h.exclusive)
	state.AllocatedSharedDeviceIDs = state.AllocatedSharedDeviceIDs.Union(h.shared)
	victims := structured.NewConsumedCapacityCollection()
	for uid, capacity := range h.victimClaims {
		if released[uid] != nil {
			// Already subtracted.
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
			// A quantity which is not consumed yet is zero.
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

// deallocating returns true if the claim is a claim of the holders' victims
// which is being deallocated.
func (h *nodeHolds) deallocating(uid types.UID) bool {
	if h == nil {
		return false
	}
	_, ok := h.victimClaims[uid]
	return ok
}
