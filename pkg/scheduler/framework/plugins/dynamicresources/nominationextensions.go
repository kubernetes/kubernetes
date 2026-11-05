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
	"context"
	"slices"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// This file implements framework.NominationExtensions (KEP-5690), recording
// and releasing the device holds associated with nominated preemptors in
// PodNominator.

var _ framework.NominationExtensions = &DynamicResources{}

// PrepareNomination returns the device hold to associate with pod if nominated
// to nodeName: the Filter allocation from the final simulated state of
// nodeName's preemption dry run (where only the chosen victims remain removed),
// or nil if nothing should be held.
func (pl *DynamicResources) PrepareNomination(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) any {
	if pl.store == nil {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil || state.preemption == nil {
		return nil
	}
	logger := klog.FromContext(ctx)
	hold := state.holdForNomination(nodeName)
	if hold == nil {
		logger.V(5).Info("No devices to hold", "pod", klog.KObj(pod), "node", nodeName)
		return nil
	}
	logger.V(5).Info("Prepared hold", "pod", klog.KObj(pod), "node", nodeName, "numClaims", len(hold.claims), "numReleasedClaims", len(hold.released))
	return hold
}

func (state *stateData) holdForNomination(nodeName string) *allocation {
	if !state.preemption.canRelease {
		// Pods that cannot release victim devices via preemption never acquire holds.
		return nil
	}
	result := state.dryRuns.final(nodeName)
	if result == nil || len(result.claims) == 0 {
		return nil
	}
	hold := &allocation{
		nodeName: nodeName,
		claims:   result.claims,
		released: make(map[types.UID]*releasedClaim, len(result.released)),
	}
	for _, released := range result.released {
		hold.released[released.uid] = released
	}
	return hold
}

// AddNomination records prepared as pod's device hold on nodeName, replacing
// any existing hold for pod and emitting a release event if previously held
// devices were freed. When prepared is nil (for example, a nomination loaded
// from the API on startup or a re-nomination that needed no unallocated
// claims), any existing hold for pod on nodeName is dropped instead.
func (pl *DynamicResources) AddNomination(ctx context.Context, pod *v1.Pod, nodeName string, prepared any) framework.NominationChangeResult {
	if pl.store == nil {
		return framework.NominationChangeResult{}
	}
	hold, _ := prepared.(*allocation)
	if hold == nil || hold.nodeName != nodeName {
		return pl.releaseResult(ctx, pod, pl.store.release(pod.UID, nodeName), nil)
	}
	replaced := pl.store.nominate(pod.UID, hold)
	klog.FromContext(ctx).V(5).Info("Holding devices for nominated pod", "pod", klog.KObj(pod), "node", nodeName, "numClaims", len(hold.claims))
	return pl.releaseResult(ctx, pod, replaced, hold)
}

// RemoveNomination drops any device hold for pod on nodeName.
func (pl *DynamicResources) RemoveNomination(ctx context.Context, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	if pl.store == nil {
		return framework.NominationChangeResult{}
	}
	return pl.releaseResult(ctx, pod, pl.store.release(pod.UID, nodeName), nil)
}

// releaseResult emits EventResourceClaimHoldReleased if dropping released frees
// any held devices without a corresponding ResourceClaim event — i.e., when
// neither replacement (on the same node) nor the claim's committed allocation
// retains the same devices.
//
// An in-flight pending allocation counts as retaining the devices only when pod
// is already assumed (Spec.NodeName != ""), meaning Reserve and Permit
// succeeded and any later failure will fire AssignedPod/Delete. If pod is still
// unassigned (for example, deleted mid-cycle between Reserve and Permit),
// Unreserve will drop the pending allocation silently, so we must emit the
// release event now.
func (pl *DynamicResources) releaseResult(ctx context.Context, pod *v1.Pod, released, replacement *allocation) framework.NominationChangeResult {
	if released == nil {
		return framework.NominationChangeResult{}
	}
	// Key by claim name rather than UID: if a claim was recreated under the
	// same name, pod now references the new claim.
	var kept map[types.NamespacedName]heldClaim
	if replacement != nil && replacement.nodeName == released.nodeName {
		kept = make(map[types.NamespacedName]heldClaim, len(replacement.claims))
		for _, held := range replacement.claims {
			kept[held.claim] = held
		}
	}
	assumed := pod.Spec.NodeName != ""
	for uid, held := range released.claims {
		if keepsDevices(kept[held.claim], held) {
			continue
		}
		if pl.allocatedWith(held, assumed) {
			continue
		}
		klog.FromContext(ctx).V(5).Info("Released devices held for pod", "pod", klog.KObj(pod), "node", released.nodeName, "resourceclaim", held.claim)
		// A single event suffices because ResourceClaim queueing hints do not
		// filter by specific claim when waking pods waiting for devices.
		return framework.NominationChangeResult{Events: []framework.NominationEvent{releaseEvent(held, uid)}}
	}
	return framework.NominationChangeResult{}
}

func (pl *DynamicResources) allocatedWith(held heldClaim, withPending bool) bool {
	claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
	if err != nil {
		return false
	}
	allocation := claim.Status.Allocation
	if allocation == nil && withPending {
		allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
	}
	return allocation != nil && sameDevices(allocation.Devices.Results, held.allocation.Devices.Results)
}

func keepsDevices(kept, held heldClaim) bool {
	return kept.allocation != nil && sameDevices(kept.allocation.Devices.Results, held.allocation.Devices.Results)
}

// sameDevices compares device allocation results ignoring ShareID, which is
// freshly generated on each allocation.
func sameDevices(a, b []resourceapi.DeviceRequestAllocationResult) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		resultA, resultB := a[i], b[i]
		resultA.ShareID, resultB.ShareID = nil, nil
		if !apiequality.Semantic.DeepEqual(resultA, resultB) {
			return false
		}
	}
	return true
}

// releaseEvent constructs a synthetic allocation-to-nil transition so
// ResourceClaim queueing hints treat the released hold like a claim
// deallocation.
func releaseEvent(held heldClaim, uid types.UID) framework.NominationEvent {
	allocated := &resourceapi.ResourceClaim{}
	allocated.Namespace, allocated.Name, allocated.UID = held.claim.Namespace, held.claim.Name, uid
	allocated.Status.Allocation = held.allocation
	deallocated := allocated.DeepCopy()
	deallocated.Status.Allocation = nil
	return framework.NominationEvent{Event: framework.EventResourceClaimHoldReleased, OldObj: allocated, NewObj: deallocated}
}

// heldClaims pairs unallocated user claims with their Filter allocation
// results, skipping synthetic extended-resource claims.
func heldClaims(claimsToAllocate []*resourceapi.ResourceClaim, results []resourceapi.AllocationResult, extendedResourceClaims ...*resourceapi.ResourceClaim) map[types.UID]heldClaim {
	claims := make(map[types.UID]heldClaim, len(claimsToAllocate))
	for i, claim := range claimsToAllocate {
		if slices.Contains(extendedResourceClaims, claim) {
			continue
		}
		result := results[i]
		claims[claim.UID] = heldClaim{
			claim:      types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			allocation: &result,
		}
	}
	return claims
}

func (sim *simulationState) releasedOnNode(nodeName string) []*releasedClaim {
	var released []*releasedClaim
	for _, claim := range sim.released {
		if claim.nodeName == nodeName {
			released = append(released, claim)
		}
	}
	return released
}
