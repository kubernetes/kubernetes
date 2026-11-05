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
// to nodeName, taken from the final Filter allocation of the preemption
// simulation for nodeName (where only the chosen victims remain removed), or
// nil if nothing should be held.
func (pl *DynamicResources) PrepareNomination(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) any {
	if pl.preemptionTracker == nil {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil || state.preemption == nil || !state.preemption.canPreemptDevices {
		// The pod has no claims, or it can't use devices of victims, so
		// preemption freed nothing for it to hold.
		return nil
	}
	logger := klog.FromContext(ctx)
	result := state.preemption.simulatedResults.final(nodeName)
	if result == nil || len(result.heldClaims) == 0 {
		logger.V(5).Info("No devices to hold", "pod", klog.KObj(pod), "node", nodeName)
		return nil
	}
	hold := &podHold{
		nodeName:   nodeName,
		heldClaims: result.heldClaims,
	}
	logger.V(5).Info("Prepared hold", "pod", klog.KObj(pod), "node", nodeName, "numClaims", len(hold.heldClaims))
	return hold
}

// AddNomination records prepared as pod's device hold on nodeName, replacing
// any existing hold for pod and emitting a release event if previously held
// devices were freed. When prepared is nil (for example, a nomination loaded
// from the API on startup or a re-nomination that needed no unallocated
// claims), any existing hold for pod is dropped instead.
//
// PodNominator only passes what PrepareNomination returned for nodeName, and
// it calls RemoveNomination for the old node before AddNomination for a new
// one, so prepared and any existing hold are on nodeName.
func (pl *DynamicResources) AddNomination(ctx context.Context, pod *v1.Pod, nodeName string, prepared any) framework.NominationChangeResult {
	if pl.preemptionTracker == nil {
		return framework.NominationChangeResult{}
	}
	hold, _ := prepared.(*podHold)
	if hold == nil {
		return pl.releaseResult(ctx, pod, pl.preemptionTracker.release(pod.UID), nil)
	}
	replaced := pl.preemptionTracker.nominate(pod.UID, hold)
	klog.FromContext(ctx).V(5).Info("Holding devices for nominated pod", "pod", klog.KObj(pod), "node", nodeName, "numClaims", len(hold.heldClaims))
	return pl.releaseResult(ctx, pod, replaced, hold)
}

// RemoveNomination drops any device hold for pod, which is on nodeName (see
// AddNomination).
func (pl *DynamicResources) RemoveNomination(ctx context.Context, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	if pl.preemptionTracker == nil {
		return framework.NominationChangeResult{}
	}
	return pl.releaseResult(ctx, pod, pl.preemptionTracker.release(pod.UID), nil)
}

// releaseResult emits EventResourceClaimHoldReleased if dropping released frees
// any held devices without a corresponding ResourceClaim event — i.e., when
// neither replacement nor the claim's committed allocation retains the same
// devices.
//
// An in-flight pending allocation counts as retaining the devices only when pod
// is already assumed (Spec.NodeName != ""), meaning Reserve and Permit
// succeeded and any later failure will fire AssignedPod/Delete. If pod is still
// unassigned (for example, deleted mid-cycle between Reserve and Permit),
// Unreserve will drop the pending allocation silently, so we must emit the
// release event now.
func (pl *DynamicResources) releaseResult(ctx context.Context, pod *v1.Pod, released, replacement *podHold) framework.NominationChangeResult {
	if released == nil {
		return framework.NominationChangeResult{}
	}
	// Key by claim name rather than UID because a recreated claim under the
	// same name has a new UID.
	var kept map[types.NamespacedName]heldClaim
	if replacement != nil {
		kept = make(map[types.NamespacedName]heldClaim, len(replacement.heldClaims))
		for _, held := range replacement.heldClaims {
			kept[held.claim] = held
		}
	}
	assumed := pod.Spec.NodeName != ""
	for uid, held := range released.heldClaims {
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
	return allocation != nil && sameDevices(allocation.Devices.Results, held.allocatedDevices.Results)
}

func keepsDevices(kept, held heldClaim) bool {
	return kept.allocatedDevices != nil && sameDevices(kept.allocatedDevices.Results, held.allocatedDevices.Results)
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
	allocated.Status.Allocation = &resourceapi.AllocationResult{Devices: *held.allocatedDevices}
	deallocated := allocated.DeepCopy()
	deallocated.Status.Allocation = nil
	return framework.NominationEvent{Event: framework.EventResourceClaimHoldReleased, OldObj: allocated, NewObj: deallocated}
}
