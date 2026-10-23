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
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// This file implements framework.AllocationExtensions (KEP-5690): while the
// PodNominator nominates a pod that preempted others, the devices which
// Filter selected for it in the preemption dry run are held for it, so that
// pods with lower or equal priority don't get them first. Holds end with the
// nomination. Devices which an assumed pod allocated without persisting them
// become available again when it gets forgotten.

var _ framework.AllocationExtensions = &DynamicResources{}

// PrepareAllocation records the allocation which Filter stored for the node
// in this scheduling cycle as the candidate allocation of the pod. For a
// nomination, that is the result of the preemption dry run.
func (pl *DynamicResources) PrepareAllocation(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if pl.nominations == nil {
		return
	}
	candidate := pl.preparedAllocation(cs, pod, nodeName)
	pl.nominations.setCandidate(pod.UID, candidate)
	if candidate == nil {
		return
	}
	klog.FromContext(ctx).V(5).Info("Prepared allocation", "pod", klog.KObj(pod), "node", nodeName, "assumed", candidate.assumed, "numClaims", len(candidate.claims))
	// If the pod got deleted before the candidate got recorded, then
	// nothing drops it anymore.
	if pl.podGone(pod.UID, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}) {
		pl.nominations.dropCandidate(pod.UID)
	}
}

// preparedAllocation returns the allocation of the claims of the pod which
// Filter stored for the node, nil if the pod cannot get devices for them.
func (pl *DynamicResources) preparedAllocation(cs fwk.CycleState, pod *v1.Pod, nodeName string) *allocation {
	state, err := getStateData(cs)
	if err != nil || state.preemption == nil {
		return nil
	}
	assumed := pod.Spec.NodeName != ""
	if !assumed && !state.preemption.root.canRelease {
		// Only pods which may get devices of preempted pods hold devices.
		return nil
	}
	state.mutex.Lock()
	result, ok := state.nodeAllocations[nodeName]
	state.mutex.Unlock()
	a := &allocation{nodeName: nodeName, priority: corev1helpers.PodPriority(pod), assumed: assumed}
	if ok {
		a.released = result.released
	}
	// The results match the claims to allocate without a pending allocation,
	// like in Filter.
	extendedResourceClaim := state.claims.extendedResourceClaim()
	i := 0
	for index, claim := range state.claims.toAllocate() {
		if state.informationsForClaim[index].allocation != nil {
			continue
		}
		if i >= len(result.allocationResults) {
			break
		}
		allocationResult := &result.allocationResults[i]
		i++
		if claim == extendedResourceClaim {
			continue
		}
		if a.claims == nil {
			a.claims = make(map[types.UID]heldClaim)
		}
		a.claims[claim.UID] = heldClaim{
			claim:      types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			allocation: allocationResult,
		}
	}
	if assumed && len(a.claims) == 0 {
		// Nothing to release when the assumed pod gets forgotten.
		return nil
	}
	return a
}

// AddAllocation confirms the candidate allocation which was prepared for the
// pod on the node. Without one, the plugin still remembers which node a pod
// with claims got nominated to: the nominated node in the pod object lags
// behind while asynchronous API calls are pending. A pod which got assumed
// is not a victim anymore, for example after it got preempted in memory
// while it waited in Permit.
func (pl *DynamicResources) AddAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations != nil {
		assumed := pod.Spec.NodeName != ""
		if assumed {
			pl.nominations.forgetVictim(pod.UID)
		}
		pl.nominations.addAllocation(pod.UID, nodeName, assumed, corev1helpers.PodPriority(pod), !assumed && len(pod.Spec.ResourceClaims) > 0)
	}
	return framework.AllocationChangeResult{}
}

// RemoveAllocation drops the allocation of the pod on the node. The devices
// become available for other pods, unless the pod keeps them: the assumed
// pod got the devices which were held for its nomination, or it got
// nominated to the same node again with the same devices. Devices of an
// assumed pod which got persisted get released by the API server instead.
func (pl *DynamicResources) RemoveAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations == nil {
		return framework.AllocationChangeResult{}
	}
	assumed := pod.Spec.NodeName != ""
	removed, keeper := pl.nominations.removeAllocation(pod.UID, nodeName, assumed)
	if removed == nil || len(removed.claims) == 0 {
		return framework.AllocationChangeResult{}
	}
	if keeper != nil && !keeper.assumed && pl.podGone(pod.UID, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}) {
		// A deleted pod does not get nominated again.
		keeper = nil
	}
	var released []types.UID
	for uid, held := range removed.claims {
		if keeper != nil && keeper.keeps(uid, held) || assumed && pl.persisted(uid, held) {
			continue
		}
		released = append(released, uid)
	}
	if len(released) == 0 {
		return framework.AllocationChangeResult{}
	}
	klog.FromContext(ctx).V(5).Info("Released devices", "pod", klog.KObj(pod), "node", nodeName, "assumed", assumed, "numClaims", len(released))
	// One event is enough, the queueing hints don't check which claim got
	// deallocated.
	uid := slices.Min(released)
	return framework.AllocationChangeResult{Events: []framework.AllocationEvent{releasedEvent(uid, removed.claims[uid])}}
}

// persisted returns true if the claim is allocated in the API server.
func (pl *DynamicResources) persisted(uid types.UID, held heldClaim) bool {
	claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
	return err == nil && claim.UID == uid && claim.Status.Allocation != nil
}

// releasedEvent describes the devices of the claim as if the claim got
// deallocated.
func releasedEvent(uid types.UID, held heldClaim) framework.AllocationEvent {
	claim := &resourceapi.ResourceClaim{}
	claim.Namespace, claim.Name, claim.UID = held.claim.Namespace, held.claim.Name, uid
	claim.Status.Allocation = held.allocation
	deallocated := claim.DeepCopy()
	deallocated.Status.Allocation = nil
	return framework.AllocationEvent{Event: framework.EventResourceClaimHoldReleased, OldObj: claim, NewObj: deallocated}
}

// sameDevices compares allocation results without their ShareIDs, which
// are new for each allocation of a device with consumable capacity.
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
