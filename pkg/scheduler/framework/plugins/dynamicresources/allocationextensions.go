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
	"fmt"
	"maps"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// This file implements framework.AllocationExtensions (KEP-5690): devices
// which preemption freed get held for the preemptor while the PodNominator
// nominates it to the node, and an assumed pod keeps the devices which
// Filter selected for it until it got bound or forgotten.

var _ framework.AllocationExtensions = &DynamicResources{}

// PrepareAllocation records what the scheduling cycle selected for the pod on
// the node as its candidate. For a pod which gets nominated, those are the
// devices which the preemption simulation found for it with its victims
// removed and the victims with claims. For a pod which gets assumed, those
// are the devices which Filter selected for it.
func (pl *DynamicResources) PrepareAllocation(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if pl.nominations == nil {
		return
	}
	state, err := getStateData(cs)
	if err != nil {
		return
	}
	var candidate *podNomination
	if pod.Spec.NodeName != "" {
		candidate = assumedAllocation(state, nodeName)
	} else {
		candidate = pl.nominatedAllocation(ctx, state, pod, nodeName)
	}
	pl.nominations.prepare(pod.UID, candidate)
	if candidate != nil {
		klog.FromContext(ctx).V(5).Info("Prepared allocation", "pod", klog.KObj(pod), "node", nodeName, "assumed", candidate.assumed, "numHeldClaims", len(candidate.holds))
	}
}

// nominatedAllocation returns the candidate of a pod which gets nominated by
// preemption, nil if preemption cannot release devices for it.
func (pl *DynamicResources) nominatedAllocation(ctx context.Context, state *stateData, pod *v1.Pod, nodeName string) *podNomination {
	if state.preemption == nil || !state.preemption.root.canRelease {
		return nil
	}
	sim := state.preemption.simulationFor(nodeName)
	if sim == nil {
		sim = state.preemption
	}
	p := sim.clone()
	return &podNomination{
		nodeName: nodeName,
		priority: corev1helpers.PodPriority(pod),
		victims:  maps.Clone(p.sim.removedPods),
		holds:    pl.computeHold(ctx, state, p, nodeName),
	}
}

// assumedAllocation returns the candidate of a pod which gets assumed: the
// devices which Filter selected for it on the node and which Reserve is
// going to use, nil if there are none.
func assumedAllocation(state *stateData, nodeName string) *podNomination {
	state.mutex.Lock()
	allocations, ok := state.nodeAllocations[nodeName]
	state.mutex.Unlock()
	if !ok || len(allocations.allocationResults) == 0 {
		return nil
	}
	holds := make(map[types.UID]heldClaim)
	extendedResourceClaim := state.claims.extendedResourceClaim()
	// Matched like in Reserve.
	allocIndex := 0
	for _, claim := range state.claims.toAllocate() {
		if allocIndex >= len(allocations.allocationResults) {
			break
		}
		allocation := allocations.allocationResults[allocIndex]
		allocIndex++
		if claim == extendedResourceClaim {
			continue
		}
		holds[claim.UID] = heldClaim{
			claim:      types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			allocation: &allocation,
		}
	}
	if len(holds) == 0 {
		return nil
	}
	return &podNomination{nodeName: nodeName, assumed: true, holds: holds}
}

// AddAllocation confirms the candidate of the pod for the node: the devices
// held for a nominated pod (pod.Spec.NodeName empty), or the devices of an
// assumed pod.
func (pl *DynamicResources) AddAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations == nil {
		return framework.AllocationChangeResult{}
	}
	if confirmed := pl.nominations.confirm(pod.UID, nodeName, pod.Spec.NodeName != ""); confirmed != nil {
		klog.FromContext(ctx).V(5).Info("Confirmed allocation", "pod", klog.KObj(pod), "node", nodeName, "assumed", confirmed.assumed, "numHeldClaims", len(confirmed.holds))
	}
	return framework.AllocationChangeResult{}
}

// RemoveAllocation drops the nomination of a nominated pod (pod.Spec.NodeName
// empty) or the devices of an assumed pod. Claims of the victims of a
// nomination are tracked independently of it until they are deallocated.
//
// The devices held for a nominated pod may be free, so pods waiting for
// devices get another chance, unless the pod keeps them because it got
// assumed with them or nominated to the node again. The devices of an
// assumed pod are free if Unreserve released them before they got
// persisted.
func (pl *DynamicResources) RemoveAllocation(ctx context.Context, pod *v1.Pod, nodeName string) framework.AllocationChangeResult {
	if pl.nominations == nil {
		return framework.AllocationChangeResult{}
	}
	logger := klog.FromContext(ctx)
	if pod.Spec.NodeName != "" {
		assumed := pl.nominations.removeAssumed(pod.UID, nodeName)
		if assumed == nil {
			return framework.AllocationChangeResult{}
		}
		for _, held := range assumed.holds {
			claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
			if err != nil || claim.Status.Allocation != nil || pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID) != nil {
				continue
			}
			logger.V(5).Info("Released devices of assumed pod", "pod", klog.KObj(pod), "node", nodeName, "resourceclaim", held.claim)
			return framework.AllocationChangeResult{Events: releaseEvents(claim.UID, held)}
		}
		return framework.AllocationChangeResult{}
	}

	nomination, kept := pl.nominations.removeNomination(pod.UID, nodeName)
	if nomination == nil || len(nomination.holds) == 0 || kept != nil && covers(kept.holds, nomination.holds) {
		return framework.AllocationChangeResult{}
	}
	logger.V(5).Info("Released devices held for pod", "pod", klog.KObj(pod), "node", nodeName)
	for uid, held := range nomination.holds {
		// One event is enough, the queueing hints don't check which
		// claim got deallocated.
		return framework.AllocationChangeResult{Events: releaseEvents(uid, held)}
	}
	return framework.AllocationChangeResult{}
}

// releaseEvents returns the event for a claim whose devices are not used
// anymore, as if the claim got deallocated.
func releaseEvents(uid types.UID, held heldClaim) []framework.AllocationEvent {
	claim := &resourceapi.ResourceClaim{}
	claim.Namespace, claim.Name, claim.UID = held.claim.Namespace, held.claim.Name, uid
	claim.Status.Allocation = held.allocation
	deallocated := claim.DeepCopy()
	deallocated.Status.Allocation = nil
	return []framework.AllocationEvent{{Event: framework.EventResourceClaimHoldReleased, OldObj: claim, NewObj: deallocated}}
}

// covers returns true if have has the same devices for each claim in want.
func covers(have, want map[types.UID]heldClaim) bool {
	for uid, wantClaim := range want {
		haveClaim, ok := have[uid]
		if !ok || !sameDevices(haveClaim.allocation.Devices.Results, wantClaim.allocation.Devices.Results) {
			return false
		}
	}
	return true
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

// isSchedulableAfterAssignedPodDeleted is needed because DefaultPreemption
// gates a pod during asynchronous preemption, and then only its events
// (deleted assigned pods) are considered, through the hints of the plugins
// which rejected the pod.
//
// The pod gets queued when the last of its victims is gone, or when the
// deleted pod lost a pending allocation: that happens without a claim event
// when a pod in Permit or PreBind gets preempted or its binding fails.
func (pl *DynamicResources) isSchedulableAfterAssignedPodDeleted(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	deletedPod, _, err := schedutil.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterAssignedPodDeleted: %w", err)
	}
	if pl.nominations != nil {
		if nomination := pl.nominations.nomination(pod.UID); nomination != nil {
			if _, ok := nomination.victims[deletedPod.UID]; ok {
				for uid, name := range nomination.victims {
					if uid == deletedPod.UID {
						continue
					}
					if victim, err := pl.podLister.Pods(name.Namespace).Get(name.Name); err == nil && victim.UID == uid && victim.Spec.NodeName != "" {
						logger.V(5).Info("preempted pod got deleted, others remain", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "remainingPod", klog.KObj(victim))
						return fwk.QueueSkip, nil
					}
				}
				logger.V(5).Info("last preempted pod got deleted", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod))
				return fwk.Queue, nil
			}
		}
	}
	// The claim controller deallocates the claims of completed pods before
	// they get deleted, and that update already gave waiting pods another
	// chance. Pods in Permit or PreBind come as assumed pods with a node
	// name. Without one, the event is about a pod which lost its nomination
	// and has no pending allocation.
	if deletedPod.Spec.NodeName == "" || deletedPod.Status.Phase == v1.PodSucceeded || deletedPod.Status.Phase == v1.PodFailed {
		return fwk.QueueSkip, nil
	}
	for _, claim := range pl.claimsOfPod(deletedPod) {
		if claim.Status.Allocation == nil && pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID) == nil {
			logger.V(5).Info("deleted pod lost a pending allocation", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "resourceclaim", klog.KObj(claim))
			return fwk.Queue, nil
		}
	}
	return fwk.QueueSkip, nil
}
