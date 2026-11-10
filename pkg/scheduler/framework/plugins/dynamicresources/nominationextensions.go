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

// This file implements framework.NominationExtensions (KEP-5690): devices
// which preemption freed get held for the preemptor while the PodNominator
// nominates it to the node.

var _ framework.NominationExtensions = &DynamicResources{}

// PrepareNomination returns the nomination of the pod to the node with what
// the scheduling cycle selected for it: the devices which the preemption
// simulation found for it with its victims removed, and the victims. It
// returns nil if preemption cannot release devices for the pod.
func (pl *DynamicResources) PrepareNomination(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) any {
	if pl.nominations == nil {
		return nil
	}
	state, err := getStateData(cs)
	if err != nil {
		return nil
	}
	nomination := pl.nominationFor(ctx, state, pod, nodeName)
	if nomination == nil {
		return nil
	}
	klog.FromContext(ctx).V(5).Info("Prepared nomination", "pod", klog.KObj(pod), "node", nodeName, "numHeldClaims", len(nomination.holds))
	return nomination
}

// nominationFor returns the nomination of a pod which gets nominated by
// preemption, nil if preemption cannot release devices for it.
func (pl *DynamicResources) nominationFor(ctx context.Context, state *stateData, pod *v1.Pod, nodeName string) *podNomination {
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

// AddNomination records the nomination which PrepareNomination returned. A
// nomination to the same node again replaces the previous one, whose devices
// may be free unless the new one holds them too.
//
// Without a prepared nomination, for example for one read from the API,
// nothing changes. If an update of the pod object nominated the pod to the
// node right before the scheduler did, that update got the prepared
// nomination, and the scheduler's nomination comes without one.
func (pl *DynamicResources) AddNomination(ctx context.Context, pod *v1.Pod, nodeName string, prepared any) framework.NominationChangeResult {
	nomination, _ := prepared.(*podNomination)
	if pl.nominations == nil || nomination == nil || nomination.nodeName != nodeName {
		return framework.NominationChangeResult{}
	}
	klog.FromContext(ctx).V(5).Info("Added nomination", "pod", klog.KObj(pod), "node", nodeName, "numHeldClaims", len(nomination.holds))
	return pl.releasedDevices(ctx, pod, pl.nominations.add(pod.UID, nomination), nomination.holds)
}

// RemoveNomination drops the nomination of the pod. Claims of the victims of
// the nomination are tracked independently of it until they are
// deallocated.
func (pl *DynamicResources) RemoveNomination(ctx context.Context, pod *v1.Pod, nodeName string) framework.NominationChangeResult {
	if pl.nominations == nil {
		return framework.NominationChangeResult{}
	}
	return pl.releasedDevices(ctx, pod, pl.nominations.remove(pod.UID, nodeName), nil)
}

// releasedDevices returns an event for the devices held for a nomination
// which ended or got replaced, so that pods waiting for devices get another
// chance, unless the pod keeps them. It keeps them if the nomination which
// replaces this one holds them for the same claims (kept), or if its claims
// use them: the nomination of an assumed pod ends after Reserve, so the
// devices which it reserved are pending allocations of its claims then.
func (pl *DynamicResources) releasedDevices(ctx context.Context, pod *v1.Pod, nomination *podNomination, kept map[types.UID]heldClaim) framework.NominationChangeResult {
	if nomination == nil {
		return framework.NominationChangeResult{}
	}
	for uid, held := range nomination.holds {
		if keptClaim, ok := kept[uid]; ok && sameDevices(keptClaim.allocation.Devices.Results, held.allocation.Devices.Results) {
			continue
		}
		if pl.usesHeldDevices(uid, held) {
			continue
		}
		klog.FromContext(ctx).V(5).Info("Released devices held for pod", "pod", klog.KObj(pod), "node", nomination.nodeName, "resourceclaim", held.claim)
		// One event is enough, the queueing hints don't check which
		// claim got deallocated.
		return framework.NominationChangeResult{Events: releaseEvents(uid, held)}
	}
	return framework.NominationChangeResult{}
}

// usesHeldDevices returns true if the claim has the held devices, as a
// pending allocation, like after Reserve, or allocated.
func (pl *DynamicResources) usesHeldDevices(uid types.UID, held heldClaim) bool {
	allocation := pl.draManager.ResourceClaims().GetPendingAllocation(uid)
	if allocation == nil {
		claim, err := pl.draManager.ResourceClaims().Get(held.claim.Namespace, held.claim.Name)
		if err != nil || claim.UID != uid {
			return false
		}
		allocation = claim.Status.Allocation
	}
	return allocation != nil && sameDevices(allocation.Devices.Results, held.allocation.Devices.Results)
}

// releaseEvents returns the event for a claim whose devices are not used
// anymore, as if the claim got deallocated.
func releaseEvents(uid types.UID, held heldClaim) []framework.NominationEvent {
	claim := &resourceapi.ResourceClaim{}
	claim.Namespace, claim.Name, claim.UID = held.claim.Namespace, held.claim.Name, uid
	claim.Status.Allocation = held.allocation
	deallocated := claim.DeepCopy()
	deallocated.Status.Allocation = nil
	return []framework.NominationEvent{{Event: framework.EventResourceClaimHoldReleased, OldObj: claim, NewObj: deallocated}}
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
//
// Like AddNomination and RemoveNomination, it runs while the scheduling queue
// is locked, so the same lock-order rules apply (see
// framework.NominationExtensions).
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
