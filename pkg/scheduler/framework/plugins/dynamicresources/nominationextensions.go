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
	"slices"
	"strings"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// This file implements framework.NominationExtensions (KEP-5690): a
// preemptor does not preempt again while claims of the pods it preempted
// are still being deallocated.

var _ framework.NominationExtensions = &DynamicResources{}

// maxClaimsInReason limits how many claims PodEligibleToPreempt names.
const maxClaimsInReason = 3

// AddNominatedPod remembers the victims of the pod, which of their claims get
// deallocated once they are gone, and which devices the pod is going to get
// on the node, so that pods with lower or equal priority don't get them
// first.
func (pl *DynamicResources) AddNominatedPod(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string, victims []*v1.Pod) {
	if pl.nominations == nil {
		return
	}
	state, err := getStateData(cs)
	if err != nil || state.preemption == nil || !state.preemption.root.canRelease {
		return
	}
	logger := klog.FromContext(ctx)

	p := state.preemption.clone()
	nomination := &podNomination{
		nodeName: nodeName,
		priority: corev1helpers.PodPriority(pod),
		victims:  make(map[types.UID]types.NamespacedName, len(victims)),
	}
	for _, victim := range victims {
		nomination.victims[victim.UID] = types.NamespacedName{Namespace: victim.Namespace, Name: victim.Name}
		pl.removePod(logger, p, victim, victim.Spec.NodeName)
	}
	nomination.holds = pl.computeHold(ctx, state, p, nodeName)
	claims := make(map[string][]*victimClaim)
	for _, released := range p.sim.released {
		claim := &victimClaim{
			claim:   types.NamespacedName{Namespace: released.claim.Namespace, Name: released.claim.Name},
			uid:     released.claim.UID,
			results: released.allocation.Devices.Results,
			victims: make(map[types.UID]types.NamespacedName, len(released.consumers)),
		}
		for uid := range released.consumers {
			claim.victims[uid] = nomination.victims[uid]
		}
		claims[released.nodeName] = append(claims[released.nodeName], claim)
	}
	replaced := pl.nominations.add(pod.UID, nomination, claims)
	logger.V(5).Info("Recorded nomination", "pod", klog.KObj(pod), "node", nodeName, "numVictimClaims", len(p.sim.released), "numHeldClaims", len(nomination.holds))
	if replaced != nil && !sameHolds(replaced.holds, nomination.holds) {
		pl.nominations.released(pod.UID, replaced)
	}

	// If the pod got deleted before its nomination was recorded, then
	// RemoveNominatedPod was called too early.
	if current, err := pl.podLister.Pods(pod.Namespace).Get(pod.Name); err != nil || current.UID != pod.UID {
		pl.RemoveNominatedPod(ctx, pod, nodeName, false)
	}
}

// RemoveNominatedPod forgets the nomination of the pod. Claims of its
// victims stay tracked until they are deallocated, because other pods
// nominated to the same node also have to wait for them.
//
// The devices held for the pod may be free, so pods waiting for devices get
// another chance, unless the pod got assumed with exactly those devices.
func (pl *DynamicResources) RemoveNominatedPod(ctx context.Context, pod *v1.Pod, nodeName string, preemptionFailed bool) {
	if pl.nominations == nil {
		return
	}
	nomination := pl.nominations.remove(pod.UID, nodeName, preemptionFailed)
	if nomination == nil || len(nomination.holds) == 0 {
		return
	}
	if pod.Spec.NodeName == nomination.nodeName && pl.allocatedAsHeld(nomination) {
		return
	}
	klog.FromContext(ctx).V(5).Info("Released devices held for pod", "pod", klog.KObj(pod), "node", nomination.nodeName)
	pl.nominations.released(pod.UID, nomination)
}

func sameHolds(a, b map[types.UID]heldClaim) bool {
	if len(a) != len(b) {
		return false
	}
	for uid, heldA := range a {
		heldB, ok := b[uid]
		if !ok || !sameDevices(heldA.allocation.Devices.Results, heldB.allocation.Devices.Results) {
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

// allocatedAsHeld returns true if the pending allocations of all held claims
// have the held devices.
func (pl *DynamicResources) allocatedAsHeld(nomination *podNomination) bool {
	for uid, held := range nomination.holds {
		pending := pl.draManager.ResourceClaims().GetPendingAllocation(uid)
		if pending == nil || !sameDevices(pending.Devices.Results, held.allocation.Devices.Results) {
			return false
		}
	}
	return true
}

// PodEligibleToPreempt returns false while claims of preempted pods on the
// node of the pod are still being deallocated: their devices may be what the
// pod is waiting for, so preempting more pods now could be unnecessary.
func (pl *DynamicResources) PodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	if pl.nominations == nil || !pl.needsAllocation(pod) {
		return true, ""
	}
	nodeNames := sets.New[string]()
	if nodeName != "" {
		nodeNames.Insert(nodeName)
	}
	// The nominated node in the pod may lag behind.
	if nomination := pl.nominations.nomination(pod.UID); nomination != nil {
		nodeNames.Insert(nomination.nodeName)
	}
	var settling []string
	for _, nodeName := range sets.List(nodeNames) {
		settling = append(settling, pl.settlingVictimClaims(nodeName, false)...)
	}
	if len(settling) == 0 {
		return true, ""
	}
	slices.Sort(settling)
	claims := strings.Join(settling[:min(len(settling), maxClaimsInReason)], ", ")
	if len(settling) > maxClaimsInReason {
		claims += fmt.Sprintf(" and %d more", len(settling)-maxClaimsInReason)
	}
	return false, fmt.Sprintf("not eligible while ResourceClaims of preempted pods on the nominated node are being deallocated: %s.", claims)
}

// needsAllocation returns true if the pod references a claim which is not
// allocated yet. Only such pods can use devices that preemption frees.
func (pl *DynamicResources) needsAllocation(pod *v1.Pod) bool {
	for _, podClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &podClaim)
		if err != nil {
			// The claim does not exist yet.
			return true
		}
		if claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		if err != nil || claim.Status.Allocation == nil {
			return true
		}
	}
	return false
}

// settlingVictimClaims returns the claims of preempted pods on the node which
// are still expected to get deallocated, or just the first one, and forgets
// those which are not.
func (pl *DynamicResources) settlingVictimClaims(nodeName string, first bool) []string {
	if pl.nominations == nil {
		return nil
	}
	var settling []string
	for _, claim := range pl.nominations.victimClaimsOnNode(nodeName) {
		if !pl.victimClaimSettling(claim) {
			pl.nominations.forget(claim.uid, claim)
			continue
		}
		settling = append(settling, claim.claim.String())
		if first {
			break
		}
	}
	return settling
}

// victimClaimSettling returns true while the claim of preempted pods is
// expected to get deallocated without further preemption.
func (pl *DynamicResources) victimClaimSettling(victimClaim *victimClaim) bool {
	claim, err := pl.draManager.ResourceClaims().Get(victimClaim.claim.Namespace, victimClaim.claim.Name)
	if err != nil || claim.UID != victimClaim.uid {
		return false
	}
	allocation := claim.Status.Allocation
	if allocation == nil {
		allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
	}
	if allocation == nil || !apiequality.Semantic.DeepEqual(allocation.Devices.Results, victimClaim.results) {
		return false
	}
	for _, consumer := range claim.Status.ReservedFor {
		if consumer.APIGroup != "" || consumer.Resource != "pods" {
			return false
		}
		if _, ok := victimClaim.victims[consumer.UID]; ok {
			// Victims may still be running, for example while
			// asynchronous preemption has not deleted them yet.
			continue
		}
		if !pl.podStoppedUsingClaims(claim.Namespace, consumer) {
			// Some other pod started to use it.
			return false
		}
	}
	if claim.Status.Allocation != nil && len(claim.Status.ReservedFor) == 0 {
		// The claim controller only deallocates when it removes the
		// last consumer, which happened earlier, for example when a
		// victim whose PreBind got cancelled was unreserved. Such a
		// claim only gets deallocated if a victim owns it (garbage
		// collection) or if a victim still exists (its PostFilter
		// deallocates it).
		for uid, victim := range victimClaim.victims {
			if slices.ContainsFunc(claim.OwnerReferences, func(owner metav1.OwnerReference) bool {
				return owner.UID == uid && owner.Controller != nil && *owner.Controller
			}) {
				return true
			}
			if pod, err := pl.podLister.Pods(victim.Namespace).Get(victim.Name); err == nil && pod.UID == uid {
				return true
			}
		}
		return false
	}
	return true
}

// deallocateVictimClaims is called by PostFilter. It deallocates a claim of
// the pod which got released for a preemptor when nothing else would do it:
// a pod whose PreBind was cancelled by preemption leaves its claim
// allocated with an empty ReservedFor, which the claim controller ignores.
func (pl *DynamicResources) deallocateVictimClaims(ctx context.Context, pod *v1.Pod, state *stateData) (bool, *fwk.Status) {
	if pl.nominations == nil {
		return false, nil
	}
	logger := klog.FromContext(ctx)
	for _, claim := range state.claims.allUserClaims() {
		if claim.Status.Allocation == nil || !pl.nominations.isVictimClaim(claim.UID) {
			continue
		}
		if len(claim.Status.ReservedFor) > 1 ||
			len(claim.Status.ReservedFor) == 1 && claim.Status.ReservedFor[0].UID != pod.UID {
			continue
		}
		claim := claim.DeepCopy()
		claim.Status.ReservedFor = nil
		claim.Status.Allocation = nil
		claim.Status.Devices = nil
		logger.V(5).Info("Deallocation of ResourceClaim released by preemption", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		if _, err := pl.clientset.ResourceV1().ResourceClaims(claim.Namespace).UpdateStatus(ctx, claim, metav1.UpdateOptions{}); err != nil {
			return true, statusError(logger, err)
		}
		return true, fwk.NewStatus(fwk.Unschedulable, "deallocation of ResourceClaim completed")
	}
	return false, nil
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
