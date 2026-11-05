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
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// This file implements framework.VictimExtensions (KEP-5690). Because a
// preempted pod's ResourceClaims are deallocated asynchronously after the pod
// terminates, preemptors nominated to that node must wait for those claims to
// settle instead of preempting additional pods elsewhere.

var _ framework.VictimExtensions = &DynamicResources{}

const maxClaimsInReason = 3

// AssumeVictim records allocated pod-scoped claims of victim so preemptors
// nominated to victim's node wait for those claims to be deallocated.
//
// Unlike RemovePod, it does not filter out non-node-local devices: AssumeVictim
// has no PreFilter slice index, and waiting for a non-local claim to settle at
// worst briefly delays further preemption.
func (pl *DynamicResources) AssumeVictim(ctx context.Context, victim *v1.Pod) {
	if pl.store == nil || victim.Spec.NodeName == "" {
		return
	}
	logger := klog.FromContext(ctx)
	victimName := types.NamespacedName{Namespace: victim.Namespace, Name: victim.Name}
	for _, claim := range pl.claimsOfPod(victim) {
		allocation := pl.allocationOrPending(claim)
		if allocation == nil || slices.ContainsFunc(claim.Status.ReservedFor, isNonPodConsumer) {
			// Claims reserved for a PodGroup outlive individual member pods.
			continue
		}
		logger.V(5).Info("Tracking claim of preempted pod", "pod", klog.KObj(victim), "resourceclaim", klog.KObj(claim), "node", victim.Spec.NodeName)
		pl.store.assumeVictimClaim(&releasedClaim{
			claim:      types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			uid:        claim.UID,
			nodeName:   victim.Spec.NodeName,
			allocation: allocation,
			pods:       map[types.UID]types.NamespacedName{victim.UID: victimName},
		})
	}
}

// PodEligibleToPreempt returns false while victim claims on pod's nominated
// node (passed in from the in-memory PodNominator, since pod.Status may lag)
// are still awaiting deallocation, avoiding redundant preemptions while devices
// are being freed.
func (pl *DynamicResources) PodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	if pl.store == nil || nodeName == "" || !pl.needsAllocation(pod) {
		return true, ""
	}
	settling := pl.settlingVictimClaims(nodeName, false)
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

// needsAllocation reports whether pod has at least one unallocated claim and
// could therefore benefit from devices freed by preemption.
func (pl *DynamicResources) needsAllocation(pod *v1.Pod) bool {
	for _, podClaim := range pod.Spec.ResourceClaims {
		claimName, _, err := resourceclaim.Name(pod, &podClaim)
		if err != nil {
			// Template-generated claim name is not recorded in pod status yet.
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

// settlingVictimClaims returns names of tracked victim claims on nodeName that
// are still expected to be deallocated (stopping after the first match when
// firstOnly is true), pruning entries whose recorded allocation is already gone.
func (pl *DynamicResources) settlingVictimClaims(nodeName string, firstOnly bool) []string {
	if pl.store == nil {
		return nil
	}
	var settling []string
	for _, claim := range pl.store.victimClaimsOnNode(nodeName) {
		isSettling, final := pl.victimClaimSettling(claim)
		if final {
			pl.store.forgetVictimClaim(claim.uid, claim)
		}
		if !isSettling {
			continue
		}
		settling = append(settling, claim.claim.String())
		if firstOnly {
			break
		}
	}
	return settling
}

// victimClaimSettling reports whether victimClaim is still expected to be
// deallocated without further preemption, and whether the entry is permanently
// obsolete (final) and can be pruned from preemptionStore.
//
// When another consumer currently holds the same allocation (for example, a
// co-victim whose AssumeVictim call has not run yet), settling is false so we
// do not block on it yet, while final is false so the entry stays tracked.
func (pl *DynamicResources) victimClaimSettling(victimClaim *releasedClaim) (settling, final bool) {
	claim, err := pl.draManager.ResourceClaims().Get(victimClaim.claim.Namespace, victimClaim.claim.Name)
	if err != nil || claim.UID != victimClaim.uid {
		return false, true
	}
	allocation := pl.allocationOrPending(claim)
	if allocation == nil || !apiequality.Semantic.DeepEqual(allocation.Devices.Results, victimClaim.allocation.Devices.Results) {
		return false, true
	}
	return pl.deallocatedOncePodsGone(claim, victimClaim), false
}

// deallocatedOncePodsGone reports whether claim (which still carries released's
// allocation) has no remaining active consumers outside released.pods and has
// an actor that will deallocate or delete it.
func (pl *DynamicResources) deallocatedOncePodsGone(claim *resourceapi.ResourceClaim, released *releasedClaim) bool {
	// TODO(troychiu): Before beta, handle victims preempted in Permit or
	// PreBind. Because such a pod is re-queued rather than deleted, it keeps
	// its UID; if it later schedules again with this same claim, its UID in
	// released.pods still matches here, making the claim appear to be
	// settling while the pod runs and blocking preemption on the node.
	// AssumeVictim could record that the victim was unbound so it is ignored
	// once bound again.
	for _, consumer := range claim.Status.ReservedFor {
		if isNonPodConsumer(consumer) {
			return false
		}
		if _, ok := released.pods[consumer.UID]; ok {
			// Recorded victims may still be running (e.g. awaiting async deletion).
			continue
		}
		if !pl.podStoppedUsingClaims(claim.Namespace, consumer) {
			return false
		}
	}
	if claim.Status.Allocation != nil && len(claim.Status.ReservedFor) == 0 {
		// The ResourceClaim controller deallocates only on the transition
		// that empties ReservedFor. If ReservedFor is already empty (for
		// example, Unreserve ran after PreBind was canceled), the claim will
		// be freed only if a victim owns it (garbage collection on pod
		// deletion) or still exists to deallocate it in PostFilter.
		for uid, pod := range released.pods {
			if slices.ContainsFunc(claim.OwnerReferences, func(owner metav1.OwnerReference) bool {
				return owner.UID == uid && owner.Controller != nil && *owner.Controller
			}) {
				return true
			}
			if existing, err := pl.podLister.Pods(pod.Namespace).Get(pod.Name); err == nil && existing.UID == uid {
				return true
			}
		}
		return false
	}
	return true
}

// deallocateVictimClaims deallocates a tracked victim claim referenced by pod
// during PostFilter when no other pod uses it. When preemption cancels a pod in
// Permit or PreBind, Unreserve clears ReservedFor without deallocating the
// claim, so the ResourceClaim controller will not deallocate it; the re-queued
// victim's own PostFilter cleans it up here.
func (pl *DynamicResources) deallocateVictimClaims(ctx context.Context, pod *v1.Pod, state *stateData) (bool, *fwk.Status) {
	if pl.store == nil {
		return false, nil
	}
	logger := klog.FromContext(ctx)
	for _, claim := range state.claims.allUserClaims() {
		if claim.Status.Allocation == nil || !pl.store.isVictimClaim(claim.UID) {
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

// isSchedulableAfterAssignedPodDeleted wakes a nominated preemptor when a pod
// with claims is deleted from its nominated node.
//
// While asynchronous preemption is in flight, DefaultPreemption blocks the
// preemptor in PreEnqueue and the queue ignores ResourceClaim events for it,
// consulting only AssignedPod/Delete hints on plugins that rejected the pod.
// Returning Queue here lets the pod pass PreEnqueue once its victims are
// deleted (even if their claims are still deallocating). Because victims are
// not tracked per preemptor, any pod with claims deleted from the nominated
// node triggers a wakeup.
func (pl *DynamicResources) isSchedulableAfterAssignedPodDeleted(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	deletedPod, _, err := schedutil.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterAssignedPodDeleted: %w", err)
	}
	// Skip terminal pods (the ResourceClaim controller already deallocated
	// their claims on completion) and unassigned pods whose nomination was
	// cleared. An assumed pod clearing its nomination carries the assumed
	// NodeName and may trigger one benign wakeup on that node.
	if deletedPod.Spec.NodeName == "" || len(deletedPod.Spec.ResourceClaims) == 0 ||
		deletedPod.Status.Phase == v1.PodSucceeded || deletedPod.Status.Phase == v1.PodFailed {
		return fwk.QueueSkip, nil
	}
	if !pl.nominatedTo(pod, deletedPod.Spec.NodeName) {
		return fwk.QueueSkip, nil
	}
	logger.V(5).Info("pod with claims got deleted from the nominated node", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "node", deletedPod.Spec.NodeName)
	return fwk.Queue, nil
}

func (pl *DynamicResources) nominatedTo(pod *v1.Pod, nodeName string) bool {
	return pod.Status.NominatedNodeName == nodeName
}
