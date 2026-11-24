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
	"errors"
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
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

// This file implements framework.VictimExtensions (KEP-5690). Because a
// preempted pod's ResourceClaims are deallocated asynchronously after the pod
// terminates, preemptors nominated to that node must wait for those claims to
// settle instead of preempting additional pods elsewhere.

var _ framework.VictimExtensions = &DynamicResources{}

const maxClaimsInReason = 3

// AssumeVictim records allocated claims owned by victim (matching alpha's
// ownership scope in releasable) so preemptors nominated to victim's node wait
// for them to be deallocated.
//
// Unlike RemovePod, it does not filter out non-node-local devices or
// reservedForOthers. AssumeVictim has no PreFilter slice index and waiting for
// a non-local claim at worst briefly delays further preemption, while shared
// reservations are checked dynamically in victimClaimSettling.
func (pl *DynamicResources) AssumeVictim(ctx context.Context, victim *v1.Pod) {
	if pl.preemptionTracker == nil || victim.Spec.NodeName == "" {
		return
	}
	logger := klog.FromContext(ctx)
	for _, claim := range pl.claimsOfPod(victim) {
		allocation := pl.allocationOrPending(claim)
		if allocation == nil || !metav1.IsControlledBy(claim, victim) {
			continue
		}
		logger.V(5).Info("Tracking claim of preempted pod", "pod", klog.KObj(victim), "resourceclaim", klog.KObj(claim), "node", victim.Spec.NodeName)
		pl.preemptionTracker.assumeVictimClaim(&releasedClaim{
			claim:            types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			uid:              claim.UID,
			nodeName:         victim.Spec.NodeName,
			allocatedDevices: &allocation.Devices,
			podUID:           victim.UID,
		})
	}
}

// PodEligibleToPreempt returns false while victim claims on pod's nominated
// node (passed in from the in-memory PodNominator, since pod.Status may lag)
// are still awaiting deallocation, avoiding redundant preemptions while devices
// are being freed.
func (pl *DynamicResources) PodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	if pl.preemptionTracker == nil || nodeName == "" || !pl.needsAllocation(pod) {
		return true, ""
	}
	settling := pl.settlingVictimClaims(nodeName, false /* firstOnly */)
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
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &podClaim)
		if err != nil {
			// Template-generated claim name is not recorded in pod status yet.
			return true
		}
		if claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		if err != nil || claim.Status.Allocation == nil ||
			mustCheckOwner && resourceclaim.IsForPod(pod, claim, pl.fts.EnableDRAWorkloadResourceClaims) != nil {
			return true
		}
	}
	return false
}

// settlingVictimClaims returns names of tracked victim claims on nodeName that
// are still expected to be deallocated (stopping after the first match when
// firstOnly is true), pruning entries whose recorded allocation is already gone.
func (pl *DynamicResources) settlingVictimClaims(nodeName string, firstOnly bool) []string {
	// TODO(https://github.com/kubernetes/enhancements/issues/5690): Before
	// beta, snapshot settling claims in PreFilter (which requires passing
	// CycleState to PodEligibleToPreempt). If a victim claim is deallocated
	// after PreFilter snapshots AllocatedState, Filter still sees its devices
	// as allocated while live lookups here see nothing settling, letting the
	// preemptor preempt again on another node.
	if pl.preemptionTracker == nil {
		return nil
	}
	var settling []string
	for _, claim := range pl.preemptionTracker.victimClaimsOnNode(nodeName) {
		isSettling, final := pl.victimClaimSettling(claim)
		if final {
			pl.preemptionTracker.forgetVictimClaim(claim.uid, claim)
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
// obsolete (final) and can be pruned from preemptionTracker.
func (pl *DynamicResources) victimClaimSettling(victimClaim *releasedClaim) (settling, final bool) {
	claim, err := pl.draManager.ResourceClaims().Get(victimClaim.claim.Namespace, victimClaim.claim.Name)
	if err != nil || claim.UID != victimClaim.uid {
		return false, true
	}
	allocation := pl.allocationOrPending(claim)
	if allocation == nil || !apiequality.Semantic.DeepEqual(allocation.Devices.Results, victimClaim.allocatedDevices.Results) {
		return false, true
	}
	// The victim owns the claim, so it will be deleted with the victim (or
	// deallocated in PostFilter if PreBind preemption requeued the victim).
	// Another consumer sharing the claim by name keeps it allocated; don't
	// wait while that consumer holds it, but keep tracking the entry.
	return !reservedForOthers(claim, victimClaim.podUID), false
}

// isVictimClaimSettling reports whether claim is a tracked victim claim still
// awaiting deallocation, pruning obsolete entries from preemptionTracker.
func (pl *DynamicResources) isVictimClaimSettling(claim *resourceapi.ResourceClaim) bool {
	if pl.preemptionTracker == nil {
		return false
	}
	vc := pl.preemptionTracker.victimClaim(claim.UID)
	if vc == nil {
		return false
	}
	settling, final := pl.victimClaimSettling(vc)
	if final {
		pl.preemptionTracker.forgetVictimClaim(vc.uid, vc)
	}
	return settling
}

// deallocateVictimClaims is the PostFilter fallback when Unreserve leaves a
// PreBind-preempted pod's claim allocated (for example, on conflict, in a
// PodGroup, or for an allocation from an earlier cycle). Because the
// ResourceClaim controller will not deallocate it once ReservedFor is cleared,
// the re-queued victim deallocates it here if no other pod uses it.
func (pl *DynamicResources) deallocateVictimClaims(ctx context.Context, pod *v1.Pod, state *stateData) (bool, *fwk.Status) {
	if pl.preemptionTracker == nil {
		return false, fwk.NewStatus(fwk.Unschedulable)
	}
	logger := klog.FromContext(ctx)
	for _, claim := range state.claims.allUserClaims() {
		if claim.Status.Allocation == nil || !pl.isVictimClaimSettling(claim) {
			continue
		}
		// isVictimClaimSettling checks against victimClaim.podUID; guard
		// against a different pod sharing the claim by name while the victim
		// still reserves it.
		if reservedForOthers(claim, pod.UID) {
			continue
		}
		logger.V(5).Info("Deallocation of ResourceClaim released by preemption", "pod", klog.KObj(pod), "resourceclaim", klog.KObj(claim))
		if _, err := pl.deallocateClaim(ctx, claim); err != nil {
			return true, statusError(logger, err)
		}
		return true, fwk.NewStatus(fwk.Unschedulable, "deallocation of ResourceClaim completed")
	}
	return false, fwk.NewStatus(fwk.Unschedulable)
}

// isSchedulableAfterAssignedPodDeleted wakes waiting pods on AssignedPod/Delete
// in two cases:
//
//  1. Preemptor gated in DefaultPreemption.PreEnqueue during async preemption:
//     while gated, the queue ignores ResourceClaim events for the pod and only
//     checks AssignedPod/Delete hints, so deleting any pod with claims from
//     the preemptor's nominated node wakes it to pass PreEnqueue.
//  2. Forgotten assumed pod: if an assumed pod fails in Permit, PreBind, or
//     Bind, Unreserve drops its pending claim allocations without emitting a
//     ResourceClaim event, so AssignedPod/Delete is the only signal that those
//     in-flight devices are free again.
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
	if pl.nominatedTo(pod, deletedPod.Spec.NodeName) {
		logger.V(5).Info("pod with claims got deleted from the nominated node", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "node", deletedPod.Spec.NodeName)
		return fwk.Queue, nil
	}
	if pl.hasUnallocatedClaim(deletedPod) {
		logger.V(5).Info("deleted pod has a claim without allocation", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "node", deletedPod.Spec.NodeName)
		return fwk.Queue, nil
	}
	return fwk.QueueSkip, nil
}

// hasUnallocatedClaim reports whether pod references a claim with no committed
// or pending allocation — the state Unreserve leaves after an assumed pod fails
// before binding. Missing claims also count because if the pod was deleted
// before PreBind committed the allocation, GC may have deleted the unallocated
// claim, whose Delete event does not wake waiting pods.
func (pl *DynamicResources) hasUnallocatedClaim(pod *v1.Pod) bool {
	for _, podClaim := range pod.Spec.ResourceClaims {
		claimName, mustCheckOwner, err := resourceclaim.Name(pod, &podClaim)
		if err != nil || claimName == nil {
			continue
		}
		claim, err := pl.draManager.ResourceClaims().Get(pod.Namespace, *claimName)
		switch {
		case errors.Is(err, assumecache.ErrNotFound):
			return true
		case err != nil, mustCheckOwner && resourceclaim.IsForPod(pod, claim, pl.fts.EnableDRAWorkloadResourceClaims) != nil:
			continue
		case pl.allocationOrPending(claim) == nil:
			return true
		}
	}
	return false
}

// nominatedTo checks both pod.Status.NominatedNodeName and any active hold in
// preemptionTracker, since pod.Status updates lag behind in-memory nominations.
func (pl *DynamicResources) nominatedTo(pod *v1.Pod, nodeName string) bool {
	if pod.Status.NominatedNodeName == nodeName {
		return true
	}
	if pl.preemptionTracker == nil {
		return false
	}
	hold := pl.preemptionTracker.hold(pod.UID)
	return hold != nil && hold.nodeName == nodeName
}
