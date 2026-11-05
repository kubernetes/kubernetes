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
