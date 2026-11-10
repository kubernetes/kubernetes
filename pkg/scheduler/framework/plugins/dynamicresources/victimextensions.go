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
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/ptr"
)

// This file implements framework.VictimExtensions (KEP-5690): a pod does
// not preempt again while claims of pods preempted on its nominated node
// are still being deallocated.

var _ framework.VictimExtensions = &DynamicResources{}

// maxClaimsInReason limits how many claims PodEligibleToPreempt names.
const maxClaimsInReason = 3

// AssumeVictim tracks the allocated claims of the victim on its node until
// they got deallocated. Claims which don't get deallocated, for example
// because other pods keep using them, are forgotten by settlingVictimClaims.
func (pl *DynamicResources) AssumeVictim(ctx context.Context, victim *v1.Pod) {
	if pl.victimClaims == nil || victim.Spec.NodeName == "" {
		return
	}
	logger := klog.FromContext(ctx)
	now := pl.victimClaims.clock.Now()
	graceSeconds := max(ptr.Deref(victim.Spec.TerminationGracePeriodSeconds, v1.DefaultTerminationGracePeriodSeconds), 0)
	gracePeriod := time.Duration(graceSeconds) * time.Second
	victims := map[types.UID]types.NamespacedName{victim.UID: {Namespace: victim.Namespace, Name: victim.Name}}
	for _, claim := range pl.claimsOfPod(victim) {
		allocation := claim.Status.Allocation
		if allocation == nil {
			// The victim is in Permit or PreBind.
			allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
		}
		if allocation == nil {
			continue
		}
		pl.victimClaims.add(victim.Spec.NodeName, &victimClaim{
			claim:       types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			uid:         claim.UID,
			results:     allocation.Devices.Results,
			victims:     victims,
			addedAt:     now,
			gracePeriod: gracePeriod,
		})
		logger.V(5).Info("Tracking ResourceClaim of preempted pod", "pod", klog.KObj(victim), "node", victim.Spec.NodeName, "resourceclaim", klog.KObj(claim))
	}
}

// PodEligibleToPreempt returns false while claims of preempted pods on the
// node of the pod are still being deallocated: their devices may be what the
// pod is waiting for, so preempting more pods now could be unnecessary.
func (pl *DynamicResources) PodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	if pl.victimClaims == nil || !pl.needsAllocation(pod) {
		return true, ""
	}
	logger := klog.FromContext(ctx)
	nodeNames := sets.New[string]()
	if nodeName != "" {
		nodeNames.Insert(nodeName)
	}
	// The nominated node in the pod may lag behind.
	if pl.nominations != nil {
		if nomination := pl.nominations.nomination(pod.UID); nomination != nil {
			nodeNames.Insert(nomination.nodeName)
		}
	}
	var settling []string
	for _, nodeName := range sets.List(nodeNames) {
		settling = append(settling, pl.settlingVictimClaims(logger, nodeName, false)...)
	}
	if len(settling) == 0 {
		return true, ""
	}
	if schedmetrics.DRAPreemptionDeferredTotal != nil {
		var profile string
		if pl.fh != nil {
			profile = pl.fh.ProfileName()
		}
		schedmetrics.DRAPreemptionDeferredTotal.WithLabelValues(profile).Inc()
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
func (pl *DynamicResources) settlingVictimClaims(logger klog.Logger, nodeName string, first bool) []string {
	if pl.victimClaims == nil {
		return nil
	}
	var settling []string
	for _, claim := range pl.victimClaims.onNode(nodeName) {
		if !pl.victimClaimSettling(logger, claim) {
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
// expected to get deallocated without further preemption, and forgets it
// otherwise.
func (pl *DynamicResources) victimClaimSettling(logger klog.Logger, victimClaim *victimClaim) bool {
	settling, timedOut := pl.evalVictimClaimSettling(victimClaim)
	if !settling {
		if pl.victimClaims.forget(victimClaim.uid, victimClaim) && timedOut {
			logger.V(2).Info("Stopped waiting for ResourceClaim of preempted pods to be deallocated after timeout", "resourceclaim", victimClaim.claim, "timeout", victimClaimSettlingTimeout)
			if schedmetrics.DRAPreemptionVictimClaimTimeoutsTotal != nil {
				schedmetrics.DRAPreemptionVictimClaimTimeoutsTotal.Inc()
			}
		}
		return false
	}
	return true
}

func (pl *DynamicResources) evalVictimClaimSettling(victimClaim *victimClaim) (settling, timedOut bool) {
	claim, err := pl.draManager.ResourceClaims().Get(victimClaim.claim.Namespace, victimClaim.claim.Name)
	if err != nil || claim.UID != victimClaim.uid {
		return false, false
	}
	allocation := claim.Status.Allocation
	if allocation == nil {
		allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
	}
	if allocation == nil || !apiequality.Semantic.DeepEqual(allocation.Devices.Results, victimClaim.results) {
		return false, false
	}
	for _, consumer := range claim.Status.ReservedFor {
		if consumer.APIGroup != "" || consumer.Resource != "pods" {
			return false, false
		}
		if _, ok := victimClaim.victims[consumer.UID]; ok {
			// Victims may still be running, for example while
			// asynchronous preemption has not deleted them yet.
			continue
		}
		if !pl.podStoppedUsingClaims(claim.Namespace, consumer) {
			// Some other pod started to use it.
			return false, false
		}
	}
	if claim.Status.Allocation != nil && len(claim.Status.ReservedFor) == 0 {
		// The claim controller only deallocates when it removes the
		// last consumer, which happened earlier, for example when a
		// victim whose PreBind got cancelled was unreserved. Such a
		// claim only gets deallocated if a victim owns it (garbage
		// collection) or if a victim still exists (its PostFilter
		// deallocates it).
		canDeallocate := false
		for uid, victim := range victimClaim.victims {
			if slices.ContainsFunc(claim.OwnerReferences, func(owner metav1.OwnerReference) bool {
				return owner.UID == uid && owner.Controller != nil && *owner.Controller
			}) {
				canDeallocate = true
				break
			}
			if pod, err := pl.podLister.Pods(victim.Namespace).Get(victim.Name); err == nil && pod.UID == uid {
				canDeallocate = true
				break
			}
		}
		if !canDeallocate {
			return false, false
		}
		if !victimClaim.addedAt.IsZero() && pl.victimClaims.clock.Since(victimClaim.addedAt) >= victimClaimSettlingTimeout {
			return false, true
		}
		return true, false
	}
	if pl.victimClaimTimedOut(victimClaim) {
		return false, true
	}
	return true, false
}

func (pl *DynamicResources) victimClaimTimedOut(victimClaim *victimClaim) bool {
	if pl.victimClaims == nil || victimClaim.addedAt.IsZero() {
		return false
	}
	now := pl.victimClaims.clock.Now()
	maxTotalWait := victimClaim.gracePeriod + victimClaimSettlingTimeout
	for uid, victim := range victimClaim.victims {
		if pod, err := pl.podLister.Pods(victim.Namespace).Get(victim.Name); err == nil &&
			pod.UID == uid &&
			pod.Spec.NodeName != "" &&
			pod.DeletionTimestamp != nil &&
			pod.Status.Phase != v1.PodSucceeded &&
			pod.Status.Phase != v1.PodFailed {
			victimClaim.settlingSince.Store(0)
			return now.Sub(victimClaim.addedAt) >= maxTotalWait
		}
	}
	unixNano := now.UnixNano()
	since := victimClaim.settlingSince.Load()
	if since == 0 {
		if victimClaim.settlingSince.CompareAndSwap(0, unixNano) {
			since = unixNano
		} else {
			since = victimClaim.settlingSince.Load()
		}
	}
	return (since != 0 && now.Sub(time.Unix(0, since)) >= victimClaimSettlingTimeout) ||
		now.Sub(victimClaim.addedAt) >= maxTotalWait
}

// deallocateVictimClaims is called by PostFilter. It deallocates a claim of
// the pod which got released for a preemptor when nothing else would do it:
// a pod whose PreBind was cancelled by preemption leaves its claim
// allocated with an empty ReservedFor, which the claim controller ignores.
func (pl *DynamicResources) deallocateVictimClaims(ctx context.Context, pod *v1.Pod, state *stateData) (bool, *fwk.Status) {
	if pl.victimClaims == nil {
		return false, nil
	}
	logger := klog.FromContext(ctx)
	for _, claim := range state.claims.allUserClaims() {
		if claim.Status.Allocation == nil || !pl.victimClaims.isTracked(claim.UID) {
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
