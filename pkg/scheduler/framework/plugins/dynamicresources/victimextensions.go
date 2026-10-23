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
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// This file implements framework.VictimExtensions (KEP-5690): a pod does not
// preempt again while claims of pods which got preempted on its nominated
// node are still being deallocated.

var _ framework.VictimExtensions = &DynamicResources{}

// maxClaimsInReason limits how many claims PodEligibleToPreempt names.
const maxClaimsInReason = 3

// AssumeVictim remembers the preempted pod and its allocated claims until
// they are deallocated. Claims which are not going to get deallocated, for
// example because some other pod keeps using them, stop being tracked when
// they get checked.
func (pl *DynamicResources) AssumeVictim(ctx context.Context, victim *v1.Pod) {
	if pl.nominations == nil || victim.Spec.NodeName == "" {
		return
	}
	nodeName := victim.Spec.NodeName
	name := types.NamespacedName{Namespace: victim.Namespace, Name: victim.Name}
	var claims []*victimClaim
	for _, claim := range pl.claimsOfPod(victim) {
		allocation := claim.Status.Allocation
		if allocation == nil {
			// The victim is in Permit or PreBind.
			allocation = pl.draManager.ResourceClaims().GetPendingAllocation(claim.UID)
		}
		if allocation == nil || slices.ContainsFunc(claim.Status.ReservedFor, func(consumer resourceapi.ResourceClaimConsumerReference) bool {
			return consumer.APIGroup != "" || consumer.Resource != "pods"
		}) {
			continue
		}
		claims = append(claims, &victimClaim{
			claim:   types.NamespacedName{Namespace: claim.Namespace, Name: claim.Name},
			uid:     claim.UID,
			results: allocation.Devices.Results,
			victims: map[types.UID]types.NamespacedName{victim.UID: name},
		})
	}
	pl.nominations.assumeVictim(nodeName, victim.UID, name, claims, pl.podGone)
	klog.FromContext(ctx).V(5).Info("Assumed preempted pod", "pod", klog.KObj(victim), "node", nodeName, "numClaims", len(claims))
}

// podGone returns true if the pod does not exist anymore.
func (pl *DynamicResources) podGone(uid types.UID, name types.NamespacedName) bool {
	pod, err := pl.podLister.Pods(name.Namespace).Get(name.Name)
	return err != nil || pod.UID != uid
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
	// The nominated node in the pod may lag behind the PodNominator.
	if nominatedNode := pl.nominations.nominatedNode(pod.UID); nominatedNode != "" {
		nodeNames.Insert(nominatedNode)
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
		// Victims may still be running, for example while asynchronous
		// preemption has not deleted them yet. Victims which share the
		// claim get assumed one after the other.
		if _, ok := victimClaim.victims[consumer.UID]; ok || pl.nominations.isVictim(consumer.UID) {
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
			if !pl.podGone(uid, victim) {
				return true
			}
		}
		return false
	}
	return true
}

// deallocateVictimClaims is called by PostFilter. It deallocates a claim of
// the pod which got released by preemption when nothing else would do it:
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
// The pod gets queued when the last pod which got preempted on its
// nominated node is gone. Devices which an assumed pod allocated come with
// a claim event from RemoveAllocation when the pod gets forgotten.
func (pl *DynamicResources) isSchedulableAfterAssignedPodDeleted(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	deletedPod, _, err := schedutil.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, fmt.Errorf("unexpected object in isSchedulableAfterAssignedPodDeleted: %w", err)
	}
	if pl.nominations == nil {
		return fwk.QueueSkip, nil
	}
	nodeName := pl.nominations.nominatedNode(pod.UID)
	if nodeName == "" || deletedPod.Spec.NodeName != nodeName {
		return fwk.QueueSkip, nil
	}
	victims := pl.nominations.victimsOnNode(nodeName)
	if _, ok := victims[deletedPod.UID]; !ok {
		return fwk.QueueSkip, nil
	}
	for uid, name := range victims {
		if uid == deletedPod.UID {
			continue
		}
		// Victims which are not being deleted on the node got preempted in
		// memory and are not going to be gone.
		if victim, err := pl.podLister.Pods(name.Namespace).Get(name.Name); err == nil && victim.UID == uid && victim.Spec.NodeName == nodeName && victim.DeletionTimestamp != nil {
			logger.V(5).Info("preempted pod got deleted, others on the nominated node remain", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod), "remainingPod", klog.KObj(victim))
			return fwk.QueueSkip, nil
		}
	}
	logger.V(5).Info("last preempted pod on the nominated node got deleted", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod))
	return fwk.Queue, nil
}
