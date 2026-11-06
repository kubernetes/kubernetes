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
// which preemption freed get held for the preemptor on its nominated node.

var _ framework.NominationExtensions = &DynamicResources{}

// AddNominatedPod remembers the victims with claims which the preemption
// simulation removed for the node and which devices the pod is going to get
// there, so that pods with lower or equal priority don't get them first.
// Claims of the victims get tracked by AssumeVictim.
func (pl *DynamicResources) AddNominatedPod(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if pl.nominations == nil {
		return
	}
	state, err := getStateData(cs)
	if err != nil || state.preemption == nil || !state.preemption.root.canRelease {
		return
	}
	logger := klog.FromContext(ctx)

	sim := state.preemption.simulationFor(nodeName)
	if sim == nil {
		sim = state.preemption
	}
	p := sim.clone()
	nomination := &podNomination{
		nodeName: nodeName,
		priority: corev1helpers.PodPriority(pod),
		victims:  maps.Clone(p.sim.removedPods),
	}
	nomination.holds = pl.computeHold(ctx, state, p, nodeName)
	replaced := pl.nominations.add(pod.UID, nomination)
	logger.V(5).Info("Recorded nomination", "pod", klog.KObj(pod), "node", nodeName, "numHeldClaims", len(nomination.holds))
	if replaced != nil && !sameHolds(replaced.holds, nomination.holds) {
		pl.nominations.released(pod.UID, replaced)
	}

	// If the pod got deleted before its nomination was recorded, then
	// RemoveNominatedPod was called too early.
	if current, err := pl.podLister.Pods(pod.Namespace).Get(pod.Name); err != nil || current.UID != pod.UID {
		pl.RemoveNominatedPod(ctx, pod, nodeName)
	}
}

// RemoveNominatedPod forgets the nomination of the pod. Claims of its
// victims are tracked independently of it until they are deallocated.
//
// The devices held for the pod may be free, so pods waiting for devices get
// another chance, unless the pod got assumed with exactly those devices.
func (pl *DynamicResources) RemoveNominatedPod(ctx context.Context, pod *v1.Pod, nodeName string) {
	if pl.nominations == nil {
		return
	}
	nomination := pl.nominations.remove(pod.UID, nodeName)
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
