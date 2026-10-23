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
	"maps"
	"sync"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// claimNominations is what the DynamicResources plugin remembers about
// preemption across scheduling cycles (KEP-5690): the allocations which
// AllocationExtensions report for pods, in particular the devices held for
// pods which the PodNominator nominated, and the pods which got preempted
// together with those of their claims which are not deallocated yet. Victims
// are tracked per node, independently of which pod preempted them.
// DefaultDRAManager owns it, so all profiles share it.
//
// The scheduling queue calls AllocationExtensions and queueing hints while
// it is locked, and both use it. Therefore nothing which locks the queue may
// be called while mutex is locked.
type claimNominations struct {
	mutex sync.RWMutex
	// pods maps pod UIDs to their allocations.
	pods map[types.UID]*podAllocations
	// nominations maps node names to the pods nominated to that node with
	// their confirmed allocation, which Filter looks up for each node.
	nominations map[string]map[types.UID]*allocation
	// victims maps node names to the preempted pods on that node, by pod UID.
	victims map[string]map[types.UID]types.NamespacedName
	// victimNodes maps the UIDs of preempted pods to their node.
	victimNodes map[types.UID]string
	// victimClaims maps node names to the claims of preempted pods on that
	// node, by claim UID.
	victimClaims map[string]map[types.UID]*victimClaim
	// victimClaimNodes maps the UIDs of victim claims to their node.
	victimClaimNodes map[types.UID]string
}

// podAllocations are the allocations of a pod, as described by
// framework.AllocationExtensions.
type podAllocations struct {
	// candidate is what PrepareAllocation recorded and AddAllocation did not
	// confirm yet.
	candidate *allocation
	// nominated is the confirmed allocation of the pod while the
	// PodNominator nominates it.
	nominated *allocation
	// assumed is the confirmed allocation of the pod while it is assumed.
	assumed *allocation
}

func (p *podAllocations) empty() bool {
	return p.candidate == nil && p.nominated == nil && p.assumed == nil
}

// allocation is what a pod is going to get on a node. Instances are never
// modified, only replaced.
type allocation struct {
	nodeName string
	priority int32
	// assumed is true if the allocation is for the assumed pod.
	assumed bool
	// claims are the allocations of the claims of the pod which the
	// scheduler allocates, by claim UID. For a nominated pod, these are
	// the devices which pods with lower or equal priority must not get.
	claims map[types.UID]heldClaim
	// released are the UIDs of claims of preempted pods whose devices the
	// allocation may use once they got deallocated.
	released sets.Set[types.UID]
}

// heldClaim is the allocation that a claim is expected to get.
type heldClaim struct {
	claim      types.NamespacedName
	allocation *resourceapi.AllocationResult
}

// keeps returns true if the allocation has the claim with the same devices.
func (a *allocation) keeps(uid types.UID, held heldClaim) bool {
	kept, ok := a.claims[uid]
	return ok && sameDevices(kept.allocation.Devices.Results, held.allocation.Devices.Results)
}

// victimClaim is a claim of preempted pods which gets deallocated once they
// are gone. Instances are never modified, only replaced.
type victimClaim struct {
	claim types.NamespacedName
	uid   types.UID
	// results are the released devices. Different results mean that the
	// claim got deallocated and allocated again in the meantime.
	results []resourceapi.DeviceRequestAllocationResult
	// victims are the preempted pods which use the claim. They are not
	// expected to keep using it.
	victims map[types.UID]types.NamespacedName
}

func newClaimNominations() *claimNominations {
	return &claimNominations{
		pods:             make(map[types.UID]*podAllocations),
		nominations:      make(map[string]map[types.UID]*allocation),
		victims:          make(map[string]map[types.UID]types.NamespacedName),
		victimNodes:      make(map[types.UID]string),
		victimClaims:     make(map[string]map[types.UID]*victimClaim),
		victimClaimNodes: make(map[types.UID]string),
	}
}

// setCandidate replaces the candidate allocation of the pod. A nil
// candidate drops it.
func (n *claimNominations) setCandidate(podUID types.UID, candidate *allocation) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	p := n.pods[podUID]
	if p == nil {
		if candidate == nil {
			return
		}
		p = &podAllocations{}
		n.pods[podUID] = p
	}
	p.candidate = candidate
	n.pruneLocked(podUID, p)
}

// addAllocation confirms the candidate allocation of the pod if it was
// prepared for the node and the same state, nominated or assumed. Without
// such a candidate, the allocation has nothing but the node, which is only
// recorded for pods with claims.
func (n *claimNominations) addAllocation(podUID types.UID, nodeName string, assumed bool, priority int32, hasClaims bool) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	p := n.pods[podUID]
	var a *allocation
	if p != nil && p.candidate != nil && p.candidate.nodeName == nodeName && p.candidate.assumed == assumed {
		a = p.candidate
		p.candidate = nil
	} else if hasClaims {
		a = &allocation{nodeName: nodeName, priority: priority, assumed: assumed}
	} else {
		return
	}
	if p == nil {
		p = &podAllocations{}
		n.pods[podUID] = p
	}
	if assumed {
		p.assumed = a
		return
	}
	if p.nominated != nil {
		n.deleteNominationLocked(podUID, p.nominated.nodeName)
	}
	p.nominated = a
	if n.nominations[nodeName] == nil {
		n.nominations[nodeName] = make(map[types.UID]*allocation)
	}
	n.nominations[nodeName][podUID] = a
}

func (n *claimNominations) deleteNominationLocked(podUID types.UID, nodeName string) {
	delete(n.nominations[nodeName], podUID)
	if len(n.nominations[nodeName]) == 0 {
		delete(n.nominations, nodeName)
	}
}

// removeAllocation drops the confirmed allocation of the nominated or
// assumed pod on the node. It returns the dropped allocation, nil if there
// was none, and for a nomination the allocation which may keep its devices:
// the one of the pod assumed on the same node, or else a candidate for a new
// nomination to the same node.
func (n *claimNominations) removeAllocation(podUID types.UID, nodeName string, assumed bool) (removed, keeper *allocation) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	p := n.pods[podUID]
	if p == nil {
		return nil, nil
	}
	defer n.pruneLocked(podUID, p)
	if assumed {
		if p.assumed == nil || p.assumed.nodeName != nodeName {
			return nil, nil
		}
		removed, p.assumed = p.assumed, nil
		return removed, nil
	}
	if p.nominated == nil || p.nominated.nodeName != nodeName {
		return nil, nil
	}
	removed, p.nominated = p.nominated, nil
	n.deleteNominationLocked(podUID, nodeName)
	switch {
	case p.assumed != nil && p.assumed.nodeName == nodeName:
		keeper = p.assumed
	case p.candidate != nil && !p.candidate.assumed && p.candidate.nodeName == nodeName:
		keeper = p.candidate
	}
	return removed, keeper
}

func (n *claimNominations) pruneLocked(podUID types.UID, p *podAllocations) {
	if p.empty() {
		delete(n.pods, podUID)
	}
}

// dropCandidate drops the candidate allocation of the pod, for example
// because the pod got deleted.
func (n *claimNominations) dropCandidate(podUID types.UID) {
	n.setCandidate(podUID, nil)
}

// nominatedNode returns the node which the pod is nominated to, "" if none.
func (n *claimNominations) nominatedNode(podUID types.UID) string {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	if p := n.pods[podUID]; p != nil && p.nominated != nil {
		return p.nominated.nodeName
	}
	return ""
}

// nominatedAllocation returns the confirmed allocation of the nominated pod,
// nil if there is none.
func (n *claimNominations) nominatedAllocation(podUID types.UID) *allocation {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	if p := n.pods[podUID]; p != nil {
		return p.nominated
	}
	return nil
}

// nominationsOnNode returns the confirmed allocations of the pods nominated
// to the node, by pod UID.
func (n *claimNominations) nominationsOnNode(nodeName string) map[types.UID]*allocation {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return maps.Clone(n.nominations[nodeName])
}

// assumeVictim stores a preempted pod on its node and the claims which it
// releases. A claim which is tracked already gets the victim added. Pods
// for which gone returns true stop being tracked as victims of the node.
func (n *claimNominations) assumeVictim(nodeName string, uid types.UID, name types.NamespacedName, claims []*victimClaim, gone func(uid types.UID, name types.NamespacedName) bool) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	for victimUID, victimName := range n.victims[nodeName] {
		if victimUID != uid && gone(victimUID, victimName) {
			delete(n.victims[nodeName], victimUID)
			delete(n.victimNodes, victimUID)
		}
	}
	if n.victims[nodeName] == nil {
		n.victims[nodeName] = make(map[types.UID]types.NamespacedName)
	}
	n.victims[nodeName][uid] = name
	n.victimNodes[uid] = nodeName
	for _, claim := range claims {
		if existing := n.victimClaims[nodeName][claim.uid]; existing != nil {
			// Also released by other victims.
			merged := *claim
			merged.victims = maps.Clone(existing.victims)
			maps.Copy(merged.victims, claim.victims)
			claim = &merged
		}
		n.setVictimClaimLocked(nodeName, claim)
	}
}

func (n *claimNominations) setVictimClaimLocked(nodeName string, claim *victimClaim) {
	if oldNodeName, ok := n.victimClaimNodes[claim.uid]; ok && oldNodeName != nodeName {
		n.deleteVictimClaimLocked(oldNodeName, claim.uid)
	}
	if n.victimClaims[nodeName] == nil {
		n.victimClaims[nodeName] = make(map[types.UID]*victimClaim)
	}
	n.victimClaims[nodeName][claim.uid] = claim
	n.victimClaimNodes[claim.uid] = nodeName
}

func (n *claimNominations) deleteVictimClaimLocked(nodeName string, uid types.UID) {
	delete(n.victimClaims[nodeName], uid)
	if len(n.victimClaims[nodeName]) == 0 {
		delete(n.victimClaims, nodeName)
	}
	delete(n.victimClaimNodes, uid)
}

// forgetVictim stops tracking a preempted pod, for example because it got
// preempted in memory while it waited in Permit and then got scheduled
// again. Claims which it uses don't count as released by it anymore.
func (n *claimNominations) forgetVictim(uid types.UID) {
	// Most calls are for pods which did not get preempted, which don't need
	// to block readers.
	n.mutex.RLock()
	_, ok := n.victimNodes[uid]
	n.mutex.RUnlock()
	if !ok {
		return
	}
	n.mutex.Lock()
	defer n.mutex.Unlock()
	nodeName, ok := n.victimNodes[uid]
	if !ok {
		return
	}
	delete(n.victims[nodeName], uid)
	if len(n.victims[nodeName]) == 0 {
		delete(n.victims, nodeName)
	}
	delete(n.victimNodes, uid)
	for claimNodeName, claims := range n.victimClaims {
		for claimUID, claim := range claims {
			if _, ok := claim.victims[uid]; !ok {
				continue
			}
			updated := *claim
			updated.victims = maps.Clone(claim.victims)
			delete(updated.victims, uid)
			n.victimClaims[claimNodeName][claimUID] = &updated
		}
	}
}

// victimsOnNode returns the pods which got preempted on the node.
func (n *claimNominations) victimsOnNode(nodeName string) map[types.UID]types.NamespacedName {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return maps.Clone(n.victims[nodeName])
}

// isVictim returns true if the pod got preempted.
func (n *claimNominations) isVictim(uid types.UID) bool {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	_, ok := n.victimNodes[uid]
	return ok
}

// victimClaimsOnNode returns the claims of preempted pods on the node.
func (n *claimNominations) victimClaimsOnNode(nodeName string) []*victimClaim {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	claims := make([]*victimClaim, 0, len(n.victimClaims[nodeName]))
	for _, claim := range n.victimClaims[nodeName] {
		claims = append(claims, claim)
	}
	return claims
}

func (n *claimNominations) isVictimClaim(uid types.UID) bool {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	_, ok := n.victimClaimNodes[uid]
	return ok
}

// forget stops tracking a victim claim. If claim is not nil, then only if
// the tracked instance is still the same, because it may have been replaced
// after the caller looked at it.
func (n *claimNominations) forget(uid types.UID, claim *victimClaim) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	nodeName, ok := n.victimClaimNodes[uid]
	if !ok || claim != nil && n.victimClaims[nodeName][uid] != claim {
		return
	}
	n.deleteVictimClaimLocked(nodeName, uid)
}

// handlers forget victim claims when they get deallocated or deleted.
func (n *claimNominations) handlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj any) {
			oldClaim, newClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
			if err != nil {
				logger.Error(err, "unexpected object in claimNominations.UpdateFunc")
				return
			}
			if oldClaim.Status.Allocation != nil && newClaim.Status.Allocation == nil {
				n.forget(newClaim.UID, nil)
			}
		},
		DeleteFunc: func(obj any) {
			claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in claimNominations.DeleteFunc")
				return
			}
			n.forget(claim.UID, nil)
		},
	}
}

// podHandlers drop the candidate allocations of deleted pods, which
// AddAllocation is not going to confirm anymore.
func (n *claimNominations) podHandlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		DeleteFunc: func(obj any) {
			pod, _, err := schedutil.As[*v1.Pod](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in claimNominations pod DeleteFunc")
				return
			}
			n.dropCandidate(pod.UID)
		},
	}
}
