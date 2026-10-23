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

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// claimNominations is what the DynamicResources plugin remembers about
// preemption across scheduling cycles (KEP-5690): the nodes which the
// PodNominator nominated pods to, and the pods which got preempted together
// with those of their claims which are not deallocated yet. Victims are
// tracked per node, independently of which pod preempted them.
// DefaultDRAManager owns it, so all profiles share it.
//
// The scheduling queue calls AllocationExtensions and queueing hints while
// it is locked, and both use it. Therefore nothing which locks the queue may
// be called while mutex is locked.
type claimNominations struct {
	mutex sync.RWMutex
	// nominatedNodes maps the UIDs of nominated pods with claims to the node
	// which the PodNominator nominated them to.
	nominatedNodes map[types.UID]string
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
		nominatedNodes:   make(map[types.UID]string),
		victims:          make(map[string]map[types.UID]types.NamespacedName),
		victimNodes:      make(map[types.UID]string),
		victimClaims:     make(map[string]map[types.UID]*victimClaim),
		victimClaimNodes: make(map[types.UID]string),
	}
}

// nominate records that the PodNominator nominated the pod to the node.
func (n *claimNominations) nominate(podUID types.UID, nodeName string) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	n.nominatedNodes[podUID] = nodeName
}

// removeNomination drops the nomination of the pod if it is for the node.
func (n *claimNominations) removeNomination(podUID types.UID, nodeName string) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	if n.nominatedNodes[podUID] == nodeName {
		delete(n.nominatedNodes, podUID)
	}
}

// nominatedNode returns the node which the pod is nominated to, "" if none.
func (n *claimNominations) nominatedNode(podUID types.UID) string {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return n.nominatedNodes[podUID]
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
