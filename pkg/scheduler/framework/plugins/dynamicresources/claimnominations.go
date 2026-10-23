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
// preemption across scheduling cycles (KEP-5690): the preemptors which are
// waiting for devices of preempted pods, and the claims of preempted pods
// which are not deallocated yet. DefaultDRAManager owns it, so all profiles
// share it.
//
// Queueing hints read it while the scheduling queue is locked. Therefore
// nothing which locks the queue may be called while mutex is locked.
type claimNominations struct {
	// releaseHandler gets called without holding mutex when held devices
	// become available again. Set once before scheduling starts.
	releaseHandler func(claim *resourceapi.ResourceClaim, holderUID types.UID, priority int32)

	mutex sync.RWMutex
	// pods maps the UIDs of nominated preemptors to their nomination.
	pods map[types.UID]*podNomination
	// victimClaims maps node names to the claims of preempted pods on that
	// node, by claim UID.
	victimClaims map[string]map[types.UID]*victimClaim
	// victimClaimNodes maps the UIDs of victim claims to their node.
	victimClaimNodes map[types.UID]string
}

// podNomination is the nomination of a preemptor.
type podNomination struct {
	nodeName string
	priority int32
	// victims are the pods which got preempted for the nomination.
	victims map[types.UID]types.NamespacedName
	// holds are the devices which other pods with lower or equal priority
	// must not get, by UID of the claim of the preemptor.
	holds map[types.UID]heldClaim
}

// heldClaim is the allocation that a claim of a preemptor is expected to
// get on the nominated node.
type heldClaim struct {
	claim      types.NamespacedName
	allocation *resourceapi.AllocationResult
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
	// expected to keep using it, except after a failed preemption, so
	// pods of failed preemptions are not listed.
	victims map[types.UID]types.NamespacedName
}

func newClaimNominations() *claimNominations {
	return &claimNominations{
		pods:             make(map[types.UID]*podNomination),
		victimClaims:     make(map[string]map[types.UID]*victimClaim),
		victimClaimNodes: make(map[types.UID]string),
	}
}

// add stores the nomination of a pod and the claims which its victims
// release on each node. It returns the nomination which got replaced.
func (n *claimNominations) add(podUID types.UID, nomination *podNomination, claims map[string][]*victimClaim) *podNomination {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	replaced := n.pods[podUID]
	n.pods[podUID] = nomination
	for nodeName, nodeClaims := range claims {
		for _, claim := range nodeClaims {
			if existing := n.victimClaims[nodeName][claim.uid]; existing != nil {
				// Also released by an earlier preemption.
				merged := *claim
				merged.victims = maps.Clone(existing.victims)
				maps.Copy(merged.victims, claim.victims)
				claim = &merged
			}
			n.setVictimClaimLocked(nodeName, claim)
		}
	}
	return replaced
}

// remove drops the nomination of the pod if nodeName is empty or matches
// and returns it. After a failed preemption, its victims may keep running,
// so they stop counting as victims which are about to release their claims.
func (n *claimNominations) remove(podUID types.UID, nodeName string, preemptionFailed bool) *podNomination {
	// Most calls are for pods without a nomination, for example for each
	// assumed pod, which don't need to block readers.
	n.mutex.RLock()
	_, ok := n.pods[podUID]
	n.mutex.RUnlock()
	if !ok {
		return nil
	}
	n.mutex.Lock()
	defer n.mutex.Unlock()
	nomination := n.pods[podUID]
	if nomination == nil || nodeName != "" && nomination.nodeName != nodeName {
		return nil
	}
	delete(n.pods, podUID)
	if !preemptionFailed {
		return nomination
	}
	for victimNodeName, nodeClaims := range n.victimClaims {
		for _, claim := range nodeClaims {
			var victims map[types.UID]types.NamespacedName
			for uid := range nomination.victims {
				if _, ok := claim.victims[uid]; ok {
					if victims == nil {
						victims = maps.Clone(claim.victims)
					}
					delete(victims, uid)
				}
			}
			if victims != nil {
				updated := *claim
				updated.victims = victims
				n.setVictimClaimLocked(victimNodeName, &updated)
			}
		}
	}
	return nomination
}

func (n *claimNominations) setVictimClaimLocked(nodeName string, claim *victimClaim) {
	if n.victimClaims[nodeName] == nil {
		n.victimClaims[nodeName] = make(map[types.UID]*victimClaim)
	}
	n.victimClaims[nodeName][claim.uid] = claim
	n.victimClaimNodes[claim.uid] = nodeName
}

// nomination returns the nomination of the pod, nil if there is none.
func (n *claimNominations) nomination(podUID types.UID) *podNomination {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return n.pods[podUID]
}

// nominationsOnNode returns the nominations for the node, by pod UID.
func (n *claimNominations) nominationsOnNode(nodeName string) map[types.UID]*podNomination {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	nominations := make(map[types.UID]*podNomination)
	for uid, nomination := range n.pods {
		if nomination.nodeName == nodeName {
			nominations[uid] = nomination
		}
	}
	return nominations
}

// released reports that the devices held by the nomination of the pod may
// be available again, as if one of the claims got deallocated.
func (n *claimNominations) released(holderUID types.UID, nomination *podNomination) {
	if n.releaseHandler == nil {
		return
	}
	for uid, held := range nomination.holds {
		// One event is enough, the queueing hints don't check which
		// claim got deallocated.
		claim := &resourceapi.ResourceClaim{}
		claim.Namespace, claim.Name, claim.UID = held.claim.Namespace, held.claim.Name, uid
		claim.Status.Allocation = held.allocation
		n.releaseHandler(claim, holderUID, nomination.priority)
		return
	}
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
	delete(n.victimClaims[nodeName], uid)
	if len(n.victimClaims[nodeName]) == 0 {
		delete(n.victimClaims, nodeName)
	}
	delete(n.victimClaimNodes, uid)
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
