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
	"sync"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
)

// claimNominations is what the DynamicResources plugin remembers about
// preemptors across scheduling cycles (KEP-5690): the devices of preempted
// pods which they are waiting for. Claims of preempted pods are tracked
// separately by victimClaimTracker. DefaultDRAManager owns it, so all
// profiles share it.
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

func newClaimNominations() *claimNominations {
	return &claimNominations{
		pods: make(map[types.UID]*podNomination),
	}
}

// add stores the nomination of a pod. It returns the nomination which got
// replaced.
func (n *claimNominations) add(podUID types.UID, nomination *podNomination) *podNomination {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	replaced := n.pods[podUID]
	n.pods[podUID] = nomination
	return replaced
}

// remove drops the nomination of the pod if nodeName is empty or matches
// and returns it.
func (n *claimNominations) remove(podUID types.UID, nodeName string) *podNomination {
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
	return nomination
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
