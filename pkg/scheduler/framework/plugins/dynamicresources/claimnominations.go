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

// claimNominations is what the DynamicResources plugin remembers about the
// devices which pods are going to get before that got persisted (KEP-5690):
// those of preempted pods which nominated preemptors are waiting for. Claims
// of preempted pods are tracked separately by victimClaimTracker.
// DefaultDRAManager owns it, so all profiles share it.
//
// NominationExtensions and queueing hints use it while the scheduling queue
// is locked. Therefore nothing which locks the queue may be called while
// mutex is locked.
type claimNominations struct {
	mutex sync.RWMutex
	// pods maps the UIDs of nominated preemptors to their nomination.
	pods map[types.UID]*podNomination
}

// podNomination is what a nominated preemptor is going to get on a node.
type podNomination struct {
	nodeName string
	priority int32
	// victims are the pods with claims which got preempted for the
	// nomination.
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
	return &claimNominations{pods: make(map[types.UID]*podNomination)}
}

// add records the nomination of the pod and returns the one which it
// replaces, if any.
func (n *claimNominations) add(podUID types.UID, nomination *podNomination) (replaced *podNomination) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	replaced = n.pods[podUID]
	n.pods[podUID] = nomination
	return replaced
}

// remove drops the nomination of the pod for the node and returns it.
func (n *claimNominations) remove(podUID types.UID, nodeName string) *podNomination {
	// Most calls are for pods without a nomination, which don't need to
	// block readers.
	n.mutex.RLock()
	_, ok := n.pods[podUID]
	n.mutex.RUnlock()
	if !ok {
		return nil
	}
	n.mutex.Lock()
	defer n.mutex.Unlock()
	nomination := n.pods[podUID]
	if nomination == nil || nomination.nodeName != nodeName {
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
