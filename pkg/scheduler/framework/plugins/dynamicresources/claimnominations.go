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

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// claimNominations is what the DynamicResources plugin remembers about the
// devices which pods are going to get before that got persisted (KEP-5690):
// those of preempted pods which nominated preemptors are waiting for. Claims
// of preempted pods are tracked separately by victimClaimTracker.
// DefaultDRAManager owns it, so all profiles share it.
//
// AllocationExtensions and queueing hints use it while the scheduling queue
// is locked. Therefore nothing which locks the queue may be called while
// mutex is locked.
type claimNominations struct {
	mutex sync.RWMutex
	// candidates map pod UIDs to what the last scheduling cycle of the pod
	// selected for it, until AddAllocation confirms it.
	candidates map[types.UID]*podNomination
	// pods maps the UIDs of nominated preemptors to their confirmed
	// nomination.
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
	return &claimNominations{
		candidates: make(map[types.UID]*podNomination),
		pods:       make(map[types.UID]*podNomination),
	}
}

// prepare replaces the candidate of the pod. A nil candidate removes it.
func (n *claimNominations) prepare(podUID types.UID, candidate *podNomination) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	if candidate == nil {
		delete(n.candidates, podUID)
		return
	}
	n.candidates[podUID] = candidate
}

// confirm turns the candidate of the pod into its nomination, if it is for
// the node. It returns the confirmed candidate.
func (n *claimNominations) confirm(podUID types.UID, nodeName string) *podNomination {
	// Most calls are for pods without a candidate, which don't need to
	// block readers.
	n.mutex.RLock()
	_, ok := n.candidates[podUID]
	n.mutex.RUnlock()
	if !ok {
		return nil
	}
	n.mutex.Lock()
	defer n.mutex.Unlock()
	candidate := n.candidates[podUID]
	if candidate == nil || candidate.nodeName != nodeName {
		return nil
	}
	delete(n.candidates, podUID)
	n.pods[podUID] = candidate
	return candidate
}

// removeNomination drops the nomination of the pod for the node and returns
// it, together with a newer candidate for the same node, if any. Without a
// newer candidate, the nomination becomes the candidate again, so that
// nominating the pod to the node again restores it.
func (n *claimNominations) removeNomination(podUID types.UID, nodeName string) (removed, newer *podNomination) {
	n.mutex.RLock()
	_, ok := n.pods[podUID]
	n.mutex.RUnlock()
	if !ok {
		return nil, nil
	}
	n.mutex.Lock()
	defer n.mutex.Unlock()
	removed = n.pods[podUID]
	if removed == nil || removed.nodeName != nodeName {
		return nil, nil
	}
	delete(n.pods, podUID)
	candidate := n.candidates[podUID]
	switch {
	case candidate == nil:
		n.candidates[podUID] = removed
	case candidate.nodeName == nodeName:
		newer = candidate
	}
	return removed, newer
}

// nomination returns the confirmed nomination of the pod, nil if there is
// none.
func (n *claimNominations) nomination(podUID types.UID) *podNomination {
	n.mutex.RLock()
	defer n.mutex.RUnlock()
	return n.pods[podUID]
}

// nominationsOnNode returns the confirmed nominations for the node, by pod
// UID.
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

func (n *claimNominations) forgetCandidate(podUID types.UID) {
	n.mutex.Lock()
	defer n.mutex.Unlock()
	delete(n.candidates, podUID)
}

// handlers drop the candidates of pods which got deleted or bound, because
// they won't get nominated or assumed anymore.
func (n *claimNominations) handlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj any) {
			_, pod, err := schedutil.As[*v1.Pod](oldObj, newObj)
			if err != nil {
				logger.Error(err, "unexpected object in claimNominations.UpdateFunc")
				return
			}
			if pod.Spec.NodeName != "" {
				n.forgetCandidate(pod.UID)
			}
		},
		DeleteFunc: func(obj any) {
			pod, _, err := schedutil.As[*v1.Pod](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in claimNominations.DeleteFunc")
				return
			}
			n.forgetCandidate(pod.UID)
		},
	}
}
