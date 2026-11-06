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

// victimClaimTracker remembers the claims of preempted pods which are not
// deallocated yet, by node (KEP-5690). It does not matter which pod
// preempted them: all pods nominated to the node wait for them.
// DefaultDRAManager owns it, so all profiles share it.
//
// Queueing hints read it while the scheduling queue is locked. Therefore
// nothing which locks the queue may be called while mutex is locked.
type victimClaimTracker struct {
	mutex sync.RWMutex
	// claims maps node names to the claims of preempted pods on that
	// node, by claim UID.
	claims map[string]map[types.UID]*victimClaim
	// nodes maps the UIDs of victim claims to their node.
	nodes map[types.UID]string
}

// victimClaim is a claim of preempted pods which gets deallocated once they
// are gone. Instances are never modified, only replaced.
type victimClaim struct {
	claim types.NamespacedName
	uid   types.UID
	// results are the released devices. Different results mean that the
	// claim got deallocated and allocated again in the meantime.
	results []resourceapi.DeviceRequestAllocationResult
	// victims are the preempted pods which use the claim. They only get
	// added once their preemption is under way, so they are not expected
	// to keep using it.
	victims map[types.UID]types.NamespacedName
}

func newVictimClaimTracker() *victimClaimTracker {
	return &victimClaimTracker{
		claims: make(map[string]map[types.UID]*victimClaim),
		nodes:  make(map[types.UID]string),
	}
}

// add starts tracking a claim of preempted pods on the node. The victims of
// a claim which is already tracked get merged.
func (t *victimClaimTracker) add(nodeName string, claim *victimClaim) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	if existing := t.claims[nodeName][claim.uid]; existing != nil {
		merged := *claim
		merged.victims = maps.Clone(existing.victims)
		maps.Copy(merged.victims, claim.victims)
		claim = &merged
	}
	if t.claims[nodeName] == nil {
		t.claims[nodeName] = make(map[types.UID]*victimClaim)
	}
	t.claims[nodeName][claim.uid] = claim
	t.nodes[claim.uid] = nodeName
}

// onNode returns the claims of preempted pods on the node.
func (t *victimClaimTracker) onNode(nodeName string) []*victimClaim {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	claims := make([]*victimClaim, 0, len(t.claims[nodeName]))
	for _, claim := range t.claims[nodeName] {
		claims = append(claims, claim)
	}
	return claims
}

func (t *victimClaimTracker) isTracked(uid types.UID) bool {
	t.mutex.RLock()
	defer t.mutex.RUnlock()
	_, ok := t.nodes[uid]
	return ok
}

// forget stops tracking a victim claim. If claim is not nil, then only if
// the tracked instance is still the same, because it may have been replaced
// after the caller looked at it.
func (t *victimClaimTracker) forget(uid types.UID, claim *victimClaim) {
	t.mutex.Lock()
	defer t.mutex.Unlock()
	nodeName, ok := t.nodes[uid]
	if !ok || claim != nil && t.claims[nodeName][uid] != claim {
		return
	}
	delete(t.claims[nodeName], uid)
	if len(t.claims[nodeName]) == 0 {
		delete(t.claims, nodeName)
	}
	delete(t.nodes, uid)
}

// handlers forget victim claims when they get deallocated or deleted.
func (t *victimClaimTracker) handlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj any) {
			oldClaim, newClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
			if err != nil {
				logger.Error(err, "unexpected object in victimClaimTracker.UpdateFunc")
				return
			}
			if oldClaim.Status.Allocation != nil && newClaim.Status.Allocation == nil {
				t.forget(newClaim.UID, nil)
			}
		},
		DeleteFunc: func(obj any) {
			claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in victimClaimTracker.DeleteFunc")
				return
			}
			t.forget(claim.UID, nil)
		},
	}
}
