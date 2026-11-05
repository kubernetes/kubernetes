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
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// preemptionTracker tracks cross-cycle DRA preemption state (KEP-5690), owned by
// DefaultDRAManager and shared across scheduler profiles when DRAPreemption is
// enabled.
//
// Lock ordering: queueing hints acquire mutex while holding the scheduling
// queue lock, so callers holding mutex must never call into the scheduling
// queue.
type preemptionTracker struct {
	mutex sync.RWMutex
	// victimClaims tracks claims of preempted pods awaiting deallocation,
	// indexed by node name and claim UID. Entries are replaced, never mutated.
	victimClaims map[string]map[types.UID]*releasedClaim
	// victimClaimNodes is the reverse index from claim UID to node name in
	// victimClaims.
	victimClaimNodes map[types.UID]string
}

// releasedClaim describes a claim whose devices become free once its removed
// or preempted owning pod is gone. Instances are immutable because CycleState
// clones and preemptionTracker callers share them.
type releasedClaim struct {
	claim    types.NamespacedName
	uid      types.UID
	nodeName string
	// allocatedDevices are from the claim's committed or in-flight
	// allocation. Only Devices is kept because PreBind mutates other fields of
	// a shared in-flight AllocationResult.
	allocatedDevices *resourceapi.DeviceAllocationResult
	// podUID is the UID of the owning pod.
	podUID types.UID
}

func newPreemptionTracker() *preemptionTracker {
	return &preemptionTracker{
		victimClaims:     make(map[string]map[types.UID]*releasedClaim),
		victimClaimNodes: make(map[types.UID]string),
	}
}

func (s *preemptionTracker) assumeVictimClaim(claim *releasedClaim) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if nodeName, ok := s.victimClaimNodes[claim.uid]; ok {
		s.forgetVictimClaimLocked(claim.uid, nodeName)
	}
	if s.victimClaims[claim.nodeName] == nil {
		s.victimClaims[claim.nodeName] = make(map[types.UID]*releasedClaim)
	}
	s.victimClaims[claim.nodeName][claim.uid] = claim
	s.victimClaimNodes[claim.uid] = claim.nodeName
}

func (s *preemptionTracker) victimClaimsOnNode(nodeName string) []*releasedClaim {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	claims := make([]*releasedClaim, 0, len(s.victimClaims[nodeName]))
	for _, claim := range s.victimClaims[nodeName] {
		claims = append(claims, claim)
	}
	return claims
}

func (s *preemptionTracker) victimClaim(uid types.UID) *releasedClaim {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	nodeName, ok := s.victimClaimNodes[uid]
	if !ok {
		return nil
	}
	return s.victimClaims[nodeName][uid]
}

// forgetVictimClaim removes uid from victimClaims. When claim is non-nil, it
// removes the entry only if the stored pointer still matches, avoiding a race
// when assumeVictimClaim replaced the entry after the caller read it.
func (s *preemptionTracker) forgetVictimClaim(uid types.UID, claim *releasedClaim) {
	// Claim events and Reserve call this for every claim, most of which are
	// untracked. Check under the read lock first to avoid taking the write lock
	// for untracked claims.
	s.mutex.RLock()
	_, tracked := s.victimClaimNodes[uid]
	s.mutex.RUnlock()
	if !tracked {
		return
	}
	s.mutex.Lock()
	defer s.mutex.Unlock()
	nodeName, ok := s.victimClaimNodes[uid]
	if !ok || claim != nil && s.victimClaims[nodeName][uid] != claim {
		return
	}
	s.forgetVictimClaimLocked(uid, nodeName)
}

func (s *preemptionTracker) forgetVictimClaimLocked(uid types.UID, nodeName string) {
	delete(s.victimClaims[nodeName], uid)
	if len(s.victimClaims[nodeName]) == 0 {
		delete(s.victimClaims, nodeName)
	}
	delete(s.victimClaimNodes, uid)
}

// claimHandlers returns cache event handlers that forget tracked victim claims
// once they are deallocated or deleted.
func (s *preemptionTracker) claimHandlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj any) {
			oldClaim, newClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
			if err != nil {
				logger.Error(err, "unexpected object in preemptionTracker claim UpdateFunc")
				return
			}
			if oldClaim.Status.Allocation != nil && newClaim.Status.Allocation == nil {
				s.forgetVictimClaim(newClaim.UID, nil)
			}
		},
		DeleteFunc: func(obj any) {
			claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in preemptionTracker claim DeleteFunc")
				return
			}
			s.forgetVictimClaim(claim.UID, nil)
		},
	}
}
