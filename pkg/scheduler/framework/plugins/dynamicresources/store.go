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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// preemptionStore tracks cross-cycle DRA preemption state (KEP-5690): device
// holds for nominated preemptors and victim claims awaiting deallocation.
// Owned by DefaultDRAManager and shared across scheduler profiles when
// DRAPreemption is enabled.
//
// Lock ordering: the scheduling queue invokes AddNomination, RemoveNomination,
// and queueing hints while holding its own lock, so callers holding mutex must
// never call into the scheduling queue or make API calls.
type preemptionStore struct {
	mutex sync.RWMutex
	// holds maps nominated pod UIDs to their device holds on the nominated node.
	holds map[types.UID]*allocation
	// victimClaims tracks claims of preempted pods awaiting deallocation,
	// indexed by node name and claim UID. Entries are replaced, never mutated.
	victimClaims map[string]map[types.UID]*releasedClaim
	// victimClaimNodes is the reverse index from claim UID to node name in
	// victimClaims.
	victimClaimNodes map[types.UID]string
}

// allocation records the devices selected for a pod on nodeName and held while
// the pod remains nominated there. Instances are immutable and replaced atomically.
type allocation struct {
	nodeName string
	// claims maps UIDs of unallocated user claims to the allocations chosen by Filter.
	claims map[types.UID]heldClaim
	// released maps UIDs of victim claims freed during the preemption dry run for
	// this pod. While those claims are still settling, nodeHolds uses them to
	// avoid double-counting shared capacity and Filter rejects other pods from
	// reserving them.
	released map[types.UID]*releasedClaim
}

type heldClaim struct {
	claim      types.NamespacedName
	allocation *resourceapi.AllocationResult
}

// releasedClaim describes a claim whose devices become free once the removed
// or preempted pods using it are gone. Instances are immutable because
// CycleState clones, holds, and preemptionStore share them; updates replace
// the map entry instead of mutating in place.
type releasedClaim struct {
	claim    types.NamespacedName
	uid      types.UID
	nodeName string
	// allocation is the claim's committed or in-flight AllocationResult.
	// Callers must only read allocation.Devices: an in-flight AllocationResult
	// is shared with the claim tracker, where PreBind concurrently mutates
	// AllocationTimestamp.
	allocation *resourceapi.AllocationResult
	// pods are the removed or preempted pods that use this claim. In a dry-run
	// simulation, re-adding any of them via AddPod restores the claim.
	pods map[types.UID]types.NamespacedName
}

func newPreemptionStore() *preemptionStore {
	return &preemptionStore{
		holds:            make(map[types.UID]*allocation),
		victimClaims:     make(map[string]map[types.UID]*releasedClaim),
		victimClaimNodes: make(map[types.UID]string),
	}
}

// nominate sets podUID's hold and returns any previous hold it replaced.
func (s *preemptionStore) nominate(podUID types.UID, hold *allocation) *allocation {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	replaced := s.holds[podUID]
	s.holds[podUID] = hold
	return replaced
}

// release deletes and returns podUID's hold if it is on nodeName, or nil otherwise.
func (s *preemptionStore) release(podUID types.UID, nodeName string) *allocation {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	hold := s.holds[podUID]
	if hold == nil || hold.nodeName != nodeName {
		return nil
	}
	delete(s.holds, podUID)
	return hold
}

func (s *preemptionStore) nominatedAllocation(podUID types.UID) *allocation {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	return s.holds[podUID]
}

// assumeVictimClaim records a victim's claim as awaiting deallocation. If an
// entry already exists for the same node and device allocation, the victim
// pods are merged (multiple victims can share a claim); otherwise the stale
// entry from an earlier allocation is replaced.
func (s *preemptionStore) assumeVictimClaim(claim *releasedClaim) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	if nodeName, ok := s.victimClaimNodes[claim.uid]; ok {
		existing := s.victimClaims[nodeName][claim.uid]
		if nodeName == claim.nodeName && apiequality.Semantic.DeepEqual(existing.allocation.Devices.Results, claim.allocation.Devices.Results) {
			merged := *claim
			merged.pods = maps.Clone(existing.pods)
			maps.Copy(merged.pods, claim.pods)
			claim = &merged
		}
		s.forgetVictimClaimLocked(claim.uid)
	}
	if s.victimClaims[claim.nodeName] == nil {
		s.victimClaims[claim.nodeName] = make(map[types.UID]*releasedClaim)
	}
	s.victimClaims[claim.nodeName][claim.uid] = claim
	s.victimClaimNodes[claim.uid] = claim.nodeName
}

func (s *preemptionStore) victimClaimsOnNode(nodeName string) []*releasedClaim {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	claims := make([]*releasedClaim, 0, len(s.victimClaims[nodeName]))
	for _, claim := range s.victimClaims[nodeName] {
		claims = append(claims, claim)
	}
	return claims
}

func (s *preemptionStore) isVictimClaim(uid types.UID) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()
	_, ok := s.victimClaimNodes[uid]
	return ok
}

// forgetVictimClaim removes uid from victimClaims. When claim is non-nil, it
// removes the entry only if the stored pointer still matches, avoiding a race
// when assumeVictimClaim replaced the entry after the caller read it.
func (s *preemptionStore) forgetVictimClaim(uid types.UID, claim *releasedClaim) {
	s.mutex.Lock()
	defer s.mutex.Unlock()
	nodeName, ok := s.victimClaimNodes[uid]
	if !ok || claim != nil && s.victimClaims[nodeName][uid] != claim {
		return
	}
	s.forgetVictimClaimLocked(uid)
}

func (s *preemptionStore) forgetVictimClaimLocked(uid types.UID) {
	nodeName := s.victimClaimNodes[uid]
	delete(s.victimClaims[nodeName], uid)
	if len(s.victimClaims[nodeName]) == 0 {
		delete(s.victimClaims, nodeName)
	}
	delete(s.victimClaimNodes, uid)
}

// claimHandlers returns cache event handlers that forget tracked victim claims
// once they are deallocated or deleted.
func (s *preemptionStore) claimHandlers(logger klog.Logger) cache.ResourceEventHandler {
	return cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(oldObj, newObj any) {
			oldClaim, newClaim, err := schedutil.As[*resourceapi.ResourceClaim](oldObj, newObj)
			if err != nil {
				logger.Error(err, "unexpected object in preemptionStore claim UpdateFunc")
				return
			}
			if oldClaim.Status.Allocation != nil && newClaim.Status.Allocation == nil {
				s.forgetVictimClaim(newClaim.UID, nil)
			}
		},
		DeleteFunc: func(obj any) {
			claim, _, err := schedutil.As[*resourceapi.ResourceClaim](obj, nil)
			if err != nil {
				logger.Error(err, "unexpected object in preemptionStore claim DeleteFunc")
				return
			}
			s.forgetVictimClaim(claim.UID, nil)
		},
	}
}
