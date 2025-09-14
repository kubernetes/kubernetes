/*
Copyright 2025 The Kubernetes Authors.

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
	"fmt"
	"iter"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
)

// claimStore manages all ResourceClaims for the pod, whether they were
// created by the user directly (ResourceClaim) or indirectly (ResourceClaimTemplate)
// as well as the special ResourceClaim for extended resources.
//
// The zero value is usable and empty.
type claimStore struct {
	// claims contains all user-owned claims, optionally followed by the special ResourceClaim for extended resources.
	claims []*resourceapi.ResourceClaim
	// numUserOwned is the number of claims without the special ResourceClaim.
	numUserOwned int
	// initialExtendedResourceClaimUID is the initial extended resource claim UID from PreFilter phase.
	// It is different from the UID set by API server when the in-memory, temproary claim is written to the API server in PreBind phase.
	initialExtendedResourceClaimUID types.UID
}

// newClaimStore stores the list of user-owned claims and the optional claim owned by the scheduler.
func newClaimStore(claims []*resourceapi.ResourceClaim, extendedResourceClaim *resourceapi.ResourceClaim) claimStore {
	cs := claimStore{
		claims:                          claims,
		numUserOwned:                    len(claims),
		initialExtendedResourceClaimUID: "",
	}
	if extendedResourceClaim != nil {
		cs.claims = append(cs.claims, extendedResourceClaim)
		cs.initialExtendedResourceClaimUID = extendedResourceClaim.UID
	}
	return cs
}

// empty returns true when there are no ResourceClaims.
func (cs *claimStore) empty() bool {
	return len(cs.claims) == 0
}

// len returns number of all claims, whether they are owned by the user or the scheduler.
func (cs *claimStore) len() int {
	return len(cs.claims)
}

// all returns an iterator for all claims, whether they are owned by the user
// or the scheduler. If there is a special ResourceClaim for extended resoures,
// then it comes last.
func (cs *claimStore) all() iter.Seq2[int, *resourceapi.ResourceClaim] {
	return func(yield func(int, *resourceapi.ResourceClaim) bool) {
		for i, claim := range cs.claims {
			if !yield(i, claim) {
				return
			}
		}
	}
}

// allUserClaims returns an iterator which excludes the special ResourceClaim for extended resoures.
func (cs *claimStore) allUserClaims() iter.Seq2[int, *resourceapi.ResourceClaim] {
	return func(yield func(int, *resourceapi.ResourceClaim) bool) {
		for i, claim := range cs.claims[0:cs.numUserOwned] {
			if !yield(i, claim) {
				return
			}
		}
	}
}

// toAllocate returns an iterator for all claims which have no allocation result.
// the index returned is the original index in the underlying claim store, it
// may not be sequentially numbered (e.g. 0, 1, 2 ...).
func (cs *claimStore) toAllocate() iter.Seq2[int, *resourceapi.ResourceClaim] {
	return func(yield func(int, *resourceapi.ResourceClaim) bool) {
		for i, claim := range cs.claims {
			if claim.Status.Allocation != nil {
				continue
			}
			if !yield(i, claim) {
				return
			}
		}
	}
}

// extendedResourceClaim returns the special ResourceClaim if there is one, otherwise nil.
// The ResourceClaim is read-only and must be cloned before modifying it.
func (cs *claimStore) extendedResourceClaim() *resourceapi.ResourceClaim {
	if cs.numUserOwned == len(cs.claims) {
		return nil
	}
	return cs.claims[cs.numUserOwned]
}

// noUserClaim returns true when there is no user claim.
func (cs *claimStore) noUserClaim() bool {
	return cs.numUserOwned == 0
}

// get returns the claim at the input index
func (cs claimStore) get(i int) *resourceapi.ResourceClaim {
	return cs.claims[i]
}

// set sets the input claim at the input index for the internal claims slice.
func (cs claimStore) set(i int, c *resourceapi.ResourceClaim) {
	cs.claims[i] = c
}

// updateExtendedResourceClaim updates the input claim as extended resource
// claim in the internal claims slice.
// It returns error when there is no extended resource claim in the internal
// claims slice.
func (cs *claimStore) updateExtendedResourceClaim(c *resourceapi.ResourceClaim) error {
	if cs.numUserOwned == len(cs.claims) {
		return fmt.Errorf("no extended resource claim")
	}
	cs.claims[cs.numUserOwned] = c
	return nil
}

// getInitialExtendedResourceClaimUID returns the UID of the claim in use until
// PreBind creates the ResourceClaim in API server.
// It can only be called when extended resource claim exists.
func (cs claimStore) getInitialExtendedResourceClaimUID() types.UID {
	return cs.initialExtendedResourceClaimUID
}
