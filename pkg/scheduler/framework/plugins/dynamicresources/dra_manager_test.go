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
	"testing"

	"github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const (
	claimUID      = types.UID("claim-uid-1")
	otherClaimUID = types.UID("claim-uid-2")
)

func TestSignalClaimPendingAllocation(t *testing.T) {
	testSignalClaimPendingAllocation(ktesting.Init(t))
}
func testSignalClaimPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocations         map[types.UID]inFlightAllocation
		claimUID                    types.UID
		allocatedClaim              *resourceapi.ResourceClaim
		expectedInFlightAllocations map[types.UID]inFlightAllocation
	}{
		"empty": {
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: allocatedClaim, sharers: 1},
			},
		},
		"already-exists": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 1},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 2},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
		},
		"already-exists-ignores-different-claim-argument": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: allocatedClaim, sharers: 1},
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim2,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: allocatedClaim, sharers: 2},
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:              tCtx.Logger(),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
			}
			maps.Copy(c.inFlightAllocations, test.inFlightAllocations)

			err := c.SignalClaimPendingAllocation(test.claimUID, test.allocatedClaim)
			tCtx.ExpectNoError(err)
			tCtx.Expect(c.inFlightAllocations).To(gomega.Equal(test.expectedInFlightAllocations))
		})
	}
}

func TestGetPendingAllocation(t *testing.T) {
	testGetPendingAllocation(ktesting.Init(t))
}
func testGetPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocations map[types.UID]inFlightAllocation
		claimUID            types.UID
		expected            *resourceapi.AllocationResult
	}{
		"empty": {
			claimUID: claimUID,
			expected: nil,
		},
		"nil-claim": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: nil},
			},
			claimUID: claimUID,
			expected: nil,
		},
		"claim": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 1},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID: claimUID,
			expected: allocationResult,
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:              tCtx.Logger(),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
			}
			maps.Copy(c.inFlightAllocations, test.inFlightAllocations)
			beforeInFlight := maps.Clone(c.inFlightAllocations)

			actual := c.GetPendingAllocation(test.claimUID)
			tCtx.Expect(actual).To(gomega.Equal(test.expected))
			// Get is strictly read-only
			tCtx.Expect(c.inFlightAllocations).To(gomega.Equal(beforeInFlight))
		})
	}
}

func TestMaybeRemoveClaimPendingAllocation(t *testing.T) {
	testMaybeRemoveClaimPendingAllocation(ktesting.Init(t))
}
func testMaybeRemoveClaimPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocations         map[types.UID]inFlightAllocation
		claimUID                    types.UID
		forceRemove                 bool
		expected                    bool
		expectedInFlightAllocations map[types.UID]inFlightAllocation
	}{
		"empty": {
			claimUID:                    claimUID,
			forceRemove:                 true,
			expected:                    false,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{},
		},
		"delete-last-sharer": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 1},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID: claimUID,
			expected: true,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
		},
		"force-delete-last-sharer": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 1},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID:    claimUID,
			forceRemove: true,
			expected:    true,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
		},
		"decrement-remaining-sharers": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 2},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID: claimUID,
			expected: false,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 1},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
		},
		"force-delete-remaining-sharers": {
			inFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID:      {claim: allocatedClaim, sharers: 2},
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
			claimUID:    claimUID,
			forceRemove: true,
			expected:    true,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				otherClaimUID: {claim: allocatedClaim2, sharers: 10},
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:              tCtx.Logger(),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
			}
			maps.Copy(c.inFlightAllocations, test.inFlightAllocations)

			actual := c.MaybeRemoveClaimPendingAllocation(test.claimUID, test.forceRemove)
			tCtx.Expect(actual).To(gomega.Equal(test.expected), "wrong value for deletion indicator")
			tCtx.Expect(c.inFlightAllocations).To(gomega.Equal(test.expectedInFlightAllocations))
		})
	}
}
