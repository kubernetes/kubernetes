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

func TestAddSharedClaimPendingAllocation(t *testing.T) {
	testAddSharedClaimPendingAllocation(ktesting.Init(t))
}
func testAddSharedClaimPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocationSharers         map[types.UID]int
		claimUID                          types.UID
		allocatedClaim                    *resourceapi.ResourceClaim
		expectedInFlightAllocationSharers map[types.UID]int
	}{
		"empty": {
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocationSharers: map[types.UID]int{
				claimUID: 1,
			},
		},
		"increment": {
			inFlightAllocationSharers: map[types.UID]int{
				claimUID:      1,
				otherClaimUID: 10,
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocationSharers: map[types.UID]int{
				claimUID:      2,
				otherClaimUID: 10,
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:                    tCtx.Logger(),
				inFlightAllocationSharers: &sync.Map{},
			}

			for key, value := range test.inFlightAllocationSharers {
				c.inFlightAllocationSharers.Store(key, value)
			}
			err := c.AddSharedClaimPendingAllocation(test.claimUID, test.allocatedClaim)
			tCtx.ExpectNoError(err)
			actual := map[types.UID]int{}
			c.inFlightAllocationSharers.Range(func(key, value any) bool {
				actual[key.(types.UID)] = value.(int)
				return true
			})
			tCtx.Expect(actual).To(gomega.Equal(test.expectedInFlightAllocationSharers))
		})
	}
}

func TestRemoveSharedClaimPendingAllocation(t *testing.T) {
	testRemoveSharedClaimPendingAllocation(ktesting.Init(t))
}
func testRemoveSharedClaimPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocationSharers         map[types.UID]int
		claimUID                          types.UID
		allocatedClaim                    *resourceapi.ResourceClaim
		expectedInFlightAllocationSharers map[types.UID]int
		expectedErr                       string
	}{
		"empty": {
			inFlightAllocationSharers: map[types.UID]int{
				claimUID:      1,
				otherClaimUID: 10,
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocationSharers: map[types.UID]int{
				otherClaimUID: 10,
			},
		},
		"decrement": {
			inFlightAllocationSharers: map[types.UID]int{
				claimUID:      2,
				otherClaimUID: 10,
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocationSharers: map[types.UID]int{
				claimUID:      1,
				otherClaimUID: 10,
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:                    tCtx.Logger(),
				inFlightAllocationSharers: &sync.Map{},
			}

			for key, value := range test.inFlightAllocationSharers {
				c.inFlightAllocationSharers.Store(key, value)
			}
			err := c.RemoveSharedClaimPendingAllocation(test.claimUID, test.allocatedClaim)
			tCtx.ExpectNoError(err)
			actual := map[types.UID]int{}
			c.inFlightAllocationSharers.Range(func(key, value any) bool {
				actual[key.(types.UID)] = value.(int)
				return true
			})
			tCtx.Expect(actual).To(gomega.Equal(test.expectedInFlightAllocationSharers))
		})
	}
}

func TestMaybeRemoveClaimPendingAllocation(t *testing.T) {
	testMaybeRemoveClaimPendingAllocation(ktesting.Init(t))
}
func testMaybeRemoveClaimPendingAllocation(tCtx ktesting.TContext) {
	tests := map[string]struct {
		inFlightAllocations               map[types.UID]*resourceapi.ResourceClaim
		inFlightAllocationSharers         map[types.UID]int
		claimUID                          types.UID
		shareable                         bool
		expected                          bool
		expectedInFlightAllocationSharers map[types.UID]int
		expectedInFlightAllocations       map[types.UID]*resourceapi.ResourceClaim
	}{
		"empty": {
			claimUID:                          claimUID,
			shareable:                         true,
			expected:                          false,
			expectedInFlightAllocations:       map[types.UID]*resourceapi.ResourceClaim{},
			expectedInFlightAllocationSharers: map[types.UID]int{},
		},
		"delete-existing-not-shareable": {
			inFlightAllocations: map[types.UID]*resourceapi.ResourceClaim{
				claimUID: allocatedClaim,
			},
			claimUID:                          claimUID,
			expected:                          true,
			expectedInFlightAllocations:       map[types.UID]*resourceapi.ResourceClaim{},
			expectedInFlightAllocationSharers: map[types.UID]int{},
		},
		"delete-existing-shareable-unshared": {
			inFlightAllocations: map[types.UID]*resourceapi.ResourceClaim{
				claimUID: allocatedClaim,
			},
			inFlightAllocationSharers: map[types.UID]int{
				otherClaimUID: 10,
			},
			claimUID:                    claimUID,
			shareable:                   true,
			expected:                    true,
			expectedInFlightAllocations: map[types.UID]*resourceapi.ResourceClaim{},
			expectedInFlightAllocationSharers: map[types.UID]int{
				otherClaimUID: 10,
			},
		},
		"keep-existing-shareable-shared": {
			inFlightAllocations: map[types.UID]*resourceapi.ResourceClaim{
				claimUID: allocatedClaim,
			},
			inFlightAllocationSharers: map[types.UID]int{
				claimUID:      1,
				otherClaimUID: 0,
			},
			claimUID:  claimUID,
			shareable: true,
			expected:  false,
			expectedInFlightAllocations: map[types.UID]*resourceapi.ResourceClaim{
				claimUID: allocatedClaim,
			},
			expectedInFlightAllocationSharers: map[types.UID]int{
				claimUID:      1,
				otherClaimUID: 0,
			},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:                    tCtx.Logger(),
				inFlightAllocations:       &sync.Map{},
				inFlightAllocationSharers: &sync.Map{},
			}

			for key, value := range test.inFlightAllocations {
				c.inFlightAllocations.Store(key, value)
			}
			for key, value := range test.inFlightAllocationSharers {
				c.inFlightAllocationSharers.Store(key, value)
			}
			actual := c.MaybeRemoveClaimPendingAllocation(test.claimUID, test.shareable)
			tCtx.Expect(actual).To(gomega.Equal(test.expected), "wrong value for deletion indicator")
			actualInFlight := map[types.UID]*resourceapi.ResourceClaim{}
			c.inFlightAllocations.Range(func(key, value any) bool {
				actualInFlight[key.(types.UID)] = value.(*resourceapi.ResourceClaim)
				return true
			})
			tCtx.Expect(actualInFlight).To(gomega.Equal(test.expectedInFlightAllocations))
			actualSharers := map[types.UID]int{}
			c.inFlightAllocationSharers.Range(func(key, value any) bool {
				actualSharers[key.(types.UID)] = value.(int)
				return true
			})
			tCtx.Expect(actualSharers).To(gomega.Equal(test.expectedInFlightAllocationSharers))
		})
	}
}
