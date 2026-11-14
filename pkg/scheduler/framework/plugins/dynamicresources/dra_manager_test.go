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
	"fmt"
	"maps"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/onsi/gomega"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

const (
	claimUID      = types.UID("claim-uid-1")
	otherClaimUID = types.UID("claim-uid-2")
)

// testInformer implements [assumecache.Informer] and can be used to feed changes into an assume
// cache during unit testing. Only a single event handler is supported, which is
// sufficient for one assume cache.
type testInformer struct {
	handler cache.ResourceEventHandler
}

func (i *testInformer) AddEventHandlerWithOptions(handler cache.ResourceEventHandler, options cache.HandlerOptions) (cache.ResourceEventHandlerRegistration, error) {
	i.handler = handler
	return nil, nil
}

func (i *testInformer) add(obj any) {
	if i.handler == nil {
		return
	}
	i.handler.OnAdd(obj, false)
}

func TestSignalClaimPendingAllocation(t *testing.T) {
	testSignalClaimPendingAllocation(ktesting.Init(t))
}
func testSignalClaimPendingAllocation(tCtx ktesting.TContext) {
	specialAllocatedClaim := allocatedClaim.DeepCopy()
	specialAllocatedClaim.Name = specialClaimInMemName

	tests := map[string]struct {
		inFlightAllocations         map[types.UID]inFlightAllocation
		cachedClaims                []*resourceapi.ResourceClaim
		claimUID                    types.UID
		allocatedClaim              *resourceapi.ResourceClaim
		expectedInFlightAllocations map[types.UID]inFlightAllocation
	}{
		"empty": {
			cachedClaims: []*resourceapi.ResourceClaim{
				st.FromResourceClaim(pendingClaim).UID(string(claimUID)).Obj(),
			},
			claimUID:       claimUID,
			allocatedClaim: allocatedClaim,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: allocatedClaim, sharers: 1},
			},
		},
		"claim-already-allocated": {
			claimUID: claimUID,
			cachedClaims: []*resourceapi.ResourceClaim{
				st.FromResourceClaim(allocatedClaim).UID(string(claimUID)).Obj(),
			},
			allocatedClaim:              allocatedClaim,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{},
		},
		"extended-resource-claim-not-in-cache": {
			claimUID:       claimUID,
			allocatedClaim: specialAllocatedClaim,
			expectedInFlightAllocations: map[types.UID]inFlightAllocation{
				claimUID: {claim: specialAllocatedClaim, sharers: 1},
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

			informer := &testInformer{}
			c.cache = assumecache.NewAssumeCache(tCtx.Logger(), informer, "", "", nil)
			for _, cached := range test.cachedClaims {
				informer.add(cached)
			}

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

const (
	sharedDriver = "driver.example.com"
	sharedPool   = "pool-0"
	sharedDevice = "device-0"
)

// claimWithConsumedCapacity returns a claim with one share of the same device.
func claimWithConsumedCapacity(uid types.UID, memory string) *resourceapi.ResourceClaim {
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: string(uid), UID: uid},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Results: []resourceapi.DeviceRequestAllocationResult{{
						Request:          "req-0",
						Driver:           sharedDriver,
						Pool:             sharedPool,
						Device:           sharedDevice,
						ShareID:          new(uid + "-share"),
						ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"memory": resource.MustParse(memory)},
					}},
				},
			},
		},
	}
}

func TestGatherAllocatedStateConsumedCapacity(t *testing.T) {
	testGatherAllocatedStateConsumedCapacity(ktesting.Init(t))
}
func testGatherAllocatedStateConsumedCapacity(tCtx ktesting.TContext) {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)

	valid := claimWithConsumedCapacity("valid", "5")
	otherValid := claimWithConsumedCapacity("other-valid", "1")
	negative := claimWithConsumedCapacity("negative", "-2")
	otherNegative := claimWithConsumedCapacity("other-negative", "-3")
	small := claimWithConsumedCapacity("small", "2")
	negativeLarge := claimWithConsumedCapacity("negative-large", "-5")
	// The qualified name replaces the unqualified one in the aggregated capacity.
	negativeUnqualified := claimWithConsumedCapacity("negative-unqualified", "-2")
	negativeUnqualified.Status.Allocation.Devices.Results[0].ConsumedCapacity[sharedDriver+"/memory"] = resource.MustParse("5")
	negativeDedicated := claimWithConsumedCapacity("negative-dedicated", "-2")
	negativeDedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	negativeAdminAccess := claimWithConsumedCapacity("negative-admin-access", "-2")
	negativeAdminAccess.Status.Allocation.Devices.Results[0].AdminAccess = new(true)

	tests := map[string]struct {
		allocated []*resourceapi.ResourceClaim
		removed   []*resourceapi.ResourceClaim
		inFlight  []*resourceapi.ResourceClaim
		// expectErr is a substring of the expected error. If empty,
		// expectMemory is the aggregated memory of the shared device.
		expectErr    string
		expectMemory string
	}{
		"valid": {
			allocated:    []*resourceapi.ResourceClaim{valid, otherValid},
			expectMemory: "6",
		},
		"negative-alone": {
			allocated: []*resourceapi.ResourceClaim{negative},
			expectErr: "claim default/negative, device driver.example.com/pool-0/device-0: consumed capacity \"memory\" must not be negative, got -2",
		},
		"negative-after-positive": {
			allocated: []*resourceapi.ResourceClaim{valid, negative},
			expectErr: "claim default/negative",
		},
		// The aggregated capacity would be 3, which is not negative.
		"negative-hidden-by-later-positive": {
			allocated: []*resourceapi.ResourceClaim{negative, valid},
			expectErr: "claim default/negative",
		},
		"several-negative": {
			allocated: []*resourceapi.ResourceClaim{negative, otherNegative},
			expectErr: "claim default/negative, device driver.example.com/pool-0/device-0: consumed capacity \"memory\" must not be negative, got -2 (and 1 more claims with invalid consumed capacity)",
		},
		"negative-in-flight": {
			allocated: []*resourceapi.ResourceClaim{valid},
			inFlight:  []*resourceapi.ResourceClaim{negative},
			expectErr: "claim default/negative",
		},
		"negative-unqualified": {
			allocated: []*resourceapi.ResourceClaim{negativeUnqualified},
			expectErr: "claim default/negative-unqualified, device driver.example.com/pool-0/device-0: consumed capacity \"memory\" must not be negative, got -2",
		},
		"negative-unqualified-in-flight": {
			inFlight:  []*resourceapi.ResourceClaim{negativeUnqualified},
			expectErr: "claim default/negative-unqualified",
		},
		"negative-dedicated": {
			allocated: []*resourceapi.ResourceClaim{negativeDedicated},
			expectErr: "claim default/negative-dedicated",
		},
		"negative-admin-access": {
			allocated: []*resourceapi.ResourceClaim{negativeAdminAccess},
			expectErr: "claim default/negative-admin-access",
		},
		"negative-admin-access-removed": {
			allocated:    []*resourceapi.ResourceClaim{negativeAdminAccess, valid},
			removed:      []*resourceapi.ResourceClaim{negativeAdminAccess},
			expectMemory: "5",
		},
		"negative-removed": {
			allocated:    []*resourceapi.ResourceClaim{negative, valid},
			removed:      []*resourceapi.ResourceClaim{negative},
			expectMemory: "5",
		},
		// 5 + 2 - 5 becomes zero when the small claim is removed, which must
		// not drop the device while the valid claim still consumes it.
		"negative-removed-after-sum-is-zero": {
			allocated:    []*resourceapi.ResourceClaim{valid, small, negativeLarge},
			removed:      []*resourceapi.ResourceClaim{small, negativeLarge},
			expectMemory: "5",
		},
		"one-of-several-negative-removed": {
			allocated: []*resourceapi.ResourceClaim{negative, otherNegative, valid},
			removed:   []*resourceapi.ResourceClaim{negative},
			expectErr: "claim default/other-negative",
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			c := &claimTracker{
				logger:              tCtx.Logger(),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
				allocatedDevices:    newAllocatedDevices(tCtx.Logger()),
			}
			for _, claim := range test.allocated {
				c.allocatedDevices.onAdd(claim)
			}
			for _, claim := range test.removed {
				c.allocatedDevices.onDelete(claim)
			}
			for _, claim := range test.inFlight {
				c.inFlightAllocations[claim.UID] = inFlightAllocation{claim: claim, sharers: 1}
			}

			state, err := c.GatherAllocatedState()
			if test.expectErr != "" {
				tCtx.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(test.expectErr)))
				return
			}
			tCtx.ExpectNoError(err)
			deviceID := structured.MakeDeviceID(sharedDriver, sharedPool, sharedDevice)
			capacity := state.AggregatedCapacity[deviceID]
			tCtx.Expect(capacity).To(gomega.HaveLen(1))
			for _, memory := range capacity {
				tCtx.Expect(memory.Cmp(resource.MustParse(test.expectMemory))).To(gomega.BeZero(), "aggregated memory %s", memory)
			}
		})
	}
}

func TestGatherAllocatedStateShareCounts(t *testing.T) {
	testGatherAllocatedStateShareCounts(ktesting.Init(t))
}
func testGatherAllocatedStateShareCounts(tCtx ktesting.TContext) {
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)

	device := structured.MakeDeviceID(sharedDriver, sharedPool, sharedDevice)
	share := claimWithConsumedCapacity("share", "1")
	otherShare := claimWithConsumedCapacity("other-share", "1")
	dedicated := claimWithConsumedCapacity("dedicated", "1")
	dedicated.Status.Allocation.Devices.Results[0].ShareID = nil

	tests := map[string]struct {
		disablePreemption bool
		// preemptionOnlyAtCall enables DRAPreemption only after newAllocatedDevices
		// runs; share counting is initialized at creation time, so counts stay nil.
		preemptionOnlyAtCall bool
		allocated            []*resourceapi.ResourceClaim
		inFlight             []*resourceapi.ResourceClaim
		expectCounts         map[structured.DeviceID]int
	}{
		"without-DRAPreemption": {
			disablePreemption: true,
			allocated:         []*resourceapi.ResourceClaim{share, otherShare},
		},
		"DRAPreemption-only-at-call": {
			preemptionOnlyAtCall: true,
			allocated:            []*resourceapi.ResourceClaim{share, otherShare},
		},
		"allocated": {
			allocated:    []*resourceapi.ResourceClaim{share, otherShare},
			expectCounts: map[structured.DeviceID]int{device: 2},
		},
		"in-flight": {
			allocated:    []*resourceapi.ResourceClaim{share},
			inFlight:     []*resourceapi.ResourceClaim{otherShare},
			expectCounts: map[structured.DeviceID]int{device: 2},
		},
		// During PreBind, a claim can be both in flight and in the assume cache.
		"in-flight-and-allocated": {
			allocated:    []*resourceapi.ResourceClaim{share},
			inFlight:     []*resourceapi.ResourceClaim{share},
			expectCounts: map[structured.DeviceID]int{device: 1},
		},
		"dedicated": {
			allocated:    []*resourceapi.ResourceClaim{dedicated},
			expectCounts: map[structured.DeviceID]int{},
		},
	}

	for name, test := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, !test.disablePreemption && !test.preemptionOnlyAtCall)
			c := &claimTracker{
				logger:              tCtx.Logger(),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
				allocatedDevices:    newAllocatedDevices(tCtx.Logger()),
			}
			if test.preemptionOnlyAtCall {
				featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAPreemption, true)
			}
			for _, claim := range test.allocated {
				c.allocatedDevices.onAdd(claim)
			}
			for _, claim := range test.inFlight {
				c.inFlightAllocations[claim.UID] = inFlightAllocation{claim: claim, sharers: 1}
			}

			state, err := c.GatherAllocatedState()
			tCtx.ExpectNoError(err)
			tCtx.Expect(state.AllocatedShareCounts).To(gomega.Equal(test.expectCounts))
		})
	}
}

func claimWithShares(i int, shares ...int) *resourceapi.ResourceClaim {
	uid := types.UID(fmt.Sprintf("claim-%d", i))
	claim := &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: string(uid), UID: uid},
		Status:     resourceapi.ResourceClaimStatus{Allocation: &resourceapi.AllocationResult{}},
	}
	for _, s := range shares {
		shareID := types.UID(fmt.Sprintf("share-%d", s))
		claim.Status.Allocation.Devices.Results = append(claim.Status.Allocation.Devices.Results, resourceapi.DeviceRequestAllocationResult{
			Request: "req-0", Driver: "driver", Pool: "pool", Device: fmt.Sprintf("dev-%d", s%3),
			ShareID:          &shareID,
			ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{"memory": resource.MustParse("1Gi")},
		})
	}
	return claim
}

// TestShareCountsMatchShareIDs verifies that shareCounts stays in sync with
// shareIDs across random add, update, and delete events (including duplicates).
func TestShareCountsMatchShareIDs(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRAPreemption:         true,
	})
	r := rand.New(rand.NewPCG(1, 2))
	a := newAllocatedDevices(klog.Background())
	claims := make([]*resourceapi.ResourceClaim, 20)
	for i := range claims {
		claims[i] = claimWithShares(i, 2*i, 2*i+1)
	}
	deallocated := func(claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
		claim = claim.DeepCopy()
		claim.Status.Allocation = nil
		return claim
	}
	for step := range 5000 {
		claim := claims[r.IntN(len(claims))]
		switch r.IntN(4) {
		case 0:
			a.onAdd(claim)
		case 1:
			a.onDelete(claim)
		case 2:
			a.onUpdate(deallocated(claim), claim)
		case 3:
			a.onUpdate(claim, deallocated(claim))
		}
		want := make(map[structured.DeviceID]int)
		for id := range a.shareIDs {
			want[id.GetDeviceID()]++
		}
		if len(want) != len(a.shareCounts) {
			t.Fatalf("step %d: counts %v, want %v", step, a.shareCounts, want)
		}
		for id, n := range want {
			if a.shareCounts[id] != n {
				t.Fatalf("step %d: count of %s is %d, want %d", step, id, a.shareCounts[id], n)
			}
		}
	}
}

// TestGatherAllocatedStateConcurrent verifies that GatherAllocatedState
// returns share counts consistent with AllocatedSharedDeviceIDs while cached
// claims and in-flight allocations mutate concurrently.
func TestGatherAllocatedStateConcurrent(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRAPreemption:         true,
	})
	logger := klog.Background()
	c := &claimTracker{
		logger:              logger,
		inFlightAllocations: make(map[types.UID]inFlightAllocation),
		allocatedDevices:    newAllocatedDevices(logger),
	}
	claims := make([]*resourceapi.ResourceClaim, 10)
	for i := range claims {
		claims[i] = claimWithShares(i, i)
	}
	var wg sync.WaitGroup
	stop := make(chan struct{})
	wg.Go(func() {
		r := rand.New(rand.NewPCG(3, 4))
		for {
			select {
			case <-stop:
				return
			default:
			}
			claim := claims[r.IntN(len(claims))]
			switch r.IntN(4) {
			case 0:
				c.allocatedDevices.onAdd(claim)
			case 1:
				c.allocatedDevices.onDelete(claim)
			case 2:
				c.inFlightMutex.Lock()
				c.inFlightAllocations[claim.UID] = inFlightAllocation{claim: claim, sharers: 1}
				c.inFlightMutex.Unlock()
			case 3:
				c.inFlightMutex.Lock()
				delete(c.inFlightAllocations, claim.UID)
				c.inFlightMutex.Unlock()
			}
		}
	})
	succeeded := 0
	for range 2000 {
		state, err := c.GatherAllocatedState()
		if err != nil {
			continue
		}
		succeeded++
		total := 0
		for id, n := range state.AllocatedShareCounts {
			if !state.AllocatedSharedDeviceIDs.Has(id) || n < 1 {
				t.Errorf("count %d for %s, shared devices %v", n, id, state.AllocatedSharedDeviceIDs)
			}
			total += n
		}
		if len(state.AllocatedShareCounts) != state.AllocatedSharedDeviceIDs.Len() {
			t.Errorf("counts %v don't cover shared devices %v", state.AllocatedShareCounts, state.AllocatedSharedDeviceIDs)
		}
		// Each claim has a single unique share, so total exceeds len(claims)
		// only if an in-flight share already in the cache is double-counted.
		if total > len(claims) {
			t.Errorf("total %d exceeds number of claims", total)
		}
	}
	close(stop)
	wg.Wait()
	if succeeded == 0 {
		t.Fatal("GatherAllocatedState never succeeded, nothing got checked")
	}
}
