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

// TestGatherAllocatedStateConsumedCounters exercises bookkeeping without a claim-list cache.
func TestGatherAllocatedStateConsumedCounters(t *testing.T) {
	testGatherAllocatedStateConsumedCounters(ktesting.Init(t))
}
func testGatherAllocatedStateConsumedCounters(tCtx ktesting.TContext) {
	withCounters := func(uid types.UID, amount string) *resourceapi.ResourceClaim {
		claim := claimWithConsumedCapacity(uid, "1")
		claim.Status.Allocation.Devices.Results[0].ConsumedCounters = []resourceapi.CounterSetConsumption{{
			CounterSet: "shared", Counters: map[string]resource.Quantity{"memory": resource.MustParse(amount)},
		}}
		return claim
	}
	first, second := withCounters("first", "5"), withCounters("second", "3")
	zero := withCounters("zero", "0")
	unknown := claimWithConsumedCapacity("unknown", "1")
	unknownDedicated := unknown.DeepCopy()
	unknownDedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	admin := withCounters("admin", "7")
	admin.Status.Allocation.Devices.Results[0].AdminAccess = new(true)
	dedicated := withCounters("dedicated", "2")
	dedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	overridden := withCounters("first", "7")
	missingOverride := claimWithConsumedCapacity("first", "1")

	tests := map[string]struct {
		allocated []*resourceapi.ResourceClaim
		removed   []*resourceapi.ResourceClaim
		inFlight  []*resourceapi.ResourceClaim
		disabled  bool
		want      []string
	}{
		"ordinary DRA":                                  {allocated: []*resourceapi.ResourceClaim{unknown}},
		"disabled":                                      {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{second}, disabled: true},
		"separate share snapshots":                      {allocated: []*resourceapi.ResourceClaim{first, second}, want: []string{"5", "3"}},
		"cached and in-flight duplicate":                {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{first}, want: []string{"5"}},
		"cached missing share duplicated in flight":     {allocated: []*resourceapi.ResourceClaim{unknown, first}, inFlight: []*resourceapi.ResourceClaim{unknown}, want: []string{"missing", "5"}},
		"cached and in-flight shares including missing": {allocated: []*resourceapi.ResourceClaim{unknown, first, second}, inFlight: []*resourceapi.ResourceClaim{unknown, first, second}, want: []string{"missing", "5", "3"}},
		"in-flight replaces cached snapshot":            {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{overridden}, want: []string{"7"}},
		"in-flight replaces snapshot with unknown":      {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{missingOverride}},
		"missing cached dedicated allocation":           {allocated: []*resourceapi.ResourceClaim{unknownDedicated, first}, want: []string{"missing", "5"}},
		"missing in-flight dedicated allocation":        {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{unknownDedicated}, want: []string{"missing", "5"}},
		"snapshot after cached dedicated allocation":    {allocated: []*resourceapi.ResourceClaim{unknownDedicated}, inFlight: []*resourceapi.ResourceClaim{first}, want: []string{"missing", "5"}},
		"missing cached share":                          {allocated: []*resourceapi.ResourceClaim{unknown, first}, want: []string{"missing", "5"}},
		"missing in-flight share":                       {allocated: []*resourceapi.ResourceClaim{first}, inFlight: []*resourceapi.ResourceClaim{unknown}, want: []string{"missing", "5"}},
		"snapshot after missing in-flight share":        {inFlight: []*resourceapi.ResourceClaim{unknown, first}, want: []string{"missing", "5"}},
		"release first share":                           {allocated: []*resourceapi.ResourceClaim{first, second}, removed: []*resourceapi.ResourceClaim{first}, want: []string{"3"}},
		"release last share":                            {allocated: []*resourceapi.ResourceClaim{first}, removed: []*resourceapi.ResourceClaim{first}},
		"release missing share":                         {allocated: []*resourceapi.ResourceClaim{unknown, zero}, removed: []*resourceapi.ResourceClaim{unknown}, want: []string{"0"}},
		"known zero":                                    {allocated: []*resourceapi.ResourceClaim{zero}, want: []string{"0"}},
		"zero and missing":                              {allocated: []*resourceapi.ResourceClaim{zero, unknown}, want: []string{"0", "missing"}},
		"admin excluded":                                {allocated: []*resourceapi.ResourceClaim{admin, first}, want: []string{"5"}},
		"dedicated allocation":                          {allocated: []*resourceapi.ResourceClaim{dedicated}, want: []string{"2"}},
	}
	for name, tc := range tests {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, true)
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRASharedConsumableCapacity, !tc.disabled)
			tracker := &claimTracker{logger: tCtx.Logger(), allocatedDevices: newAllocatedDevices(tCtx.Logger()), inFlightAllocations: make(map[types.UID]inFlightAllocation)}
			for _, claim := range tc.allocated {
				tracker.allocatedDevices.onAdd(claim)
			}
			for _, claim := range tc.removed {
				tracker.allocatedDevices.onDelete(claim)
			}
			for _, claim := range tc.inFlight {
				tracker.inFlightAllocations[claim.UID] = inFlightAllocation{claim: claim, sharers: 1}
			}
			state, err := tracker.GatherAllocatedState()
			tCtx.ExpectNoError(err)
			if tc.want == nil {
				tCtx.Expect(state.ConsumedCounters).To(gomega.BeNil())
				return
			}
			tCtx.Expect(state.ConsumedCounters).To(gomega.HaveLen(1))
			var amounts []string
			for _, snapshot := range state.ConsumedCounters[structured.MakeDeviceID(sharedDriver, sharedPool, sharedDevice)] {
				if len(snapshot) == 0 {
					amounts = append(amounts, "missing")
					continue
				}
				tCtx.Expect(snapshot).To(gomega.HaveLen(1))
				amount := snapshot[0].Counters["memory"]
				amounts = append(amounts, amount.String())
			}
			tCtx.Expect(amounts).To(gomega.ConsistOf(tc.want))
		})
	}
}

func TestGatherAllocatedStateParity(t *testing.T) {
	testGatherAllocatedStateParity(ktesting.Init(t))
}

// testGatherAllocatedStateParity checks that scheduler and allocator tests construct the same state.
func testGatherAllocatedStateParity(tCtx ktesting.TContext) {
	known := claimWithConsumedCapacity("known", "1")
	known.Status.Allocation.Devices.Results[0].ConsumedCounters = []resourceapi.CounterSetConsumption{{
		CounterSet: "shared", Counters: map[string]resource.Quantity{"memory": resource.MustParse("5")},
	}}
	zero := known.DeepCopy()
	zero.UID = "zero"
	zero.Status.Allocation.Devices.Results[0].ShareID = new(types.UID("zero-share"))
	zero.Status.Allocation.Devices.Results[0].ConsumedCounters[0].Counters["memory"] = resource.MustParse("0")
	missing := claimWithConsumedCapacity("missing", "2")
	dedicated := known.DeepCopy()
	dedicated.UID = "dedicated"
	dedicated.Status.Allocation.Devices.Results[0].Device = "dedicated-device"
	dedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	admin := known.DeepCopy()
	admin.UID = "admin"
	admin.Status.Allocation.Devices.Results[0].AdminAccess = new(true)
	unallocated := claimWithConsumedCapacity("unallocated", "1")
	unallocated.Status.Allocation = nil

	for name, claims := range map[string][]*resourceapi.ResourceClaim{
		"empty":                 nil,
		"known":                 {known},
		"missing":               {missing},
		"missing-before-known":  {missing, known},
		"known-before-missing":  {known, missing},
		"zero-and-missing":      {zero, missing},
		"dedicated-and-shared":  {dedicated, known},
		"admin-and-unallocated": {admin, unallocated, known},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			for _, enabled := range []structured.Features{
				{},
				{ConsumableCapacity: true},
				{ConsumableCapacity: true, SharedConsumableCapacity: true},
			} {
				tCtx.Run(fmt.Sprintf("capacity=%t/counters=%t", enabled.ConsumableCapacity, enabled.SharedConsumableCapacity), func(tCtx ktesting.TContext) {
					featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, enabled.ConsumableCapacity)
					featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRASharedConsumableCapacity, enabled.SharedConsumableCapacity)
					var expected structured.AllocatedState
					structured.AddToAllocatedState(&expected, func(yield func(*resourceapi.AllocationResult) bool) {
						for _, claim := range claims {
							if !yield(claim.Status.Allocation) {
								return
							}
						}
					}, enabled)
					for split := 0; split <= len(claims); split++ {
						tCtx.Run(fmt.Sprintf("cached=%d", split), func(tCtx ktesting.TContext) {
							tracker := &claimTracker{logger: tCtx.Logger(), allocatedDevices: newAllocatedDevices(tCtx.Logger()), inFlightAllocations: make(map[types.UID]inFlightAllocation)}
							before := make([]*resourceapi.ResourceClaim, len(claims))
							for i, claim := range claims {
								before[i] = claim.DeepCopy()
								if i < split {
									tracker.allocatedDevices.onAdd(claim)
								} else {
									tracker.inFlightAllocations[claim.UID] = inFlightAllocation{claim: claim, sharers: 1}
								}
							}
							actual, err := tracker.GatherAllocatedState()
							tCtx.ExpectNoError(err)
							tCtx.Expect(actual.AllocatedDevices).To(gomega.Equal(expected.AllocatedDevices))
							tCtx.Expect(actual.AllocatedSharedDeviceIDs).To(gomega.Equal(expected.AllocatedSharedDeviceIDs))
							tCtx.Expect(actual.AggregatedCapacity).To(gomega.Equal(expected.AggregatedCapacity))
							if expected.ConsumedCounters == nil {
								tCtx.Expect(actual.ConsumedCounters).To(gomega.BeNil())
							} else {
								tCtx.Expect(actual.ConsumedCounters).To(gomega.HaveLen(len(expected.ConsumedCounters)))
								for id, snapshots := range expected.ConsumedCounters {
									tCtx.Expect(actual.ConsumedCounters).To(gomega.HaveKey(id))
									tCtx.Expect(actual.ConsumedCounters[id]).To(gomega.ConsistOf(snapshots))
								}
							}
							for i, claim := range claims {
								tCtx.Expect(claim).To(gomega.Equal(before[i]))
							}
						})
					}
				})
			}
		})
	}
}
