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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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

// TestAllocatedDevicesReplacesAllocation covers informer updates that skip a
// cleared allocation or claim deletion, including cleanup of the replacement.
func TestAllocatedDevicesReplacesAllocation(t *testing.T) {
	testAllocatedDevicesReplacesAllocation(ktesting.Init(t))
}

func testAllocatedDevicesReplacesAllocation(tCtx ktesting.TContext) {
	original := claimWithConsumedCapacity("original", "1")
	original.ResourceVersion = "1"
	replacement := claimWithConsumedCapacity("replacement", "2")
	replacement.Name = original.Name
	replacement.ResourceVersion = "3"
	sameUID := replacement.DeepCopy()
	sameUID.UID = original.UID
	other := claimWithConsumedCapacity("other", "3")
	other.ResourceVersion = "1"

	oldDedicated := original.DeepCopy()
	oldDedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	oldDedicated.Status.Allocation.Devices.Results[0].Device = "old-device"
	newDedicated := replacement.DeepCopy()
	newDedicated.Status.Allocation.Devices.Results[0].ShareID = nil
	newDedicated.Status.Allocation.Devices.Results[0].Device = "new-device"
	newSharedDevice := replacement.DeepCopy()
	newSharedDevice.Status.Allocation.Devices.Results[0].Device = "new-device"
	admin := replacement.DeepCopy()
	admin.Status.Allocation.Devices.Results[0].AdminAccess = new(true)
	oldInvalid := original.DeepCopy()
	oldInvalid.Status.Allocation.Devices.Results[0].ConsumedCapacity["memory"] = resource.MustParse("-1")
	newInvalid := replacement.DeepCopy()
	newInvalid.Status.Allocation.Devices.Results[0].ConsumedCapacity["memory"] = resource.MustParse("-2")

	for name, tc := range map[string]struct {
		old, replacement *resourceapi.ResourceClaim
		disabled         bool
		expectErr        string
	}{
		"recreated claim":              {old: original, replacement: replacement},
		"reallocated claim":            {old: original, replacement: sameUID},
		"different dedicated device":   {old: oldDedicated, replacement: newDedicated},
		"admin replacement":            {old: original, replacement: admin},
		"invalid allocation removed":   {old: oldInvalid, replacement: replacement},
		"invalid replacement":          {old: original, replacement: newInvalid, expectErr: "claim default/original"},
		"consumable capacity disabled": {old: oldDedicated, replacement: newSharedDevice, disabled: true},
	} {
		tCtx.Run(name, func(tCtx ktesting.TContext) {
			featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.DRAConsumableCapacity, !tc.disabled)
			informer := &testInformer{}
			tracker := &claimTracker{
				logger:              tCtx.Logger(),
				allocatedDevices:    newAllocatedDevices(tCtx.Logger()),
				inFlightAllocations: make(map[types.UID]inFlightAllocation),
			}
			tracker.cache = assumecache.NewAssumeCache(tCtx.Logger(), informer, "", "", nil)
			tracker.cache.AddEventHandler(tracker.allocatedDevices.handlers())
			informer.add(other)
			informer.add(tc.old)
			revision := tracker.allocatedDevices.getRevision()

			// Informers may skip deletion/recreation or an intermediate cleared allocation.
			informer.handler.OnUpdate(tc.old, tc.replacement)
			cached, err := tracker.Get(tc.replacement.Namespace, tc.replacement.Name)
			tCtx.ExpectNoError(err)
			tCtx.Expect(cached).To(gomega.Equal(tc.replacement))

			expectAllocations := func(claims ...*resourceapi.ResourceClaim) {
				tCtx.Helper()
				var expected structured.AllocatedState
				structured.AddToAllocatedState(&expected, func(yield func(*resourceapi.AllocationResult) bool) {
					for _, claim := range claims {
						if !yield(claim.Status.Allocation) {
							return
						}
					}
				}, structured.Features{ConsumableCapacity: !tc.disabled})
				actual, err := tracker.GatherAllocatedState()
				tCtx.ExpectNoError(err)
				tCtx.Expect(apiequality.Semantic.DeepEqual(actual, &expected)).To(gomega.BeTrue(), "actual allocation state: %v, expected: %v", actual, &expected)
			}
			if tc.expectErr != "" {
				_, err := tracker.GatherAllocatedState()
				tCtx.Expect(err).To(gomega.MatchError(gomega.ContainSubstring(tc.expectErr)))
			} else {
				expectAllocations(other, tc.replacement)
			}

			tCtx.Expect(tracker.allocatedDevices.getRevision()).To(gomega.Equal(revision + 1))

			unchanged := tc.replacement.DeepCopy()
			unchanged.ResourceVersion = "4"
			unchanged.Labels = map[string]string{"metadata": "updated"}
			revision = tracker.allocatedDevices.getRevision()
			informer.handler.OnUpdate(tc.replacement, unchanged)
			tCtx.Expect(tracker.allocatedDevices.getRevision()).To(gomega.Equal(revision))

			informer.handler.OnDelete(unchanged)
			expectAllocations(other)
		})
	}
}

// TestAllocatedDevicesAllocationsEqual checks the accounting boundary without reflection.
func TestAllocatedDevicesAllocationsEqual(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		a := &allocatedDevices{enabledConsumableCapacity: enabled}
		original := claimWithConsumedCapacity("original", "1").Status.Allocation
		for name, tc := range map[string]struct {
			mutate func(*resourceapi.AllocationResult)
			equal  bool
		}{
			"unchanged":     {func(x *resourceapi.AllocationResult) {}, true},
			"driver":        {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].Driver = "other" }, false},
			"pool":          {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].Pool = "other" }, false},
			"device":        {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].Device = "other" }, false},
			"admin":         {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].AdminAccess = new(true) }, false},
			"default admin": {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].AdminAccess = new(false) }, true},
			"share":         {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].ShareID = new(types.UID("other")) }, !enabled},
			"dedicated":     {func(x *resourceapi.AllocationResult) { x.Devices.Results[0].ShareID = nil }, !enabled},
			"capacity": {func(x *resourceapi.AllocationResult) {
				x.Devices.Results[0].ConsumedCapacity["memory"] = resource.MustParse("2")
			}, !enabled},
			"negative capacity": {func(x *resourceapi.AllocationResult) {
				x.Devices.Results[0].ConsumedCapacity["memory"] = resource.MustParse("-1")
			}, !enabled},
			"equal quantity": {func(x *resourceapi.AllocationResult) {
				x.Devices.Results[0].ConsumedCapacity["memory"] = resource.MustParse("1000m")
			}, true},
			"capacity key": {func(x *resourceapi.AllocationResult) {
				x.Devices.Results[0].ConsumedCapacity = map[resourceapi.QualifiedName]resource.Quantity{"other": resource.MustParse("1")}
			}, !enabled},
			"added zero capacity": {func(x *resourceapi.AllocationResult) {
				x.Devices.Results[0].ConsumedCapacity["other"] = resource.MustParse("0")
			}, !enabled},
			"results removed": {func(x *resourceapi.AllocationResult) { x.Devices.Results = nil }, false},
			"untracked fields": {func(x *resourceapi.AllocationResult) {
				x.AllocationTimestamp = &metav1.Time{}
				x.Devices.Results[0].Request = "other"
				x.Devices.Results[0].BindingConditions = []string{"Ready"}
			}, true},
		} {
			t.Run(fmt.Sprintf("capacity=%t/%s", enabled, name), func(t *testing.T) {
				modified := original.DeepCopy()
				tc.mutate(modified)
				if got := a.allocationsEqual(original, modified); got != tc.equal {
					t.Fatalf("equal = %t, want %t", got, tc.equal)
				}
			})
		}
		if !a.allocationsEqual(nil, nil) || a.allocationsEqual(nil, original) || a.allocationsEqual(original, nil) {
			t.Fatal("incorrect nil allocation comparison")
		}
		empty := &resourceapi.AllocationResult{}
		emptySlice := &resourceapi.AllocationResult{Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{}}}
		if !a.allocationsEqual(empty, emptySlice) {
			t.Fatal("nil and empty results differ")
		}
	}
}

// BenchmarkAllocatedDevicesAllocationsEqual compares deep-copied unchanged informer allocations.
func BenchmarkAllocatedDevicesAllocationsEqual(b *testing.B) {
	for _, count := range []int{1, 32} {
		for _, capacities := range []int{0, 1, 32} {
			original := claimWithConsumedCapacity("original", "1").Status.Allocation
			result := original.Devices.Results[0]
			result.ConsumedCapacity = make(map[resourceapi.QualifiedName]resource.Quantity)
			for i := 0; i < capacities; i++ {
				result.ConsumedCapacity[resourceapi.QualifiedName(fmt.Sprintf("capacity%d", i))] = resource.MustParse("1")
			}
			original.Devices.Results = make([]resourceapi.DeviceRequestAllocationResult, count)
			for i := range original.Devices.Results {
				original.Devices.Results[i] = result
				original.Devices.Results[i].Device = fmt.Sprintf("device%d", i)
			}
			modified := original.DeepCopy()
			a := &allocatedDevices{enabledConsumableCapacity: true}
			for name, compare := range map[string]func() bool{
				"semantic": func() bool { return apiequality.Semantic.DeepEqual(original, modified) },
				"typed":    func() bool { return a.allocationsEqual(original, modified) },
			} {
				b.Run(fmt.Sprintf("devices=%d/capacities=%d/%s", count, capacities, name), func(b *testing.B) {
					b.ReportAllocs()
					for b.Loop() {
						if !compare() {
							b.Fatal("unequal allocations")
						}
					}
				})
			}
		}
	}
}
