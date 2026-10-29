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
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apiresource "k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/dynamic-resource-allocation/cel/sharingaffinity"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	schedmetrics "k8s.io/kubernetes/pkg/scheduler/metrics"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/utils/ptr"
)

type staticSliceLister struct {
	slices []*resourceapi.ResourceSlice
	err    error
}

func toInternalResourceSlices(slices []*resourceapi.ResourceSlice) []*draapi.ResourceSlice {
	out := make([]*draapi.ResourceSlice, len(slices))
	for i, s := range slices {
		if s == nil {
			continue
		}
		var internalSlice draapi.ResourceSlice
		if err := draapi.Convert_v1_ResourceSlice_To_api_ResourceSlice(s, &internalSlice, nil); err != nil {
			panic(err)
		}
		out[i] = &internalSlice
	}
	return out
}

func (l *staticSliceLister) ListWithDeviceTaintRules() ([]*draapi.ResourceSlice, error) {
	if l.err != nil {
		return nil, l.err
	}
	return toInternalResourceSlices(l.slices), nil
}

func makeSharingAffinityPoolSlices(nodeName, driverName, poolName string, generation int64, complete bool, extractors []resourceapi.SharingAffinityExtractor) []*resourceapi.ResourceSlice {
	sliceCount := int64(1)
	if len(extractors) > 0 {
		sliceCount = 2
	}
	if !complete {
		sliceCount++
	}
	deviceSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: poolName + "-devices",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				Generation:         generation,
				ResourceSliceCount: sliceCount,
			},
			Devices: []resourceapi.Device{
				{
					Name: "instance-1",
					Capacity: map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
						"shares": {
							Value: apiresource.MustParse("4"),
							RequestPolicy: &resourceapi.CapacityRequestPolicy{
								Default: ptr.To(apiresource.MustParse("1")),
								ValidRange: &resourceapi.CapacityRequestPolicyRange{
									Min: ptr.To(apiresource.MustParse("1")),
									Max: ptr.To(apiresource.MustParse("4")),
								},
							},
						},
					},
					AllowMultipleAllocations: ptr.To(true),
				},
			},
		},
	}
	if len(extractors) == 0 {
		return []*resourceapi.ResourceSlice{deviceSlice}
	}
	metaSlice := &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name: poolName + "-meta",
		},
		Spec: resourceapi.ResourceSliceSpec{
			Driver:   driverName,
			NodeName: &nodeName,
			Pool: resourceapi.ResourcePool{
				Name:               poolName,
				Generation:         generation,
				ResourceSliceCount: sliceCount,
			},
			SharingAffinity: extractors,
		},
	}
	return []*resourceapi.ResourceSlice{deviceSlice, metaSlice}
}

func makeSharedAllocatedClaim(name, uid, rv, driverName, poolName, deviceName, requestName, shareID, rawJSON string, adminAccess *bool) *resourceapi.ResourceClaim {
	var shareUID *types.UID
	if shareID != "" {
		shareUID = ptr.To(types.UID(shareID))
	}
	var specConfigs []resourceapi.DeviceClaimConfiguration
	var configs []resourceapi.DeviceAllocationConfiguration
	if rawJSON != "" {
		specConfigs = []resourceapi.DeviceClaimConfiguration{
			{
				Requests: []string{requestName},
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver:     driverName,
						Parameters: apiruntime.RawExtension{Raw: []byte(rawJSON)},
					},
				},
			},
		}
		configs = []resourceapi.DeviceAllocationConfiguration{
			{
				Source:   resourceapi.AllocationConfigSourceClaim,
				Requests: []string{requestName},
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver:     driverName,
						Parameters: apiruntime.RawExtension{Raw: []byte(rawJSON)},
					},
				},
			},
		}
	}
	return &resourceapi.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       namespace,
			UID:             types.UID(uid),
			ResourceVersion: rv,
		},
		Spec: resourceapi.ResourceClaimSpec{
			Devices: resourceapi.DeviceClaim{
				Config: specConfigs,
			},
		},
		Status: resourceapi.ResourceClaimStatus{
			Allocation: &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{
					Config: configs,
					Results: []resourceapi.DeviceRequestAllocationResult{
						{
							Request:     requestName,
							Driver:      driverName,
							Pool:        poolName,
							Device:      deviceName,
							ShareID:     shareUID,
							AdminAccess: adminAccess,
							ConsumedCapacity: map[resourceapi.QualifiedName]apiresource.Quantity{
								"shares": apiresource.MustParse("1"),
							},
						},
					},
				},
			},
		},
	}
}

func newTestClaimTracker(slices []*resourceapi.ResourceSlice, claims ...*resourceapi.ResourceClaim) (*claimTracker, *staticSliceLister) {
	logger := klog.Background()
	allocated := newAllocatedDevices(logger)
	for _, claim := range claims {
		allocated.onAdd(claim)
	}
	lister := &staticSliceLister{slices: slices}
	tracker := &claimTracker{
		inFlightAllocations: make(map[types.UID]inFlightAllocation),
		allocatedDevices:    allocated,
		logger:              logger,
	}
	tracker.enableSharingAffinity(context.Background(), lister, nil)
	return tracker, lister
}

func TestSharingAffinityGatherAllocatedState(t *testing.T) {
	tenantExtractor := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "tenant",
			APIGroup:   "*",
			Kind:       "*",
			Expression: `parameters.tenant`,
		},
	}
	brokenExtractor := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "tenant",
			APIGroup:   "*",
			Kind:       "*",
			Expression: `parameters.nonExistentMethod()`,
		},
	}
	devID := structured.MakeDeviceID(driver, nodeName, "instance-1")
	poolID := schedulerapi.PoolID{Driver: driver, Pool: nodeName}

	testCases := map[string]struct {
		slices          []*resourceapi.ResourceSlice
		claims          []*resourceapi.ResourceClaim
		wantState       structured.AffinityStateCollection
		wantStalePools  []schedulerapi.PoolID
		wantCacheLength int
	}{
		"single shared claim locks device": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status:         structured.AffinityStatusLocked,
					LockedAffinity: map[string]string{"tenant": "alpha"},
				},
			},
			wantCacheLength: 1,
		},
		"compatible shared claims keep device locked": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
				makeSharedAllocatedClaim("claim-2", "uid-2", "1", driver, nodeName, "instance-1", "req-1", "share-2", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status:         structured.AffinityStatusLocked,
					LockedAffinity: map[string]string{"tenant": "alpha"},
				},
			},
			wantCacheLength: 2,
		},
		"non-shared and admin allocations do not lock device": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-exclusive", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "", `{"tenant":"alpha"}`, nil),
				makeSharedAllocatedClaim("claim-admin", "uid-2", "1", driver, nodeName, "instance-1", "req-1", "share-2", `{"tenant":"beta"}`, ptr.To(true)),
			},
			wantState:       structured.AffinityStateCollection{},
			wantCacheLength: 0,
		},
		"unconstrained pool does not lock device": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, nil),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			},
			wantState:       structured.AffinityStateCollection{},
			wantCacheLength: 0,
		},
		"incomplete pool marks stale pool and device unreconstructable": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, false, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status: structured.AffinityStatusUnreconstructable,
				},
			},
			wantStalePools:  []schedulerapi.PoolID{poolID},
			wantCacheLength: 0,
		},
		"invalid pool CEL expression marks device unreconstructable": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, brokenExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status: structured.AffinityStatusUnreconstructable,
				},
			},
			wantCacheLength: 0,
		},
		"missing required extractor key marks device unreconstructable": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status: structured.AffinityStatusUnreconstructable,
				},
			},
			wantCacheLength: 1,
		},
		"conflicting shared claims mark device unreconstructable": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
				makeSharedAllocatedClaim("claim-2", "uid-2", "1", driver, nodeName, "instance-1", "req-1", "share-2", `{"tenant":"beta"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status: structured.AffinityStatusUnreconstructable,
				},
			},
			wantCacheLength: 2,
		},
		"duplicate extractor names in pool mark device unreconstructable": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, []resourceapi.SharingAffinityExtractor{tenantExtractor[0], tenantExtractor[0]}),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status: structured.AffinityStatusUnreconstructable,
				},
			},
			wantCacheLength: 0,
		},
		"subrequest allocation result resolves parent and subrequest config": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, tenantExtractor),
			claims: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-sub", "uid-sub", "1", driver, nodeName, "instance-1", "req-1/sub-1", "share-sub", `{"tenant":"alpha"}`, nil),
			},
			wantState: structured.AffinityStateCollection{
				devID: {
					Status:         structured.AffinityStatusLocked,
					LockedAffinity: map[string]string{"tenant": "alpha"},
				},
			},
			wantCacheLength: 1,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tracker, _ := newTestClaimTracker(tc.slices, tc.claims...)
			state, err := tracker.GatherAllocatedState()
			require.NoError(t, err)
			if diff := cmp.Diff(tc.wantState, state.DeviceSharingAffinity.DeviceStates); diff != "" {
				t.Fatalf("unexpected DeviceStates (-want +got):\n%s", diff)
			}
			for _, pool := range tc.wantStalePools {
				require.True(t, state.DeviceSharingAffinity.StalePools.Has(pool), "expected StalePools to contain %v", pool)
			}
			require.Len(t, tracker.sharingAffinity.claimExtractionCache, tc.wantCacheLength)
		})
	}
}

func TestSharingAffinityStickyAmbiguityAndDrain(t *testing.T) {
	extractors := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "tenant",
			APIGroup:   "*",
			Kind:       "*",
			Expression: `parameters.tenant`,
		},
	}
	slices := makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, extractors)
	claimA := makeSharedAllocatedClaim("claim-a", "uid-a", "1", driver, nodeName, "instance-1", "req-1", "share-a", `{"tenant":"alpha"}`, nil)
	claimB := makeSharedAllocatedClaim("claim-b", "uid-b", "1", driver, nodeName, "instance-1", "req-1", "share-b", `{"tenant":"beta"}`, nil)
	devID := structured.MakeDeviceID(driver, nodeName, "instance-1")

	tracker, _ := newTestClaimTracker(slices, claimA)

	// Step 1: claimA establishes a clean lock for tenant=alpha.
	state, err := tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusLocked, state.DeviceSharingAffinity.DeviceStates[devID].Status)

	// Step 2: conflicting claimB arrives -> device transitions to Unreconstructable.
	tracker.allocatedDevices.onAdd(claimB)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusUnreconstructable, state.DeviceSharingAffinity.DeviceStates[devID].Status)

	// Step 3: removing claimA while claimB remains active must keep the device Unreconstructable
	// (sticky ambiguity latch prevents silent flip from alpha to beta without full drain).
	tracker.allocatedDevices.onDelete(claimA)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusUnreconstructable, state.DeviceSharingAffinity.DeviceStates[devID].Status)

	// Step 4: once claimB also drains, the device lock baseline and extraction cache are cleared.
	tracker.allocatedDevices.onDelete(claimB)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Empty(t, state.DeviceSharingAffinity.DeviceStates)
	require.Empty(t, tracker.sharingAffinity.establishedDeviceLocks)
	require.Empty(t, tracker.sharingAffinity.claimExtractionCache)

	// Step 5: after full drain, claimB can cleanly lock the device to tenant=beta.
	tracker.allocatedDevices.onAdd(claimB)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.DeviceAffinityState{
		Status:         structured.AffinityStatusLocked,
		LockedAffinity: map[string]string{"tenant": "beta"},
	}, state.DeviceSharingAffinity.DeviceStates[devID])

	// Step 6: if an active claim's config/extracted value changes in place before draining,
	// the device latches into Unreconstructable until deallocated via onUpdate.
	claimBMutated := makeSharedAllocatedClaim("claim-b", "uid-b", "2", driver, nodeName, "instance-1", "req-1", "share-b", `{"tenant":"gamma"}`, nil)
	tracker.allocatedDevices.onUpdate(claimB, claimBMutated)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusUnreconstructable, state.DeviceSharingAffinity.DeviceStates[devID].Status)

	claimBDeallocated := claimBMutated.DeepCopy()
	claimBDeallocated.Status.Allocation = nil
	tracker.allocatedDevices.onUpdate(claimBMutated, claimBDeallocated)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Empty(t, state.DeviceSharingAffinity.DeviceStates)

	// Step 7: if a device has an established lock and the pool's extractors are removed while the claim is still active,
	// the device becomes Unreconstructable until drained.
	tracker.allocatedDevices.onAdd(claimA)
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusLocked, state.DeviceSharingAffinity.DeviceStates[devID].Status)

	tracker.sharingAffinity.resourceSliceLister = &staticSliceLister{slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 2, true, nil)}
	state, err = tracker.GatherAllocatedState()
	require.NoError(t, err)
	require.Equal(t, structured.AffinityStatusUnreconstructable, state.DeviceSharingAffinity.DeviceStates[devID].Status)
}

func TestSharingAffinityReserveClaimWithAffinity(t *testing.T) {
	extractors := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "tenant",
			APIGroup:   "*",
			Kind:       "*",
			Expression: `parameters.tenant`,
		},
	}
	slices := makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, extractors)
	viewGen1 := structured.ComputePoolMetadataView(toInternalResourceSlices(slices), driver, nodeName)
	require.True(t, viewGen1.Complete)

	makeClaimAffinity := func(devName, tenant string, generation int64, digest string) *structured.ClaimAffinity {
		devID := structured.MakeDeviceID(driver, nodeName, devName)
		var values map[string]string
		if tenant != "" {
			values = map[string]string{"tenant": tenant}
		}
		return &structured.ClaimAffinity{
			Devices: []structured.DeviceAffinity{
				{
					Device:  devID,
					Request: "req-1",
					Pool: sharingaffinity.PoolKey{
						Driver:     driver,
						Name:       nodeName,
						Generation: generation,
					},
					Constrained: true,
					Digest:      digest,
					Values:      values,
				},
			},
		}
	}

	testCases := map[string]struct {
		slices       []*resourceapi.ResourceSlice
		existing     []*resourceapi.ResourceClaim
		claim        *resourceapi.ResourceClaim
		affinity     *structured.ClaimAffinity
		wantErrIs    error
		wantInFlight bool
	}{
		"compatible reservation succeeds": {
			slices:       slices,
			claim:        makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:     makeClaimAffinity("instance-1", "alpha", 1, viewGen1.Digest),
			wantInFlight: true,
		},
		"stale pool generation fails with errSharingAffinityStale": {
			slices:    slices,
			claim:     makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:  makeClaimAffinity("instance-1", "alpha", 2, viewGen1.Digest),
			wantErrIs: errSharingAffinityStale,
		},
		"stale extractor digest fails with errSharingAffinityStale": {
			slices:    slices,
			claim:     makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:  makeClaimAffinity("instance-1", "alpha", 1, "stale-digest"),
			wantErrIs: errSharingAffinityStale,
		},
		"incomplete pool at reserve fails with errSharingAffinityStale": {
			slices:    makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, false, extractors),
			claim:     makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:  makeClaimAffinity("instance-1", "alpha", 1, viewGen1.Digest),
			wantErrIs: errSharingAffinityStale,
		},
		"conflicting existing lock fails with errSharingAffinityConflict": {
			slices: slices,
			existing: []*resourceapi.ResourceClaim{
				makeSharedAllocatedClaim("claim-existing", "uid-existing", "1", driver, nodeName, "instance-1", "req-1", "share-existing", `{"tenant":"beta"}`, nil),
			},
			claim:     makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:  makeClaimAffinity("instance-1", "alpha", 1, viewGen1.Digest),
			wantErrIs: errSharingAffinityConflict,
		},
		"unconstrained pool succeeds with matching unconstrained provenance": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, nil),
			claim:  makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity: &structured.ClaimAffinity{
				Devices: []structured.DeviceAffinity{
					{
						Device:  structured.MakeDeviceID(driver, nodeName, "instance-1"),
						Request: "req-1",
						Pool: sharingaffinity.PoolKey{
							Driver:     driver,
							Name:       nodeName,
							Generation: 1,
						},
						Constrained: false,
					},
				},
			},
			wantInFlight: true,
		},
		"unconstrained pool fails when provenance was constrained": {
			slices:    makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, nil),
			claim:     makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity:  makeClaimAffinity("instance-1", "alpha", 1, viewGen1.Digest),
			wantErrIs: errSharingAffinityStale,
		},
		"unconstrained pool fails when provenance generation is stale": {
			slices: makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, nil),
			claim:  makeSharedAllocatedClaim("claim-1", "uid-1", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil),
			affinity: &structured.ClaimAffinity{
				Devices: []structured.DeviceAffinity{
					{
						Device:  structured.MakeDeviceID(driver, nodeName, "instance-1"),
						Request: "req-1",
						Pool: sharingaffinity.PoolKey{
							Driver:     driver,
							Name:       nodeName,
							Generation: 2,
						},
						Constrained: false,
					},
				},
			},
			wantErrIs: errSharingAffinityStale,
		},
		"self-conflicting requests on same device in one claim fail with errSharingAffinityConflict": {
			slices: slices,
			claim: func() *resourceapi.ResourceClaim {
				c := makeSharedAllocatedClaim("claim-self-conflict", "uid-self", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil)
				r2 := c.Status.Allocation.Devices.Results[0].DeepCopy()
				r2.Request = "req-2"
				r2.ShareID = ptr.To(types.UID("share-2"))
				c.Status.Allocation.Devices.Results = append(c.Status.Allocation.Devices.Results, *r2)
				return c
			}(),
			affinity: &structured.ClaimAffinity{
				Devices: []structured.DeviceAffinity{
					{
						Device:      structured.MakeDeviceID(driver, nodeName, "instance-1"),
						Request:     "req-1",
						Pool:        sharingaffinity.PoolKey{Driver: driver, Name: nodeName, Generation: 1},
						Constrained: true,
						Digest:      viewGen1.Digest,
						Values:      map[string]string{"tenant": "alpha"},
					},
					{
						Device:      structured.MakeDeviceID(driver, nodeName, "instance-1"),
						Request:     "req-2",
						Pool:        sharingaffinity.PoolKey{Driver: driver, Name: nodeName, Generation: 1},
						Constrained: true,
						Digest:      viewGen1.Digest,
						Values:      map[string]string{"tenant": "beta"},
					},
				},
			},
			wantErrIs: errSharingAffinityConflict,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			tracker, _ := newTestClaimTracker(tc.slices, tc.existing...)
			err := tracker.ReserveClaimWithAffinity(tc.claim.UID, tc.claim, tc.affinity)
			if tc.wantErrIs != nil {
				require.ErrorIs(t, err, tc.wantErrIs)
				require.Nil(t, tracker.GetPendingAllocation(tc.claim.UID))
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantInFlight, tracker.GetPendingAllocation(tc.claim.UID) != nil)
			if tc.wantInFlight {
				// Re-reserving the same in-flight claim (e.g. second pod in a PodGroup) reuses existing affinity and increments sharers.
				require.NoError(t, tracker.ReserveClaimWithAffinity(tc.claim.UID, tc.claim, nil))
				state, err := tracker.GatherAllocatedState()
				require.NoError(t, err)
				require.NotNil(t, state.DeviceSharingAffinity)
				require.False(t, tracker.MaybeRemoveClaimPendingAllocation(tc.claim.UID, false))
				require.True(t, tracker.MaybeRemoveClaimPendingAllocation(tc.claim.UID, false))
				require.Empty(t, tracker.sharingAffinity.establishedDeviceLocks)
			}
		})
	}

	t.Run("input validation guards", func(t *testing.T) {
		claim := makeSharedAllocatedClaim("claim-guard", "uid-guard", "1", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil)
		tracker, _ := newTestClaimTracker(slices)

		// Nil claim or mismatched UID -> errSharingAffinityStale.
		require.ErrorIs(t, tracker.ReserveClaimWithAffinity(claim.UID, nil, nil), errSharingAffinityStale)
		require.ErrorIs(t, tracker.ReserveClaimWithAffinity("wrong-uid", claim, nil), errSharingAffinityStale)
	})
}

func TestSharingAffinityPluginFilterAndReserve(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity: true,
		features.DRASharingAffinity:    true,
	})
	fts := feature.Features{
		EnableDRAConsumableCapacity: true,
		EnableDRASharingAffinity:    true,
	}

	extractors := []resourceapi.SharingAffinityExtractor{
		{
			Name:       "tenant",
			APIGroup:   "*",
			Kind:       "*",
			Expression: `parameters.tenant`,
		},
	}
	slices := makeSharingAffinityPoolSlices(nodeName, driver, nodeName, 1, true, extractors)
	var sliceObjs []apiruntime.Object
	for _, s := range slices {
		sliceObjs = append(sliceObjs, s)
	}
	lockedClaim := makeSharedAllocatedClaim("locked-claim", "", "", driver, nodeName, "instance-1", "req-1", "share-locked", `{"tenant":"alpha"}`, nil)

	makePendingSharedClaim := func(name string, rawJSONs ...string) *resourceapi.ResourceClaim {
		c := st.MakeResourceClaim().
			Name(name).
			Namespace(namespace).
			RequestWithName("req-1", className).
			Obj()
		c.Spec.Devices.Requests[0].Exactly.Capacity = &resourceapi.CapacityRequirements{
			Requests: map[resourceapi.QualifiedName]apiresource.Quantity{
				"shares": apiresource.MustParse("1"),
			},
		}
		for _, raw := range rawJSONs {
			c.Spec.Devices.Config = append(c.Spec.Devices.Config, resourceapi.DeviceClaimConfiguration{
				Requests: []string{"req-1"},
				DeviceConfiguration: resourceapi.DeviceConfiguration{
					Opaque: &resourceapi.OpaqueDeviceConfiguration{
						Driver:     driver,
						Parameters: apiruntime.RawExtension{Raw: []byte(raw)},
					},
				},
			})
		}
		return c
	}

	t.Run("compatible tenant passes Filter and Reserve then Unreserve cleans up", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		claim := makePendingSharedClaim(claimName, `{"tenant":"alpha"}`)
		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claim, lockedClaim.DeepCopy()}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, podWithClaimName, nil)
		require.True(t, status.IsSuccess(), "PreFilter failed: %v", status)

		status = testCtx.p.Filter(subCtx, testCtx.state, podWithClaimName, testCtx.nodeInfos[0])
		require.True(t, status.IsSuccess(), "Filter failed: %v", status)

		status = testCtx.p.Reserve(subCtx, testCtx.state, podWithClaimName, nodeName)
		require.True(t, status.IsSuccess(), "Reserve failed: %v", status)
		require.NotNil(t, testCtx.draManager.ResourceClaims().GetPendingAllocation(types.UID("UID-0")))

		testCtx.p.Unreserve(subCtx, testCtx.state, podWithClaimName, nodeName)
		require.Nil(t, testCtx.draManager.ResourceClaims().GetPendingAllocation(types.UID("UID-0")))
	})

	t.Run("conflicting tenant is rejected in Filter with UnschedulableAndUnresolvable", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		claim := makePendingSharedClaim(claimName, `{"tenant":"beta"}`)
		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claim, lockedClaim.DeepCopy()}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, podWithClaimName, nil)
		require.True(t, status.IsSuccess(), "PreFilter failed: %v", status)

		status = testCtx.p.Filter(subCtx, testCtx.state, podWithClaimName, testCtx.nodeInfos[0])
		require.Equal(t, fwk.UnschedulableAndUnresolvable, status.Code())
	})

	t.Run("self-inconsistent claim configs fail Filter with UnschedulableAndUnresolvable", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		claim := makePendingSharedClaim(claimName, `{"tenant":"alpha"}`, `{"tenant":"beta"}`)
		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claim}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, podWithClaimName, nil)
		require.True(t, status.IsSuccess(), "PreFilter failed: %v", status)

		status = testCtx.p.Filter(subCtx, testCtx.state, podWithClaimName, testCtx.nodeInfos[0])
		require.Equal(t, fwk.UnschedulableAndUnresolvable, status.Code())
	})

	t.Run("concurrent lock conflict before Reserve returns Unschedulable", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		claim := makePendingSharedClaim(claimName, `{"tenant":"alpha"}`)
		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claim}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, podWithClaimName, nil)
		require.True(t, status.IsSuccess(), "PreFilter failed: %v", status)

		status = testCtx.p.Filter(subCtx, testCtx.state, podWithClaimName, testCtx.nodeInfos[0])
		require.True(t, status.IsSuccess(), "Filter failed: %v", status)

		// Simulate a conflicting shared claim being allocated before Reserve runs.
		conflictingClaim := makeSharedAllocatedClaim("conflicting-claim", "uid-conflict", "1", driver, nodeName, "instance-1", "req-1", "share-conflict", `{"tenant":"beta"}`, nil)
		testCtx.draManager.resourceClaimTracker.allocatedDevices.onAdd(conflictingClaim)

		status = testCtx.p.Reserve(subCtx, testCtx.state, podWithClaimName, nodeName)
		require.Equal(t, fwk.Unschedulable, status.Code())
		require.ErrorIs(t, status.AsError(), errSharingAffinityConflict)
		require.Nil(t, testCtx.draManager.ResourceClaims().GetPendingAllocation(types.UID("UID-0")))
	})

	t.Run("multi-claim Reserve rolls back earlier claim when later claim conflicts", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		claim0 := makePendingSharedClaim(claimName, `{"tenant":"alpha"}`)
		claim1 := makePendingSharedClaim(claimName+"-2", `{"tenant":"alpha"}`)
		twoClaimPod := st.MakePod().Name(podName).Namespace(namespace).
			UID(podUID).
			PodResourceClaims(
				v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claim0.Name},
				v1.PodResourceClaim{Name: "my-claim-2", ResourceClaimName: &claim1.Name},
			).
			Obj()
		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claim0, claim1}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, twoClaimPod, nil)
		require.True(t, status.IsSuccess(), "PreFilter failed: %v", status)

		status = testCtx.p.Filter(subCtx, testCtx.state, twoClaimPod, testCtx.nodeInfos[0])
		require.True(t, status.IsSuccess(), "Filter failed: %v", status)

		// Force claim1's provenance digest to be stale at Reserve time so claim0 succeeds and claim1 fails,
		// exercising Reserve's rollback of claim0's pending allocation.
		sd, err := getStateData(testCtx.state)
		require.NoError(t, err)
		sd.nodeAllocations[nodeName].affinityByClaim[1].Devices[0].Digest = "stale-digest"

		status = testCtx.p.Reserve(subCtx, testCtx.state, twoClaimPod, nodeName)
		require.Equal(t, fwk.Unschedulable, status.Code())
		require.ErrorIs(t, status.AsError(), errSharingAffinityStale)
		require.Nil(t, testCtx.draManager.ResourceClaims().GetPendingAllocation(types.UID("UID-0")))
		require.Nil(t, testCtx.draManager.ResourceClaims().GetPendingAllocation(types.UID("UID-1")))

		// Also verify assumeCache validation guards (stale ResourceVersion and already-allocated claim) against the real assumeCache.
		tracker := testCtx.draManager.resourceClaimTracker
		allocatedClaim := makeSharedAllocatedClaim(claimName, "UID-0", "99", driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil)
		require.ErrorIs(t, tracker.ReserveClaimWithAffinity("UID-0", allocatedClaim, nil), errSharingAffinityStale)

		allocatedClaim.ResourceVersion = "1"
		require.NoError(t, tracker.AssumeClaimAfterAPICall(allocatedClaim))
		require.NoError(t, tracker.ReserveClaimWithAffinity("UID-0", allocatedClaim, nil))

		conflictingAlloc := allocatedClaim.DeepCopy()
		conflictingAlloc.Status.Allocation.Devices.Results[0].Device = "instance-2"
		require.ErrorIs(t, tracker.ReserveClaimWithAffinity("UID-0", conflictingAlloc, nil), errSharingAffinityConflict)

		// Disabled feature gate delegates directly to SignalClaimPendingAllocation.
		tracker.sharingAffinity = nil
		allocatedClaim1 := makeSharedAllocatedClaim(claim1.Name, "UID-1", claim1.ResourceVersion, driver, nodeName, "instance-1", "req-1", "share-1", `{"tenant":"alpha"}`, nil)
		require.NoError(t, tracker.ReserveClaimWithAffinity("UID-1", allocatedClaim1, nil))
		require.NotNil(t, tracker.GetPendingAllocation("UID-1"))
	})

	t.Run("sharing affinity PRR metrics", func(t *testing.T) {
		subCtx := ktesting.Init(t)
		schedmetrics.InitMetrics()
		_ = setupMetrics(fts)

		// 1. First pod with tenant=alpha locks instance-1.
		claimAlpha1 := makePendingSharedClaim("claim-alpha-1", `{"tenant":"alpha"}`)
		podAlpha1 := st.MakePod().Name("pod-alpha-1").Namespace(namespace).
			UID("pod-alpha-1-uid").
			PodResourceClaims(v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claimAlpha1.Name}).
			Obj()

		// 2. Second pod with tenant=alpha reuses already-locked instance-1.
		claimAlpha2 := makePendingSharedClaim("claim-alpha-2", `{"tenant":"alpha"}`)
		podAlpha2 := st.MakePod().Name("pod-alpha-2").Namespace(namespace).
			UID("pod-alpha-2-uid").
			PodResourceClaims(v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claimAlpha2.Name}).
			Obj()

		// 3. Third pod with tenant=beta conflicts with locked instance-1.
		claimBeta := makePendingSharedClaim("claim-beta", `{"tenant":"beta"}`)
		podBeta := st.MakePod().Name("pod-beta").Namespace(namespace).
			UID("pod-beta-uid").
			PodResourceClaims(v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claimBeta.Name}).
			Obj()

		// 4. Fourth pod without matching config triggers no_keys_extracted.
		claimNoKey := makePendingSharedClaim("claim-nokey", `{}`)
		claimNoKey.Spec.Devices.Config = nil
		podNoKey := st.MakePod().Name("pod-nokey").Namespace(namespace).
			UID("pod-nokey-uid").
			PodResourceClaims(v1.PodResourceClaim{Name: "my-claim", ResourceClaimName: &claimNoKey.Name}).
			Obj()

		testCtx := setup(subCtx, &config.DynamicResourcesArgs{}, []*v1.Node{workerNode}, []*resourceapi.ResourceClaim{claimAlpha1, claimAlpha2, claimBeta, claimNoKey}, []*resourceapi.DeviceClass{deviceClass}, nil, sliceObjs, fts, false, nil)

		// Schedule podAlpha1 -> locks instance-1.
		_, status := testCtx.p.PreFilter(subCtx, testCtx.state, podAlpha1, nil)
		require.True(t, status.IsSuccess())
		status = testCtx.p.Filter(subCtx, testCtx.state, podAlpha1, testCtx.nodeInfos[0])
		require.True(t, status.IsSuccess())
		status = testCtx.p.Reserve(subCtx, testCtx.state, podAlpha1, nodeName)
		require.True(t, status.IsSuccess())

		lockedVal, err := testutil.GetGaugeMetricValue(schedmetrics.SharingAffinityLockedDevices)
		require.NoError(t, err)
		require.Equal(t, float64(1), lockedVal)

		evalsVal, err := testutil.GetCounterMetricValue(schedmetrics.SharingAffinityExtractorEvaluationsTotal)
		require.NoError(t, err)
		require.GreaterOrEqual(t, evalsVal, float64(1))

		// Schedule podAlpha2 -> compatible reuse of locked instance-1.
		state2 := framework.NewCycleState()
		_, status = testCtx.p.PreFilter(subCtx, state2, podAlpha2, nil)
		require.True(t, status.IsSuccess())
		status = testCtx.p.Filter(subCtx, state2, podAlpha2, testCtx.nodeInfos[0])
		require.True(t, status.IsSuccess())
		status = testCtx.p.Reserve(subCtx, state2, podAlpha2, nodeName)
		require.True(t, status.IsSuccess())

		reuseVal, err := testutil.GetCounterMetricValue(schedmetrics.SharingAffinityCompatibleReuseTotal)
		require.NoError(t, err)
		require.Equal(t, float64(1), reuseVal)

		// Filter podBeta -> lock conflict on instance-1.
		state3 := framework.NewCycleState()
		_, status = testCtx.p.PreFilter(subCtx, state3, podBeta, nil)
		require.True(t, status.IsSuccess())
		status = testCtx.p.Filter(subCtx, state3, podBeta, testCtx.nodeInfos[0])
		require.Equal(t, fwk.UnschedulableAndUnresolvable, status.Code())

		conflictVal, err := testutil.GetCounterMetricValue(schedmetrics.SharingAffinityLockConflictTotal)
		require.NoError(t, err)
		require.Equal(t, float64(1), conflictVal)

		// Filter podNoKey -> no keys extracted on instance-1.
		state4 := framework.NewCycleState()
		_, status = testCtx.p.PreFilter(subCtx, state4, podNoKey, nil)
		require.True(t, status.IsSuccess())
		status = testCtx.p.Filter(subCtx, state4, podNoKey, testCtx.nodeInfos[0])
		require.Equal(t, fwk.UnschedulableAndUnresolvable, status.Code())

		noKeysVal, err := testutil.GetCounterMetricValue(schedmetrics.SharingAffinityNoKeysExtractedTotal)
		require.NoError(t, err)
		require.Equal(t, float64(1), noKeysVal)

		// Add a conflicting owner on instance-2 to quarantine instance-2 as unreconstructable.
		testCtx.draManager.resourceClaimTracker.allocatedDevices.onAdd(
			makeSharedAllocatedClaim("unrec-1", "uid-unrec-1", "1", driver, nodeName, "instance-2", "req-1", "share-u1", `{"tenant":"alpha"}`, nil),
		)
		testCtx.draManager.resourceClaimTracker.allocatedDevices.onAdd(
			makeSharedAllocatedClaim("unrec-2", "uid-unrec-2", "1", driver, nodeName, "instance-2", "req-1", "share-u2", `{"tenant":"beta"}`, nil),
		)

		unrecVal, err := testutil.GetGaugeMetricValue(schedmetrics.SharingAffinityUnreconstructableDevices)
		require.NoError(t, err)
		require.Equal(t, float64(1), unrecVal)
	})
}
