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

package experimental

import (
	"context"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/internal/allocatortesting"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
)

var (
	_ *internal.ClaimAffinity               = (*ClaimAffinity)(nil)
	_ *internal.DeviceAffinity              = (*DeviceAffinity)(nil)
	_ *internal.PoolMetadataView            = (*PoolMetadataView)(nil)
	_ *schedulerapi.AffinityStatus          = (*internal.AffinityStatus)(nil)
	_ *schedulerapi.DeviceAffinityState     = (*internal.DeviceAffinityState)(nil)
	_ *schedulerapi.AffinityStateCollection = (*internal.AffinityStateCollection)(nil)
	_ *schedulerapi.SharingAffinityState    = (*internal.SharingAffinityState)(nil)
	_ *schedulerapi.AllocatedState          = (*internal.AllocatedState)(nil)
)

func TestSharingAffinityStatusAliases(t *testing.T) {
	for _, statuses := range [][2]schedulerapi.AffinityStatus{
		{internal.AffinityStatusClean, schedulerapi.AffinityStatusClean},
		{internal.AffinityStatusLocked, schedulerapi.AffinityStatusLocked},
		{internal.AffinityStatusUnreconstructable, schedulerapi.AffinityStatusUnreconstructable},
	} {
		if statuses[0] != statuses[1] {
			t.Errorf("status alias = %d, want %d", statuses[0], statuses[1])
		}
	}
}

func TestAllocator(t *testing.T) {
	newAllocator := func(
		ctx context.Context,
		features Features,
		allocatedState AllocatedState,
		classLister DeviceClassLister,
		slices []*resourceapi.ResourceSlice,
		celCache *cel.Cache,
	) (internal.Allocator, error) {
		return NewAllocator(ctx, features, allocatedState, classLister, slices, celCache)
	}
	allocatortesting.TestAllocator(t,
		SupportedFeatures,
		newAllocator,
	)
	allocatortesting.TestLexicographicalAllocator(t,
		SupportedFeatures,
		newAllocator,
	)
}
