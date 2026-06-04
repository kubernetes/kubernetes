/*
Copyright 2024 The Kubernetes Authors.

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

package structured

import (
	"context"
	"fmt"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/internal/allocatortesting"
)

func TestAllocator(t *testing.T) {
	allocatortesting.TestAllocator(t,
		internal.FeaturesAll,
		func(
			ctx context.Context,
			features Features,
			allocatedState AllocatedState,
			classLister DeviceClassLister,
			slices []*resourceapi.ResourceSlice,
			celCache *cel.Cache,
		) (allocatortesting.Allocator, error) {
			allocator, err := NewAllocator(ctx, features, allocatedState, classLister, slices, celCache)
			if err != nil {
				return nil, err
			}
			// Return the allocator with the internal interface so the tests can
			// check the channel for the allocator.
			internalAllocator, ok := allocator.(internal.Allocator)
			if !ok {
				return nil, fmt.Errorf("allocator doesn't implement internal interface")
			}
			return internalAllocator, nil
		})
}
