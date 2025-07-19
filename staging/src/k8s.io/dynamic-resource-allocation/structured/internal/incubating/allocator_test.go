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

package incubating

import (
	"context"
	"testing"

	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/internal/allocatortesting"
)

func TestAllocator(t *testing.T) {
	allocatortesting.TestAllocator(t,
		SupportedFeatures,
		func(
			ctx context.Context,
			features Features,
			allocatedDevices sets.Set[DeviceID],
			classLister DeviceClassLister,
			slices []*resourceapi.ResourceSlice,
			celCache *cel.Cache,
		) (internal.Allocator, error) {
			return NewAllocator(ctx, features, allocatedDevices, classLister, slices, celCache)
		},
	)
}
