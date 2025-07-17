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

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1beta1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/dynamic-resource-allocation/structured/internal"
	"k8s.io/dynamic-resource-allocation/structured/internal/incubating"
	"k8s.io/dynamic-resource-allocation/structured/internal/stable"
)

// To keep the code in different packages simple, type aliases are used everywhere.
// Functions are wrappers instead of variables to enable compiler optimization.
// The Allocator interface is defined twice intentionally: that way, the docs
// for this package are more useful.

type DeviceClassLister = internal.DeviceClassLister
type Features = internal.Features
type DeviceID = internal.DeviceID

func MakeDeviceID(driver, pool, device string) DeviceID {
	return internal.MakeDeviceID(driver, pool, device)
}

// Allocator calculates how to allocate a set of unallocated claims which use
// structured parameters.
//
// It needs as input the node where the allocated claims are meant to be
// available and the current state of the cluster (claims, classes, resource
// slices).
type Allocator interface {
	// Allocate calculates the allocation(s) for one particular node.
	//
	// It returns an error only if some fatal problem occurred. These are errors
	// caused by invalid input data, like for example errors in CEL selectors, so a
	// scheduler should abort and report that problem instead of trying to find
	// other nodes where the error doesn't occur.
	//
	// In the future, special errors will be defined which enable the caller to
	// identify which object (like claim or class) caused the problem. This will
	// enable reporting the problem as event for those objects.
	//
	// If the claims cannot be allocated, it returns nil. This includes the
	// situation where the resource slices are incomplete at the moment.
	//
	// If the claims can be allocated, then it prepares one allocation result for
	// each unallocated claim. It is the responsibility of the caller to persist
	// those allocations, if desired.
	//
	// Allocate is thread-safe. If the caller wants to get the node name included
	// in log output, it can use contextual logging and add the node as an
	// additional value. A name can also be useful because log messages do not
	// have a common prefix. V(5) is used for one-time log entries, V(6) for important
	// progress reports, and V(7) for detailed debug output.
	//
	//
	// Context cancellation is supported. An error wrapping the context's error will
	// be returned in case of cancellation.
	Allocate(ctx context.Context, node *v1.Node, claims []*resourceapi.ResourceClaim) (finalResult []resourceapi.AllocationResult, finalErr error)
}

// NewAllocator returns an allocator for a certain set of claims or an error if
// some problem was detected which makes it impossible to allocate claims.
//
// The returned Allocator can be used multiple times and is thread-safe.
func NewAllocator(ctx context.Context,
	features Features,
	allocatedDevices sets.Set[DeviceID],
	classLister DeviceClassLister,
	slices []*resourceapi.ResourceSlice,
	celCache *cel.Cache,
) (Allocator, error) {
	// The actual implementation may vary depending on which features are enabled.
	// At the moment there is only one. The goal is to have three:
	// - stable: the oldest, most stable code
	// - incubating: code which is aiming to become stable next
	// - experimental: brand-new code
	//
	// This corresponds roughly to GA/beta/alpha, but is intentionally not called
	// that because e.g. not all features supported by "stable" are required to be GA.
	//
	// Each implementation is completely separate in its own package.
	// Common functions are not shared and instead get duplicated to
	// keep all implementations completely independent. When comparing
	// files, stable should be a subset of incubating and incubating a
	// subset of experimental.
	//
	// Files have a _stable/incubating/experimental suffix to make it obvious
	// where log messages come from, although this breaks "diff -r".
	//
	// When "incubating" is known to be sufficiently stable, its files can
	// be copied wholesale (no editing needed except for the package and
	// file name!) into "stable", or individual chunks can be copied over.
	//
	// Unit tests are shared between all implementations.
	for _, allocator := range availableAllocators {
		// All required features supported?
		if allocator.supportedFeatures.Set().IsSuperset(features.Set()) {
			// Use it!
			return allocator.newAllocator(ctx, features, allocatedDevices, classLister, slices, celCache)
		}
	}
	return nil, fmt.Errorf("internal error: no allocator available for feature set %v", features)
}

var availableAllocators = []struct {
	supportedFeatures Features
	newAllocator      func(ctx context.Context,
		features Features,
		allocatedDevices sets.Set[DeviceID],
		classLister DeviceClassLister,
		slices []*resourceapi.ResourceSlice,
		celCache *cel.Cache,
	) (Allocator, error)
}{
	// Most stable first.
	{
		supportedFeatures: stable.SupportedFeatures,
		newAllocator: func(ctx context.Context,
			features Features,
			allocatedDevices sets.Set[DeviceID],
			classLister DeviceClassLister,
			slices []*resourceapi.ResourceSlice,
			celCache *cel.Cache,
		) (Allocator, error) {
			return stable.NewAllocator(ctx, features, allocatedDevices, classLister, slices, celCache)
		},
	},
	{
		supportedFeatures: incubating.SupportedFeatures,
		newAllocator: func(ctx context.Context,
			features Features,
			allocatedDevices sets.Set[DeviceID],
			classLister DeviceClassLister,
			slices []*resourceapi.ResourceSlice,
			celCache *cel.Cache,
		) (Allocator, error) {
			return incubating.NewAllocator(ctx, features, allocatedDevices, classLister, slices, celCache)
		},
	},
}
