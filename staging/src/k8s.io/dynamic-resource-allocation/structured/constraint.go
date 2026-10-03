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

package structured

import (
	"context"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	"k8s.io/dynamic-resource-allocation/structured/internal"
)

// DeviceAllocation describes one device the allocator is about to include
// in a claim's allocation result. All fields hold final values:
// ConsumedCapacity has already been rounded up per the device's
// requestPolicy, so a constraint sees the same numbers that end up in
// DeviceRequestAllocationResult.
type DeviceAllocation = internal.DeviceAllocation

// AllocationConstraint is a caller-supplied constraint that participates
// in the allocator's backtracking search. Add returns false to reject
// the candidate; the allocator then backtracks. For every Add which
// returned true there is exactly one Remove with an equal DeviceAllocation.
type AllocationConstraint = internal.AllocationConstraint

// NewAllocationConstraintFunc is called once per Allocate to produce
// the constraints for that invocation. Allocate runs concurrently for
// different nodes, so state shared between returned instances must be
// read-only.
type NewAllocationConstraintFunc = internal.NewAllocationConstraintFunc

// Option customizes an Allocator created by NewAllocator.
type Option func(*options)

type options struct {
	newAllocationConstraints NewAllocationConstraintFunc
}

// WithAllocationConstraints supplies a provider of caller-defined
// constraints. A nil function disables the hook; the default behaviour
// is unchanged.
func WithAllocationConstraints(fn NewAllocationConstraintFunc) Option {
	return func(o *options) {
		o.newAllocationConstraints = fn
	}
}

// The type assertions below keep the exported contract honest: the
// compiler fails here if the internal types drift away from what this
// package documents.
var (
	_ DeviceAllocation = internal.DeviceAllocation{
		Claim:            nil,
		Request:          "",
		DeviceClassName:  "",
		Device:           DeviceID{},
		ConsumedCapacity: map[resourceapi.QualifiedName]resource.Quantity{},
		AdminAccess:      false,
	}
	_ context.Context            = nil
	_ *v1.Pod                    = nil
	_ AllocationConstraint       = nil
	_ *resourceapi.ResourceClaim = nil
)
