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

import "k8s.io/dynamic-resource-allocation/structured/internal"

// DeviceAllocation describes one device the allocator is about to include
// in a claim's allocation result. All fields hold final values:
// ConsumedCapacity has already been rounded up per the device's
// requestPolicy, so a constraint sees the same numbers that end up in
// DeviceRequestAllocationResult.
type DeviceAllocation = internal.DeviceAllocation

// AllocationConstraint is a caller-supplied constraint that participates
// in the allocator's backtracking search. Add returns false to reject the
// candidate. When the search later backtracks over an accepted candidate,
// Remove is called with the same DeviceAllocation. Remove is not called for
// allocations retained in the final result.
type AllocationConstraint = internal.AllocationConstraint

// NewAllocationConstraintFunc is called once per Allocate. Each returned
// constraint applies to every candidate allocation for every claim passed to
// that call, in slice order. The provider may be called concurrently by
// different Allocate calls, but each returned constraint is used only by the
// call for which it was created.
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
