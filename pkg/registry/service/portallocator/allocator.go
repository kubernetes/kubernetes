/*
Copyright 2015 The Kubernetes Authors.

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

package portallocator

import (
	"errors"
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/registry/service/allocator"
	"k8s.io/kubernetes/pkg/util/net"

	"github.com/golang/glog"
)

// Interface manages the allocation of ports out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(int) error
	AllocateNext() (int, error)
	Release(int) error
}

var (
	ErrFull              = errors.New("range is full")
	ErrNotInRange        = errors.New("provided port is not in the valid range")
	ErrAllocated         = errors.New("provided port is already allocated")
	ErrMismatchedNetwork = errors.New("the provided port range does not match the current port range")
)

type PortAllocator struct {
	portRange net.PortRange

	alloc allocator.Interface
}

// PortAllocator implements Interface and Snapshottable
var _ Interface = &PortAllocator{}

// NewPortAllocatorCustom creates a PortAllocator over a net.PortRange, calling allocatorFactory to construct the backing store.
func NewPortAllocatorCustom(pr net.PortRange, allocatorFactory allocator.AllocatorFactory) *PortAllocator {
	max := pr.Size
	rangeSpec := pr.String()

	a := &PortAllocator{
		portRange: pr,
	}
	a.alloc = allocatorFactory(max, rangeSpec)
	return a
}

// Helper that wraps NewAllocatorCIDRRange, for creating a range backed by an in-memory store.
func NewPortAllocator(pr net.PortRange) *PortAllocator {
	return NewPortAllocatorCustom(pr, func(max int, rangeSpec string) allocator.Interface {
		return allocator.NewAllocationMap(max, rangeSpec)
	})
}

// Free returns the count of port left in the range.
func (r *PortAllocator) Free() int {
	return r.alloc.Free()
}

// Allocate attempts to reserve the provided port. ErrNotInRange or
// ErrAllocated will be returned if the port is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no ports left.
func (r *PortAllocator) Allocate(port int) error {
	ok, offset := r.contains(port)
	if !ok {
		return ErrNotInRange
	}

	allocated, err := r.alloc.Allocate(offset)
	if err != nil {
		return err
	}
	if !allocated {
		return ErrAllocated
	}
	return nil
}

// AllocateNext reserves one of the ports from the pool. ErrFull may
// be returned if there are no ports left.
func (r *PortAllocator) AllocateNext() (int, error) {
	offset, ok, err := r.alloc.AllocateNext()
	if err != nil {
		return 0, err
	}
	if !ok {
		return 0, ErrFull
	}
	return r.portRange.Base + offset, nil
}

// Release releases the port back to the pool. Releasing an
// unallocated port or a port out of the range is a no-op and
// returns no error.
func (r *PortAllocator) Release(port int) error {
	ok, offset := r.contains(port)
	if !ok {
		glog.Warningf("port is not in the range when release it. port: %v", port)
		return nil
	}

	return r.alloc.Release(offset)
}

// Has returns true if the provided port is already allocated and a call
// to Allocate(port) would fail with ErrAllocated.
func (r *PortAllocator) Has(port int) bool {
	ok, offset := r.contains(port)
	if !ok {
		return false
	}

	return r.alloc.Has(offset)
}

// Snapshot saves the current state of the pool.
func (r *PortAllocator) Snapshot(dst *api.RangeAllocation) error {
	snapshottable, ok := r.alloc.(allocator.Snapshottable)
	if !ok {
		return fmt.Errorf("not a snapshottable allocator")
	}
	rangeString, data := snapshottable.Snapshot()
	dst.Range = rangeString
	dst.Data = data
	return nil
}

// Restore restores the pool to the previously captured state. ErrMismatchedNetwork
// is returned if the provided port range doesn't exactly match the previous range.
func (r *PortAllocator) Restore(pr net.PortRange, data []byte) error {
	if pr.String() != r.portRange.String() {
		return ErrMismatchedNetwork
	}
	snapshottable, ok := r.alloc.(allocator.Snapshottable)
	if !ok {
		return fmt.Errorf("not a snapshottable allocator")
	}
	return snapshottable.Restore(pr.String(), data)
}

// contains returns true and the offset if the port is in the range, and false
// and nil otherwise.
func (r *PortAllocator) contains(port int) (bool, int) {
	if !r.portRange.Contains(port) {
		return false, 0
	}

	offset := port - r.portRange.Base
	return true, offset
}
