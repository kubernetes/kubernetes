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

	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
)

// Interface manages the allocation of ports out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(int) error
	AllocateNext() (int, error)
	Release(int) error
	ForEach(func(int))
	Has(int) bool
	Destroy()
	EnableMetrics()
}

var (
	ErrFull              = errors.New("range is full")
	ErrAllocated         = errors.New("provided port is already allocated")
	ErrMismatchedNetwork = errors.New("the provided port range does not match the current port range")
)

type ErrNotInRange struct {
	ValidPorts string
}

func (e *ErrNotInRange) Error() string {
	return fmt.Sprintf("provided port is not in the valid range. The range of valid ports is %s", e.ValidPorts)
}

type PortAllocator struct {
	portRange net.PortRange

	alloc allocator.Interface

	// metrics is a metrics recorder that can be disabled
	metrics metricsRecorderInterface
}

// PortAllocator implements Interface and Snapshottable
var _ Interface = &PortAllocator{}

// New creates a PortAllocator over a net.PortRange, calling allocatorFactory to construct the backing store.
func New(pr net.PortRange, allocatorFactory allocator.AllocatorWithOffsetFactory) (*PortAllocator, error) {
	max := pr.Size
	rangeSpec := pr.String()

	a := &PortAllocator{
		portRange: pr,
		metrics:   &emptyMetricsRecorder{},
	}

	var err error
	a.alloc, err = allocatorFactory(max, rangeSpec, calculateRangeOffset(pr))
	if err != nil {
		return nil, err
	}

	return a, err
}

// NewInMemory creates an in-memory allocator.
func NewInMemory(pr net.PortRange) (*PortAllocator, error) {
	return New(pr, func(max int, rangeSpec string, offset int) (allocator.Interface, error) {
		return allocator.NewAllocationMapWithOffset(max, rangeSpec, offset), nil
	})
}

// NewFromSnapshot allocates a PortAllocator and initializes it from a snapshot.
func NewFromSnapshot(snap *api.RangeAllocation) (*PortAllocator, error) {
	pr, err := net.ParsePortRange(snap.Range)
	if err != nil {
		return nil, err
	}
	r, err := NewInMemory(*pr)
	if err != nil {
		return nil, err
	}
	if err := r.Restore(*pr, snap.Data); err != nil {
		return nil, err
	}
	return r, nil
}

// Free returns the count of port left in the range.
func (r *PortAllocator) Free() int {
	return r.alloc.Free()
}

// Used returns the count of ports used in the range.
func (r *PortAllocator) Used() int {
	return r.portRange.Size - r.alloc.Free()
}

// Allocate attempts to reserve the provided port. ErrNotInRange or
// ErrAllocated will be returned if the port is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no ports left.
func (r *PortAllocator) Allocate(port int) error {
	ok, offset := r.contains(port)
	if !ok {
		// update metrics
		r.metrics.incrementAllocationErrors("static")

		// include valid port range in error
		validPorts := r.portRange.String()
		return &ErrNotInRange{validPorts}
	}

	allocated, err := r.alloc.Allocate(offset)
	if err != nil {
		// update metrics
		r.metrics.incrementAllocationErrors("static")
		return err
	}
	if !allocated {
		// update metrics
		r.metrics.incrementAllocationErrors("static")
		return ErrAllocated
	}

	// update metrics
	r.metrics.incrementAllocations("static")
	r.metrics.setAllocated(r.Used())
	r.metrics.setAvailable(r.Free())

	return nil
}

// AllocateNext reserves one of the ports from the pool. ErrFull may
// be returned if there are no ports left.
func (r *PortAllocator) AllocateNext() (int, error) {
	offset, ok, err := r.alloc.AllocateNext()
	if err != nil {
		r.metrics.incrementAllocationErrors("dynamic")
		return 0, err
	}
	if !ok {
		r.metrics.incrementAllocationErrors("dynamic")
		return 0, ErrFull
	}

	// update metrics
	r.metrics.incrementAllocations("dynamic")
	r.metrics.setAllocated(r.Used())
	r.metrics.setAvailable(r.Free())

	return r.portRange.Base + offset, nil
}

// ForEach calls the provided function for each allocated port.
func (r *PortAllocator) ForEach(fn func(int)) {
	r.alloc.ForEach(func(offset int) {
		fn(r.portRange.Base + offset)
	})
}

// Release releases the port back to the pool. Releasing an
// unallocated port or a port out of the range is a no-op and
// returns no error.
func (r *PortAllocator) Release(port int) error {
	ok, offset := r.contains(port)
	if !ok {
		klog.Warningf("port is not in the range when release it. port: %v", port)
		return nil
	}

	err := r.alloc.Release(offset)
	if err == nil {
		// update metrics
		r.metrics.setAllocated(r.Used())
		r.metrics.setAvailable(r.Free())
	}
	return err
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

// Destroy shuts down internal allocator.
func (r *PortAllocator) Destroy() {
	r.alloc.Destroy()
}

// EnableMetrics enables metrics recording.
func (r *PortAllocator) EnableMetrics() {
	registerMetrics()
	r.metrics = &metricsRecorder{}
}

// calculateRangeOffset estimates the offset used on the range for statically allocation based on
// the following formula `min(max($min, rangeSize/$step), $max)`, described as ~never less than
// $min or more than $max, with a graduated step function between them~. The function returns 0
// if any of the parameters is invalid.
func calculateRangeOffset(pr net.PortRange) int {
	// default values for min(max($min, rangeSize/$step), $max)
	const (
		min  = 16
		max  = 128
		step = 32
	)

	rangeSize := pr.Size
	// offset should always be smaller than the range size
	if rangeSize <= min {
		return 0
	}

	offset := rangeSize / step
	if offset < min {
		return min
	}
	if offset > max {
		return max
	}
	return int(offset)
}
