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

package ipallocator

import (
	"errors"
	"fmt"
	"math/big"
	"net"

	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/core/service/allocator"
	netutils "k8s.io/utils/net"
)

// Interface manages the allocation of IP addresses out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(net.IP) error
	AllocateNext() (net.IP, error)
	Release(net.IP) error
	ForEach(func(net.IP))
	CIDR() net.IPNet
	IPFamily() api.IPFamily
	Has(ip net.IP) bool

	// DryRun offers a way to try operations without persisting them.
	DryRun() Interface
}

var (
	ErrFull              = errors.New("range is full")
	ErrAllocated         = errors.New("provided IP is already allocated")
	ErrMismatchedNetwork = errors.New("the provided network does not match the current range")
)

type ErrNotInRange struct {
	IP         net.IP
	ValidRange string
}

func (e *ErrNotInRange) Error() string {
	return fmt.Sprintf("the provided IP (%v) is not in the valid range. The range of valid IPs is %s", e.IP, e.ValidRange)
}

// Range is a contiguous block of IPs that can be allocated atomically.
//
// The internal structure of the range is:
//
//   For CIDR 10.0.0.0/24
//   254 addresses usable out of 256 total (minus base and broadcast IPs)
//     The number of usable addresses is r.max
//
//   CIDR base IP          CIDR broadcast IP
//   10.0.0.0                     10.0.0.255
//   |                                     |
//   0 1 2 3 4 5 ...         ... 253 254 255
//     |                              |
//   r.base                     r.base + r.max
//     |                              |
//   offset #0 of r.allocated   last offset of r.allocated
type Range struct {
	net *net.IPNet
	// base is a cached version of the start IP in the CIDR range as a *big.Int
	base *big.Int
	// max is the maximum size of the usable addresses in the range
	max int
	// family is the IP family of this range
	family api.IPFamily

	alloc allocator.Interface
}

// New creates a Range over a net.IPNet, calling allocatorFactory to construct the backing store.
func New(cidr *net.IPNet, allocatorFactory allocator.AllocatorFactory) (*Range, error) {
	registerMetrics()

	max := netutils.RangeSize(cidr)
	base := netutils.BigForIP(cidr.IP)
	rangeSpec := cidr.String()
	var family api.IPFamily

	if netutils.IsIPv6CIDR(cidr) {
		family = api.IPv6Protocol
		// Limit the max size, since the allocator keeps a bitmap of that size.
		if max > 65536 {
			max = 65536
		}
	} else {
		family = api.IPv4Protocol
		// Don't use the IPv4 network's broadcast address, but don't just
		// Allocate() it - we don't ever want to be able to release it.
		max--
	}

	// Don't use the network's ".0" address, but don't just Allocate() it - we
	// don't ever want to be able to release it.
	base.Add(base, big.NewInt(1))
	max--

	r := Range{
		net:    cidr,
		base:   base,
		max:    maximum(0, int(max)),
		family: family,
	}
	var err error
	r.alloc, err = allocatorFactory(r.max, rangeSpec)

	return &r, err
}

// NewInMemory creates an in-memory allocator.
func NewInMemory(cidr *net.IPNet) (*Range, error) {
	return New(cidr, func(max int, rangeSpec string) (allocator.Interface, error) {
		return allocator.NewAllocationMap(max, rangeSpec), nil
	})
}

// NewFromSnapshot allocates a Range and initializes it from a snapshot.
func NewFromSnapshot(snap *api.RangeAllocation) (*Range, error) {
	_, ipnet, err := netutils.ParseCIDRSloppy(snap.Range)
	if err != nil {
		return nil, err
	}
	r, err := NewInMemory(ipnet)
	if err != nil {
		return nil, err
	}
	if err := r.Restore(ipnet, snap.Data); err != nil {
		return nil, err
	}
	return r, nil
}

func maximum(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Free returns the count of IP addresses left in the range.
func (r *Range) Free() int {
	return r.alloc.Free()
}

// Used returns the count of IP addresses used in the range.
func (r *Range) Used() int {
	return r.max - r.alloc.Free()
}

// CIDR returns the CIDR covered by the range.
func (r *Range) CIDR() net.IPNet {
	return *r.net
}

// DryRun returns a non-persisting form of this Range.
func (r *Range) DryRun() Interface {
	return dryRunRange{r}
}

// For clearer code.
const dryRunTrue = true
const dryRunFalse = false

// Allocate attempts to reserve the provided IP. ErrNotInRange or
// ErrAllocated will be returned if the IP is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no addresses left.
func (r *Range) Allocate(ip net.IP) error {
	return r.allocate(ip, dryRunFalse)
}

func (r *Range) allocate(ip net.IP, dryRun bool) error {
	label := r.CIDR()
	ok, offset := r.contains(ip)
	if !ok {
		// update metrics
		clusterIPAllocationErrors.WithLabelValues(label.String()).Inc()
		return &ErrNotInRange{ip, r.net.String()}
	}
	if dryRun {
		// Don't bother to check whether the IP is actually free. It's racy and
		// not worth the effort to plumb any further.
		return nil
	}

	allocated, err := r.alloc.Allocate(offset)
	if err != nil {
		// update metrics
		clusterIPAllocationErrors.WithLabelValues(label.String()).Inc()

		return err
	}
	if !allocated {
		// update metrics
		clusterIPAllocationErrors.WithLabelValues(label.String()).Inc()

		return ErrAllocated
	}
	// update metrics
	clusterIPAllocations.WithLabelValues(label.String()).Inc()
	clusterIPAllocated.WithLabelValues(label.String()).Set(float64(r.Used()))
	clusterIPAvailable.WithLabelValues(label.String()).Set(float64(r.Free()))

	return nil
}

// AllocateNext reserves one of the IPs from the pool. ErrFull may
// be returned if there are no addresses left.
func (r *Range) AllocateNext() (net.IP, error) {
	return r.allocateNext(dryRunFalse)
}

func (r *Range) allocateNext(dryRun bool) (net.IP, error) {
	label := r.CIDR()
	if dryRun {
		// Don't bother finding a free value. It's racy and not worth the
		// effort to plumb any further.
		return r.CIDR().IP, nil
	}

	offset, ok, err := r.alloc.AllocateNext()
	if err != nil {
		// update metrics
		clusterIPAllocationErrors.WithLabelValues(label.String()).Inc()

		return nil, err
	}
	if !ok {
		// update metrics
		clusterIPAllocationErrors.WithLabelValues(label.String()).Inc()

		return nil, ErrFull
	}
	// update metrics
	clusterIPAllocations.WithLabelValues(label.String()).Inc()
	clusterIPAllocated.WithLabelValues(label.String()).Set(float64(r.Used()))
	clusterIPAvailable.WithLabelValues(label.String()).Set(float64(r.Free()))

	return netutils.AddIPOffset(r.base, offset), nil
}

// Release releases the IP back to the pool. Releasing an
// unallocated IP or an IP out of the range is a no-op and
// returns no error.
func (r *Range) Release(ip net.IP) error {
	return r.release(ip, dryRunFalse)
}

func (r *Range) release(ip net.IP, dryRun bool) error {
	ok, offset := r.contains(ip)
	if !ok {
		return nil
	}
	if dryRun {
		return nil
	}

	err := r.alloc.Release(offset)
	if err == nil {
		// update metrics
		label := r.CIDR()
		clusterIPAllocated.WithLabelValues(label.String()).Set(float64(r.Used()))
		clusterIPAvailable.WithLabelValues(label.String()).Set(float64(r.Free()))
	}
	return err
}

// ForEach calls the provided function for each allocated IP.
func (r *Range) ForEach(fn func(net.IP)) {
	r.alloc.ForEach(func(offset int) {
		ip, _ := netutils.GetIndexedIP(r.net, offset+1) // +1 because Range doesn't store IP 0
		fn(ip)
	})
}

// Has returns true if the provided IP is already allocated and a call
// to Allocate(ip) would fail with ErrAllocated.
func (r *Range) Has(ip net.IP) bool {
	ok, offset := r.contains(ip)
	if !ok {
		return false
	}

	return r.alloc.Has(offset)
}

// IPFamily returns the IP family of this range.
func (r *Range) IPFamily() api.IPFamily {
	return r.family
}

// Snapshot saves the current state of the pool.
func (r *Range) Snapshot(dst *api.RangeAllocation) error {
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
// is returned if the provided IPNet range doesn't exactly match the previous range.
func (r *Range) Restore(net *net.IPNet, data []byte) error {
	if !net.IP.Equal(r.net.IP) || net.Mask.String() != r.net.Mask.String() {
		return ErrMismatchedNetwork
	}
	snapshottable, ok := r.alloc.(allocator.Snapshottable)
	if !ok {
		return fmt.Errorf("not a snapshottable allocator")
	}
	if err := snapshottable.Restore(net.String(), data); err != nil {
		return fmt.Errorf("restoring snapshot encountered %v", err)
	}
	return nil
}

// contains returns true and the offset if the ip is in the range, and false
// and nil otherwise. The first and last addresses of the CIDR are omitted.
func (r *Range) contains(ip net.IP) (bool, int) {
	if !r.net.Contains(ip) {
		return false, 0
	}

	offset := calculateIPOffset(r.base, ip)
	if offset < 0 || offset >= r.max {
		return false, 0
	}
	return true, offset
}

// calculateIPOffset calculates the integer offset of ip from base such that
// base + offset = ip. It requires ip >= base.
func calculateIPOffset(base *big.Int, ip net.IP) int {
	return int(big.NewInt(0).Sub(netutils.BigForIP(ip), base).Int64())
}

// dryRunRange is a shim to satisfy Interface without persisting state.
type dryRunRange struct {
	real *Range
}

func (dry dryRunRange) Allocate(ip net.IP) error {
	return dry.real.allocate(ip, dryRunTrue)
}

func (dry dryRunRange) AllocateNext() (net.IP, error) {
	return dry.real.allocateNext(dryRunTrue)
}

func (dry dryRunRange) Release(ip net.IP) error {
	return dry.real.release(ip, dryRunTrue)
}

func (dry dryRunRange) ForEach(cb func(net.IP)) {
	dry.real.ForEach(cb)
}

func (dry dryRunRange) CIDR() net.IPNet {
	return dry.real.CIDR()
}

func (dry dryRunRange) IPFamily() api.IPFamily {
	return dry.real.IPFamily()
}

func (dry dryRunRange) DryRun() Interface {
	return dry
}

func (dry dryRunRange) Has(ip net.IP) bool {
	return dry.real.Has(ip)
}
