/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"math/rand"
	"net"
	"sync"
)

// Interface manages the allocation of IP addresses out of a range. Interface
// should be threadsafe.
type Interface interface {
	Allocate(net.IP) error
	AllocateNext() (net.IP, error)
	Release(net.IP) error
}

// Snapshottable is an Interface that can be snapshotted and restored. Snapshottable
// should be threadsafe.
type Snapshottable interface {
	Interface
	Snapshot() (*net.IPNet, []byte)
	Restore(*net.IPNet, []byte) error
}

var (
	ErrFull               = errors.New("range is full")
	ErrNotInRange         = errors.New("provided IP is not in the valid range")
	ErrAllocated          = errors.New("provided IP is already allocated")
	ErrMismatchedNetwork  = errors.New("the provided network does not match the current range")
	ErrAllocationDisabled = errors.New("IP addresses cannot be allocated at this time")
)

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
//   first bit of r.allocated   last bit of r.allocated
//
// If an address is taken, the bit at offset:
//
//   bit offset := IP - r.base
//
// is set to one. r.count is always equal to the number of set bits and
// can be recalculated at any time by counting the set bits in r.allocated.
//
// TODO: use RLE and compact the allocator to minimize space.
type Range struct {
	net *net.IPNet
	// base is a cached version of the start IP in the CIDR range as a *big.Int
	base *big.Int
	// strategy is the strategy for choosing the next available IP out of the range
	strategy allocateStrategy
	// max is the maximum size of the usable addresses in the range
	max int

	// lock guards the following members
	lock sync.Mutex
	// count is the number of currently allocated elements in the range
	count int
	// allocated is a bit array of the allocated ips in the range
	allocated *big.Int
}

// allocateStrategy is a search strategy in the allocation map for a valid IP.
type allocateStrategy func(allocated *big.Int, max, count int) (int, error)

// NewCIDRRange creates a Range over a net.IPNet.
func NewCIDRRange(cidr *net.IPNet) *Range {
	max := RangeSize(cidr)
	base := bigForIP(cidr.IP)
	r := Range{
		net:      cidr,
		strategy: randomScanStrategy,
		base:     base.Add(base, big.NewInt(1)), // don't use the network base
		max:      maximum(0, int(max-2)),        // don't use the network broadcast

		allocated: big.NewInt(0),
		count:     0,
	}
	return &r
}

func maximum(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Free returns the count of IP addresses left in the range.
func (r *Range) Free() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return r.max - r.count
}

// Allocate attempts to reserve the provided IP. ErrNotInRange or
// ErrAllocated will be returned if the IP is not valid for this range
// or has already been reserved.  ErrFull will be returned if there
// are no addresses left.
func (r *Range) Allocate(ip net.IP) error {
	ok, offset := r.contains(ip)
	if !ok {
		return ErrNotInRange
	}

	r.lock.Lock()
	defer r.lock.Unlock()

	if r.allocated.Bit(offset) == 1 {
		return ErrAllocated
	}
	r.allocated = r.allocated.SetBit(r.allocated, offset, 1)
	r.count++
	return nil
}

// AllocateNext reserves one of the IPs from the pool. ErrFull may
// be returned if there are no addresses left.
func (r *Range) AllocateNext() (net.IP, error) {
	r.lock.Lock()
	defer r.lock.Unlock()

	next, err := r.strategy(r.allocated, r.max, r.count)
	if err != nil {
		return nil, err
	}
	r.count++
	r.allocated = r.allocated.SetBit(r.allocated, next, 1)
	return addIPOffset(r.base, next), nil
}

// Release releases the IP back to the pool. Releasing an
// unallocated IP or an IP out of the range is a no-op and
// returns no error.
func (r *Range) Release(ip net.IP) error {
	ok, offset := r.contains(ip)
	if !ok {
		return nil
	}

	r.lock.Lock()
	defer r.lock.Unlock()

	if r.allocated.Bit(offset) == 0 {
		return nil
	}

	r.allocated = r.allocated.SetBit(r.allocated, offset, 0)
	r.count--
	return nil
}

// Has returns true if the provided IP is already allocated and a call
// to Allocate(ip) would fail with ErrAllocated.
func (r *Range) Has(ip net.IP) bool {
	ok, offset := r.contains(ip)
	if !ok {
		return false
	}

	r.lock.Lock()
	defer r.lock.Unlock()

	return r.allocated.Bit(offset) == 1
}

// Snapshot saves the current state of the pool.
func (r *Range) Snapshot() (*net.IPNet, []byte) {
	r.lock.Lock()
	defer r.lock.Unlock()

	return r.net, r.allocated.Bytes()
}

// Restore restores the pool to the previously captured state. ErrMismatchedNetwork
// is returned if the provided IPNet range doesn't exactly match the previous range.
func (r *Range) Restore(net *net.IPNet, data []byte) error {
	r.lock.Lock()
	defer r.lock.Unlock()

	if !net.IP.Equal(r.net.IP) || net.Mask.String() != r.net.Mask.String() {
		return ErrMismatchedNetwork
	}
	r.allocated = big.NewInt(0).SetBytes(data)
	r.count = countBits(r.allocated)
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

// bigForIP creates a big.Int based on the provided net.IP
func bigForIP(ip net.IP) *big.Int {
	b := ip.To4()
	if b == nil {
		b = ip.To16()
	}
	return big.NewInt(0).SetBytes(b)
}

// addIPOffset adds the provided integer offset to a base big.Int representing a
// net.IP
func addIPOffset(base *big.Int, offset int) net.IP {
	return net.IP(big.NewInt(0).Add(base, big.NewInt(int64(offset))).Bytes())
}

// calculateIPOffset calculates the integer offset of ip from base such that
// base + offset = ip. It requires ip >= base.
func calculateIPOffset(base *big.Int, ip net.IP) int {
	return int(big.NewInt(0).Sub(bigForIP(ip), base).Int64())
}

// randomScanStrategy chooses a random address from the provided big.Int, and then
// scans forward looking for the next available address (it will wrap the range if
// necessary).
func randomScanStrategy(allocated *big.Int, max, count int) (int, error) {
	if count >= max {
		return 0, ErrFull
	}
	offset := rand.Intn(max)
	for i := 0; i < max; i++ {
		at := (offset + i) % max
		if allocated.Bit(at) == 0 {
			return at, nil
		}
	}
	return 0, ErrFull
}

// countBits returns the number of set bits in n
func countBits(n *big.Int) int {
	var count int = 0
	for _, b := range n.Bytes() {
		count += int(bitCounts[b])
	}
	return count
}

// bitCounts is all of the bits counted for each number between 0-255
var bitCounts = []int8{
	0, 1, 1, 2, 1, 2, 2, 3,
	1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4,
	2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5,
	3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6,
	4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7,
	5, 6, 6, 7, 6, 7, 7, 8,
}

// RangeSize returns the size of a range in valid addresses.
func RangeSize(subnet *net.IPNet) int64 {
	ones, bits := subnet.Mask.Size()
	if (bits - ones) >= 31 {
		panic("masks greater than 31 bits are not supported")
	}
	max := int64(1) << uint(bits-ones)
	return max
}

// GetIndexedIP returns a net.IP that is subnet.IP + index in the contiguous IP space.
func GetIndexedIP(subnet *net.IPNet, index int) (net.IP, error) {
	ip := addIPOffset(bigForIP(subnet.IP), index)
	if !subnet.Contains(ip) {
		return nil, fmt.Errorf("can't generate IP with index %d from subnet. subnet too small. subnet: %q", index, subnet)
	}
	return ip, nil
}
