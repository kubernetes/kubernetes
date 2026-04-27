/*
 *
 * Copyright 2026 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package mem provides utilities that facilitate memory reuse in byte slices
// that are used as buffers.
package mem

import (
	"fmt"
	"math/bits"
	"slices"
	"sort"
	"sync"
)

const (
	goPageSize = 4 * 1024 // 4KiB. N.B. this must be a power of 2.
)

var uintSize = bits.UintSize // use a variable for mocking during tests.

// bufferPool is a copy of the public bufferPool interface used to avoid
// circular dependencies.
type bufferPool interface {
	// Get returns a buffer with specified length from the pool.
	Get(length int) *[]byte

	// Put returns a buffer to the pool.
	//
	// The provided pointer must hold a prefix of the buffer obtained via
	// BufferPool.Get to ensure the buffer's entire capacity can be re-used.
	Put(*[]byte)
}

// BinaryTieredBufferPool is a buffer pool that uses multiple sub-pools with
// power-of-two sizes.
type BinaryTieredBufferPool struct {
	// exponentToNextLargestPoolMap maps a power-of-two exponent (e.g., 12 for
	// 4KB) to the index of the next largest sizedBufferPool. This is used by
	// Get() to find the smallest pool that can satisfy a request for a given
	// size.
	exponentToNextLargestPoolMap []int
	// exponentToPreviousLargestPoolMap maps a power-of-two exponent to the
	// index of the previous largest sizedBufferPool. This is used by Put()
	// to return a buffer to the most appropriate pool based on its capacity.
	exponentToPreviousLargestPoolMap []int
	sizedPools                       []bufferPool
	fallbackPool                     bufferPool
	maxPoolCap                       int // Optimization: Cache max capacity
}

// NewBinaryTieredBufferPool returns a BufferPool backed by multiple sub-pools.
// This structure enables O(1) lookup time for Get and Put operations.
//
// The arguments provided are the exponents for the buffer capacities (powers
// of 2), not the raw byte sizes. For example, to create a pool of 16KB buffers
// (2^14 bytes), pass 14 as the argument.
func NewBinaryTieredBufferPool(powerOfTwoExponents ...uint8) (*BinaryTieredBufferPool, error) {
	return newBinaryTiered(func(size int) bufferPool {
		return newSizedBufferPool(size, true)
	}, &simpleBufferPool{shouldZero: true}, powerOfTwoExponents...)
}

// NewDirtyBinaryTieredBufferPool returns a BufferPool backed by multiple
// sub-pools. It is similar to NewBinaryTieredBufferPool but it does not
// initialize the buffers before returning them.
func NewDirtyBinaryTieredBufferPool(powerOfTwoExponents ...uint8) (*BinaryTieredBufferPool, error) {
	return newBinaryTiered(func(size int) bufferPool {
		return newSizedBufferPool(size, false)
	}, &simpleBufferPool{shouldZero: false}, powerOfTwoExponents...)
}

func newBinaryTiered(sizedPoolFactory func(int) bufferPool, fallbackPool bufferPool, powerOfTwoExponents ...uint8) (*BinaryTieredBufferPool, error) {
	slices.Sort(powerOfTwoExponents)
	powerOfTwoExponents = slices.Compact(powerOfTwoExponents)

	// Determine the maximum exponent we need to support. This depends on the
	// word size (32-bit vs 64-bit).
	maxExponent := uintSize - 2
	indexOfNextLargestBit := slices.Repeat([]int{-1}, maxExponent+1)
	indexOfPreviousLargestBit := slices.Repeat([]int{-1}, maxExponent+1)

	maxTier := 0
	pools := make([]bufferPool, 0, len(powerOfTwoExponents))

	for i, exp := range powerOfTwoExponents {
		// Allocating slices of size > 2^maxExponent isn't possible on
		// maxExponent-bit machines.
		if int(exp) > maxExponent {
			return nil, fmt.Errorf("mem: allocating slice of size 2^%d is not possible", exp)
		}
		tierSize := 1 << exp
		pools = append(pools, sizedPoolFactory(tierSize))
		maxTier = max(maxTier, tierSize)

		// Map the exact power of 2 to this pool index.
		indexOfNextLargestBit[exp] = i
		indexOfPreviousLargestBit[exp] = i
	}

	// Fill gaps for Get() (Next Largest)
	// We iterate backwards. If current is empty, take the value from the right (larger).
	for i := maxExponent - 1; i >= 0; i-- {
		if indexOfNextLargestBit[i] == -1 {
			indexOfNextLargestBit[i] = indexOfNextLargestBit[i+1]
		}
	}

	// Fill gaps for Put() (Previous Largest)
	// We iterate forwards. If current is empty, take the value from the left (smaller).
	for i := 1; i <= maxExponent; i++ {
		if indexOfPreviousLargestBit[i] == -1 {
			indexOfPreviousLargestBit[i] = indexOfPreviousLargestBit[i-1]
		}
	}

	return &BinaryTieredBufferPool{
		exponentToNextLargestPoolMap:     indexOfNextLargestBit,
		exponentToPreviousLargestPoolMap: indexOfPreviousLargestBit,
		sizedPools:                       pools,
		maxPoolCap:                       maxTier,
		fallbackPool:                     fallbackPool,
	}, nil
}

// Get returns a buffer with specified length from the pool.
func (b *BinaryTieredBufferPool) Get(size int) *[]byte {
	return b.poolForGet(size).Get(size)
}

func (b *BinaryTieredBufferPool) poolForGet(size int) bufferPool {
	if size == 0 || size > b.maxPoolCap {
		return b.fallbackPool
	}

	// Calculate the exponent of the smallest power of 2 >= size.
	// We subtract 1 from size to handle exact powers of 2 correctly.
	//
	// Examples:
	// size=16 (0b10000) -> size-1=15 (0b01111) -> bits.Len=4 -> Pool for 2^4
	// size=17 (0b10001) -> size-1=16 (0b10000) -> bits.Len=5 -> Pool for 2^5
	querySize := uint(size - 1)
	poolIdx := b.exponentToNextLargestPoolMap[bits.Len(querySize)]

	return b.sizedPools[poolIdx]
}

// Put returns a buffer to the pool.
func (b *BinaryTieredBufferPool) Put(buf *[]byte) {
	// We pass the capacity of the buffer, and not the size of the buffer here.
	// If we did the latter, all buffers would eventually move to the smallest
	// pool.
	b.poolForPut(cap(*buf)).Put(buf)
}

func (b *BinaryTieredBufferPool) poolForPut(bCap int) bufferPool {
	if bCap == 0 {
		return NopBufferPool{}
	}
	if bCap > b.maxPoolCap {
		return b.fallbackPool
	}
	// Find the pool with the largest capacity <= bCap.
	//
	// We calculate the exponent of the largest power of 2 <= bCap.
	// bits.Len(x) returns the minimum number of bits required to represent x;
	// i.e. the number of bits up to and including the most significant bit.
	// Subtracting 1 gives the 0-based index of the most significant bit,
	// which is the exponent of the largest power of 2 <= bCap.
	//
	// Examples:
	// cap=16 (0b10000) -> Len=5 -> 5-1=4 -> 2^4
	// cap=15 (0b01111) -> Len=4 -> 4-1=3 -> 2^3
	largestPowerOfTwo := bits.Len(uint(bCap)) - 1
	poolIdx := b.exponentToPreviousLargestPoolMap[largestPowerOfTwo]
	// The buffer is smaller than the smallest power of 2, discard it.
	if poolIdx == -1 {
		// Buffer is smaller than our smallest pool bucket.
		return NopBufferPool{}
	}
	return b.sizedPools[poolIdx]
}

// NopBufferPool is a buffer pool that returns new buffers without pooling.
type NopBufferPool struct{}

// Get returns a buffer with specified length from the pool.
func (NopBufferPool) Get(length int) *[]byte {
	b := make([]byte, length)
	return &b
}

// Put returns a buffer to the pool.
func (NopBufferPool) Put(*[]byte) {
}

// sizedBufferPool is a BufferPool implementation that is optimized for specific
// buffer sizes. For example, HTTP/2 frames within gRPC have a default max size
// of 16kb and a sizedBufferPool can be configured to only return buffers with a
// capacity of 16kb. Note that however it does not support returning larger
// buffers and in fact panics if such a buffer is requested. Because of this,
// this BufferPool implementation is not meant to be used on its own and rather
// is intended to be embedded in a TieredBufferPool such that Get is only
// invoked when the required size is smaller than or equal to defaultSize.
type sizedBufferPool struct {
	pool        sync.Pool
	defaultSize int
	shouldZero  bool
}

func (p *sizedBufferPool) Get(size int) *[]byte {
	buf, ok := p.pool.Get().(*[]byte)
	if !ok {
		buf := make([]byte, size, p.defaultSize)
		return &buf
	}
	b := *buf
	if p.shouldZero {
		clear(b[:cap(b)])
	}
	*buf = b[:size]
	return buf
}

func (p *sizedBufferPool) Put(buf *[]byte) {
	if cap(*buf) < p.defaultSize {
		// Ignore buffers that are too small to fit in the pool. Otherwise, when
		// Get is called it will panic as it tries to index outside the bounds
		// of the buffer.
		return
	}
	p.pool.Put(buf)
}

func newSizedBufferPool(size int, zero bool) *sizedBufferPool {
	return &sizedBufferPool{
		defaultSize: size,
		shouldZero:  zero,
	}
}

// TieredBufferPool implements the BufferPool interface with multiple tiers of
// buffer pools for different sizes of buffers.
type TieredBufferPool struct {
	sizedPools   []*sizedBufferPool
	fallbackPool simpleBufferPool
}

// NewTieredBufferPool returns a BufferPool implementation that uses multiple
// underlying pools of the given pool sizes.
func NewTieredBufferPool(poolSizes ...int) *TieredBufferPool {
	sort.Ints(poolSizes)
	pools := make([]*sizedBufferPool, len(poolSizes))
	for i, s := range poolSizes {
		pools[i] = newSizedBufferPool(s, true)
	}
	return &TieredBufferPool{
		sizedPools:   pools,
		fallbackPool: simpleBufferPool{shouldZero: true},
	}
}

// Get returns a buffer with specified length from the pool.
func (p *TieredBufferPool) Get(size int) *[]byte {
	return p.getPool(size).Get(size)
}

// Put returns a buffer to the pool.
func (p *TieredBufferPool) Put(buf *[]byte) {
	p.getPool(cap(*buf)).Put(buf)
}

func (p *TieredBufferPool) getPool(size int) bufferPool {
	poolIdx := sort.Search(len(p.sizedPools), func(i int) bool {
		return p.sizedPools[i].defaultSize >= size
	})

	if poolIdx == len(p.sizedPools) {
		return &p.fallbackPool
	}

	return p.sizedPools[poolIdx]
}

// simpleBufferPool is an implementation of the BufferPool interface that
// attempts to pool buffers with a sync.Pool. When Get is invoked, it tries to
// acquire a buffer from the pool but if that buffer is too small, it returns it
// to the pool and creates a new one.
type simpleBufferPool struct {
	pool       sync.Pool
	shouldZero bool
}

func (p *simpleBufferPool) Get(size int) *[]byte {
	bs, ok := p.pool.Get().(*[]byte)
	if ok && cap(*bs) >= size {
		if p.shouldZero {
			clear((*bs)[:cap(*bs)])
		}
		*bs = (*bs)[:size]
		return bs
	}

	// A buffer was pulled from the pool, but it is too small. Put it back in
	// the pool and create one large enough.
	if ok {
		p.pool.Put(bs)
	}

	// If we're going to allocate, round up to the nearest page. This way if
	// requests frequently arrive with small variation we don't allocate
	// repeatedly if we get unlucky and they increase over time. By default we
	// only allocate here if size > 1MiB. Because goPageSize is a power of 2, we
	// can round up efficiently.
	allocSize := (size + goPageSize - 1) & ^(goPageSize - 1)

	b := make([]byte, size, allocSize)
	return &b
}

func (p *simpleBufferPool) Put(buf *[]byte) {
	p.pool.Put(buf)
}
