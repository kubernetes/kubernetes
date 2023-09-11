/*
 *
 * Copyright 2023 gRPC authors.
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

package grpc

import "sync"

// SharedBufferPool is a pool of buffers that can be shared, resulting in
// decreased memory allocation. Currently, in gRPC-go, it is only utilized
// for parsing incoming messages.
//
// # Experimental
//
// Notice: This API is EXPERIMENTAL and may be changed or removed in a
// later release.
type SharedBufferPool interface {
	// Get returns a buffer with specified length from the pool.
	//
	// The returned byte slice may be not zero initialized.
	Get(length int) []byte

	// Put returns a buffer to the pool.
	Put(*[]byte)
}

// NewSharedBufferPool creates a simple SharedBufferPool with buckets
// of different sizes to optimize memory usage. This prevents the pool from
// wasting large amounts of memory, even when handling messages of varying sizes.
//
// # Experimental
//
// Notice: This API is EXPERIMENTAL and may be changed or removed in a
// later release.
func NewSharedBufferPool() SharedBufferPool {
	return &simpleSharedBufferPool{
		pools: [poolArraySize]simpleSharedBufferChildPool{
			newBytesPool(level0PoolMaxSize),
			newBytesPool(level1PoolMaxSize),
			newBytesPool(level2PoolMaxSize),
			newBytesPool(level3PoolMaxSize),
			newBytesPool(level4PoolMaxSize),
			newBytesPool(0),
		},
	}
}

// simpleSharedBufferPool is a simple implementation of SharedBufferPool.
type simpleSharedBufferPool struct {
	pools [poolArraySize]simpleSharedBufferChildPool
}

func (p *simpleSharedBufferPool) Get(size int) []byte {
	return p.pools[p.poolIdx(size)].Get(size)
}

func (p *simpleSharedBufferPool) Put(bs *[]byte) {
	p.pools[p.poolIdx(cap(*bs))].Put(bs)
}

func (p *simpleSharedBufferPool) poolIdx(size int) int {
	switch {
	case size <= level0PoolMaxSize:
		return level0PoolIdx
	case size <= level1PoolMaxSize:
		return level1PoolIdx
	case size <= level2PoolMaxSize:
		return level2PoolIdx
	case size <= level3PoolMaxSize:
		return level3PoolIdx
	case size <= level4PoolMaxSize:
		return level4PoolIdx
	default:
		return levelMaxPoolIdx
	}
}

const (
	level0PoolMaxSize = 16                     //  16  B
	level1PoolMaxSize = level0PoolMaxSize * 16 // 256  B
	level2PoolMaxSize = level1PoolMaxSize * 16 //   4 KB
	level3PoolMaxSize = level2PoolMaxSize * 16 //  64 KB
	level4PoolMaxSize = level3PoolMaxSize * 16 //   1 MB
)

const (
	level0PoolIdx = iota
	level1PoolIdx
	level2PoolIdx
	level3PoolIdx
	level4PoolIdx
	levelMaxPoolIdx
	poolArraySize
)

type simpleSharedBufferChildPool interface {
	Get(size int) []byte
	Put(interface{})
}

type bufferPool struct {
	sync.Pool

	defaultSize int
}

func (p *bufferPool) Get(size int) []byte {
	bs := p.Pool.Get().(*[]byte)

	if cap(*bs) < size {
		p.Pool.Put(bs)

		return make([]byte, size)
	}

	return (*bs)[:size]
}

func newBytesPool(size int) simpleSharedBufferChildPool {
	return &bufferPool{
		Pool: sync.Pool{
			New: func() interface{} {
				bs := make([]byte, size)
				return &bs
			},
		},
		defaultSize: size,
	}
}

// nopBufferPool is a buffer pool just makes new buffer without pooling.
type nopBufferPool struct {
}

func (nopBufferPool) Get(length int) []byte {
	return make([]byte, length)
}

func (nopBufferPool) Put(*[]byte) {
}
