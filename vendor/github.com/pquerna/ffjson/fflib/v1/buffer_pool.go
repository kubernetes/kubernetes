// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.3

package v1

// Allocation pools for Buffers.

import "sync"

var pools [14]sync.Pool
var pool64 *sync.Pool

func init() {
	var i uint
	// TODO(pquerna): add science here around actual pool sizes.
	for i = 6; i < 20; i++ {
		n := 1 << i
		pools[poolNum(n)].New = func() interface{} { return make([]byte, 0, n) }
	}
	pool64 = &pools[0]
}

// This returns the pool number that will give a buffer of
// at least 'i' bytes.
func poolNum(i int) int {
	// TODO(pquerna): convert to log2 w/ bsr asm instruction:
	// 	<https://groups.google.com/forum/#!topic/golang-nuts/uAb5J1_y7ns>
	if i <= 64 {
		return 0
	} else if i <= 128 {
		return 1
	} else if i <= 256 {
		return 2
	} else if i <= 512 {
		return 3
	} else if i <= 1024 {
		return 4
	} else if i <= 2048 {
		return 5
	} else if i <= 4096 {
		return 6
	} else if i <= 8192 {
		return 7
	} else if i <= 16384 {
		return 8
	} else if i <= 32768 {
		return 9
	} else if i <= 65536 {
		return 10
	} else if i <= 131072 {
		return 11
	} else if i <= 262144 {
		return 12
	} else if i <= 524288 {
		return 13
	} else {
		return -1
	}
}

// Send a buffer to the Pool to reuse for other instances.
// You may no longer utilize the content of the buffer, since it may be used
// by other goroutines.
func Pool(b []byte) {
	if b == nil {
		return
	}
	c := cap(b)

	// Our smallest buffer is 64 bytes, so we discard smaller buffers.
	if c < 64 {
		return
	}

	// We need to put the incoming buffer into the NEXT buffer,
	// since a buffer guarantees AT LEAST the number of bytes available
	// that is the top of this buffer.
	// That is the reason for dividing the cap by 2, so it gets into the NEXT bucket.
	// We add 2 to avoid rounding down if size is exactly power of 2.
	pn := poolNum((c + 2) >> 1)
	if pn != -1 {
		pools[pn].Put(b[0:0])
	}
	// if we didn't have a slot for this []byte, we just drop it and let the GC
	// take care of it.
}

// makeSlice allocates a slice of size n -- it will attempt to use a pool'ed
// instance whenever possible.
func makeSlice(n int) []byte {
	if n <= 64 {
		return pool64.Get().([]byte)[0:n]
	}

	pn := poolNum(n)

	if pn != -1 {
		return pools[pn].Get().([]byte)[0:n]
	} else {
		return make([]byte, n)
	}
}
