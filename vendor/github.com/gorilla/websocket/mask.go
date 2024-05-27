// Copyright 2016 The Gorilla WebSocket Authors. All rights reserved.  Use of
// this source code is governed by a BSD-style license that can be found in the
// LICENSE file.

//go:build !appengine
// +build !appengine

package websocket

import "unsafe"

// #nosec G103 -- (CWE-242) Has been audited
const wordSize = int(unsafe.Sizeof(uintptr(0)))

func maskBytes(key [4]byte, pos int, b []byte) int {
	// Mask one byte at a time for small buffers.
	if len(b) < 2*wordSize {
		for i := range b {
			b[i] ^= key[pos&3]
			pos++
		}
		return pos & 3
	}

	// Mask one byte at a time to word boundary.
	//#nosec G103 -- (CWE-242) Has been audited
	if n := int(uintptr(unsafe.Pointer(&b[0]))) % wordSize; n != 0 {
		n = wordSize - n
		for i := range b[:n] {
			b[i] ^= key[pos&3]
			pos++
		}
		b = b[n:]
	}

	// Create aligned word size key.
	var k [wordSize]byte
	for i := range k {
		k[i] = key[(pos+i)&3]
	}
	//#nosec G103 -- (CWE-242) Has been audited
	kw := *(*uintptr)(unsafe.Pointer(&k))

	// Mask one word at a time.
	n := (len(b) / wordSize) * wordSize
	for i := 0; i < n; i += wordSize {
		//#nosec G103 -- (CWE-242) Has been audited
		*(*uintptr)(unsafe.Pointer(uintptr(unsafe.Pointer(&b[0])) + uintptr(i))) ^= kw
	}

	// Mask one byte at a time for remaining bytes.
	b = b[n:]
	for i := range b {
		b[i] ^= key[pos&3]
		pos++
	}

	return pos & 3
}
