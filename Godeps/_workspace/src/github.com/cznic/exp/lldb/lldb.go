// Copyright 2014 The lldb Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lldb (WIP) implements a low level database engine. The database
// model used could be considered a specific implementation of some small(est)
// intersection of models listed in [1]. As a settled term is lacking, it'll be
// called here a 'Virtual memory model' (VMM).
//
// Experimental release notes
//
// This is an experimental release. Don't open a DB from two applications or
// two instances of an application - it will get corrupted (no file locking is
// implemented and this task is delegated to lldb's clients).
//
// WARNING: THE LLDB API IS SUBJECT TO CHANGE.
//
// Filers
//
// A Filer is an abstraction of storage. A Filer may be a part of some process'
// virtual address space, an OS file, a networked, remote file etc. Persistence
// of the storage is optional, opaque to VMM and it is specific to a concrete
// Filer implementation.
//
// Space management
//
// Mechanism to allocate, reallocate (resize), deallocate (and later reclaim
// the unused) contiguous parts of a Filer, called blocks.  Blocks are
// identified and referred to by a handle, an int64.
//
// BTrees
//
// In addition to the VMM like services, lldb provides volatile and
// non-volatile BTrees. Keys and values of a BTree are limited in size to 64kB
// each (a bit more actually). Support for larger keys/values, if desired, can
// be built atop a BTree to certain limits.
//
// Handles vs pointers
//
// A handle is the abstracted storage counterpart of a memory address.  There
// is one fundamental difference, though. Resizing a block never results in a
// change to the handle which refers to the resized block, so a handle is more
// akin to an unique numeric id/key. Yet it shares one property of pointers -
// handles can be associated again with blocks after the original handle block
// was deallocated. In other words, a handle uniqueness domain is the state of
// the database and is not something comparable to e.g. an ever growing
// numbering sequence.
//
// Also, as with memory pointers, dangling handles can be created and blocks
// overwritten when such handles are used. Using a zero handle to refer to a
// block will not panic; however, the resulting error is effectively the same
// exceptional situation as dereferencing a nil pointer.
//
// Blocks
//
// Allocated/used blocks, are limited in size to only a little bit more than
// 64kB.  Bigger semantic entities/structures must be built in lldb's client
// code.  The content of a block has no semantics attached, it's only a fully
// opaque `[]byte`.
//
// Scalars
//
// Use of "scalars" applies to EncodeScalars, DecodeScalars and Collate. Those
// first two "to bytes" and "from bytes" functions are suggested for handling
// multi-valued Allocator content items and/or keys/values of BTrees (using
// Collate for keys). Types called "scalar" are:
//
//	nil (the typeless one)
//	bool
//	all integral types: [u]int8, [u]int16, [u]int32, [u]int, [u]int64
//	all floating point types: float32, float64
//	all complex types: complex64, complex128
//	[]byte (64kB max)
//	string (64kb max)
//
// Specific implementations
//
// Included are concrete implementations of some of the VMM interfaces included
// to ease serving simple client code or for testing and possibly as an
// example.  More details in the documentation of such implementations.
//
//  [1]: http://en.wikipedia.org/wiki/Database_model
package lldb

const (
	fltSz            = 0x70 // size of the FLT
	maxShort         = 251
	maxRq            = 65787
	maxFLTRq         = 4112
	maxHandle        = 1<<56 - 1
	atomLen          = 16
	tagUsedLong      = 0xfc
	tagUsedRelocated = 0xfd
	tagFreeShort     = 0xfe
	tagFreeLong      = 0xff
	tagNotCompressed = 0
	tagCompressed    = 1
)

// Content size n -> blocksize in atoms.
func n2atoms(n int) int {
	if n > maxShort {
		n += 2
	}
	return (n+1)/16 + 1
}

// Content size n -> number of padding zeros.
func n2padding(n int) int {
	if n > maxShort {
		n += 2
	}
	return 15 - (n+1)&15
}

// Handle <-> offset
func h2off(h int64) int64   { return (h + 6) * 16 }
func off2h(off int64) int64 { return off/16 - 6 }

// Get a 7B int64 from b
func b2h(b []byte) (h int64) {
	for _, v := range b[:7] {
		h = h<<8 | int64(v)
	}
	return
}

// Put a 7B int64 into b
func h2b(b []byte, h int64) []byte {
	for i := range b[:7] {
		b[i], h = byte(h>>48), h<<8
	}
	return b
}

// Content length N (must be in [252, 65787]) to long used block M field.
func n2m(n int) (m int) {
	return n % 0x10000
}

// Long used block M (must be in [0, 65535]) field to content length N.
func m2n(m int) (n int) {
	if m <= maxShort {
		m += 0x10000
	}
	return m
}

func bpack(a []byte) []byte {
	if cap(a) > len(a) {
		return append([]byte(nil), a...)
	}

	return a
}
