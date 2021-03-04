// +build !appengine

// This file encapsulates usage of unsafe.
// xxhash_safe.go contains the safe implementations.

package xxhash

import (
	"reflect"
	"unsafe"
)

// Notes:
//
// See https://groups.google.com/d/msg/golang-nuts/dcjzJy-bSpw/tcZYBzQqAQAJ
// for some discussion about these unsafe conversions.
//
// In the future it's possible that compiler optimizations will make these
// unsafe operations unnecessary: https://golang.org/issue/2205.
//
// Both of these wrapper functions still incur function call overhead since they
// will not be inlined. We could write Go/asm copies of Sum64 and Digest.Write
// for strings to squeeze out a bit more speed. Mid-stack inlining should
// eventually fix this.

// Sum64String computes the 64-bit xxHash digest of s.
// It may be faster than Sum64([]byte(s)) by avoiding a copy.
func Sum64String(s string) uint64 {
	var b []byte
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	bh.Data = (*reflect.StringHeader)(unsafe.Pointer(&s)).Data
	bh.Len = len(s)
	bh.Cap = len(s)
	return Sum64(b)
}

// WriteString adds more data to d. It always returns len(s), nil.
// It may be faster than Write([]byte(s)) by avoiding a copy.
func (d *Digest) WriteString(s string) (n int, err error) {
	var b []byte
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	bh.Data = (*reflect.StringHeader)(unsafe.Pointer(&s)).Data
	bh.Len = len(s)
	bh.Cap = len(s)
	return d.Write(b)
}
