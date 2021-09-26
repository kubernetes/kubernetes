// +build !appengine

// This file encapsulates usage of unsafe.
// xxhash_safe.go contains the safe implementations.

package xxhash

import (
	"reflect"
	"unsafe"
)

// Sum64String computes the 64-bit xxHash digest of s.
// It may be faster than Sum64([]byte(s)) by avoiding a copy.
//
// TODO(caleb): Consider removing this if an optimization is ever added to make
// it unnecessary: https://golang.org/issue/2205.
//
// TODO(caleb): We still have a function call; we could instead write Go/asm
// copies of Sum64 for strings to squeeze out a bit more speed.
func Sum64String(s string) uint64 {
	// See https://groups.google.com/d/msg/golang-nuts/dcjzJy-bSpw/tcZYBzQqAQAJ
	// for some discussion about this unsafe conversion.
	var b []byte
	bh := (*reflect.SliceHeader)(unsafe.Pointer(&b))
	bh.Data = (*reflect.StringHeader)(unsafe.Pointer(&s)).Data
	bh.Len = len(s)
	bh.Cap = len(s)
	return Sum64(b)
}
