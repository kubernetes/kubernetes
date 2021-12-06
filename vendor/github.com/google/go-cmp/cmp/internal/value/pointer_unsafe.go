// Copyright 2018, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego
// +build !purego

package value

import (
	"reflect"
	"unsafe"
)

// Pointer is an opaque typed pointer and is guaranteed to be comparable.
type Pointer struct {
	p unsafe.Pointer
	t reflect.Type
}

// PointerOf returns a Pointer from v, which must be a
// reflect.Ptr, reflect.Slice, or reflect.Map.
func PointerOf(v reflect.Value) Pointer {
	// The proper representation of a pointer is unsafe.Pointer,
	// which is necessary if the GC ever uses a moving collector.
	return Pointer{unsafe.Pointer(v.Pointer()), v.Type()}
}

// IsNil reports whether the pointer is nil.
func (p Pointer) IsNil() bool {
	return p.p == nil
}

// Uintptr returns the pointer as a uintptr.
func (p Pointer) Uintptr() uintptr {
	return uintptr(p.p)
}
