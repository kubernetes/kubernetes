// Copyright 2018, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build purego

package value

import "reflect"

// Pointer is an opaque typed pointer and is guaranteed to be comparable.
type Pointer struct {
	p uintptr
	t reflect.Type
}

// PointerOf returns a Pointer from v, which must be a
// reflect.Ptr, reflect.Slice, or reflect.Map.
func PointerOf(v reflect.Value) Pointer {
	// NOTE: Storing a pointer as an uintptr is technically incorrect as it
	// assumes that the GC implementation does not use a moving collector.
	return Pointer{v.Pointer(), v.Type()}
}

// IsNil reports whether the pointer is nil.
func (p Pointer) IsNil() bool {
	return p.p == 0
}

// Uintptr returns the pointer as a uintptr.
func (p Pointer) Uintptr() uintptr {
	return p.p
}
