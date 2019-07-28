// Copyright 2018, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE.md file.

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
