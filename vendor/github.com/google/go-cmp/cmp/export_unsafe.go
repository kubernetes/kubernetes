// Copyright 2017, The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego
// +build !purego

package cmp

import (
	"reflect"
	"unsafe"
)

const supportExporters = true

// retrieveUnexportedField uses unsafe to forcibly retrieve any field from
// a struct such that the value has read-write permissions.
//
// The parent struct, v, must be addressable, while f must be a StructField
// describing the field to retrieve. If addr is false,
// then the returned value will be shallowed copied to be non-addressable.
func retrieveUnexportedField(v reflect.Value, f reflect.StructField, addr bool) reflect.Value {
	ve := reflect.NewAt(f.Type, unsafe.Pointer(uintptr(unsafe.Pointer(v.UnsafeAddr()))+f.Offset)).Elem()
	if !addr {
		// A field is addressable if and only if the struct is addressable.
		// If the original parent value was not addressable, shallow copy the
		// value to make it non-addressable to avoid leaking an implementation
		// detail of how forcibly exporting a field works.
		if ve.Kind() == reflect.Interface && ve.IsNil() {
			return reflect.Zero(f.Type)
		}
		return reflect.ValueOf(ve.Interface()).Convert(f.Type)
	}
	return ve
}
