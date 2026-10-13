// Copyright (c) 2015-2016 Dave Collins <dave@davec.name>
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

// NOTE: Due to the following build constraints, this file will only be compiled
// when the code is not running on Google App Engine, compiled by GopherJS, and
// "-tags safe" is not added to the go build command line.  The "disableunsafe"
// tag is deprecated and thus should not be used.
// Go versions prior to 1.4 are disabled because they use a different layout
// for interfaces which make the implementation of unsafeReflectValue more complex.
// +build !js,!appengine,!safe,!disableunsafe,go1.4

package spew

import (
	"reflect"
	"unsafe"
)

const (
	// UnsafeDisabled is a build-time constant which specifies whether or
	// not access to the unsafe package is available.
	UnsafeDisabled = false

	// ptrSize is the size of a pointer on the current arch.
	ptrSize = unsafe.Sizeof((*byte)(nil))
)

type flag uintptr

var (
	// flagRO indicates whether the value field of a reflect.Value
	// is read-only.
	flagRO flag

	// flagAddr indicates whether the address of the reflect.Value's
	// value may be taken.
	flagAddr flag
)

// flagKindMask holds the bits that make up the kind
// part of the flags field. In all the supported versions,
// it is in the lower 5 bits.
const flagKindMask = flag(0x1f)

// Different versions of Go have used different
// bit layouts for the flags type. This table
// records the known combinations.
var okFlags = []struct {
	ro, addr flag
}{{
	// From Go 1.4 to 1.5
	ro:   1 << 5,
	addr: 1 << 7,
}, {
	// Up to Go tip.
	ro:   1<<5 | 1<<6,
	addr: 1 << 8,
}}

var flagValOffset = func() uintptr {
	field, ok := reflect.TypeOf(reflect.Value{}).FieldByName("flag")
	if !ok {
		panic("reflect.Value has no flag field")
	}
	return field.Offset
}()

// flagField returns a pointer to the flag field of a reflect.Value.
func flagField(v *reflect.Value) *flag {
	return (*flag)(unsafe.Pointer(uintptr(unsafe.Pointer(v)) + flagValOffset))
}

// unsafeReflectValue converts the passed reflect.Value into a one that bypasses
// the typical safety restrictions preventing access to unaddressable and
// unexported data.  It works by digging the raw pointer to the underlying
// value out of the protected value and generating a new unprotected (unsafe)
// reflect.Value to it.
//
// This allows us to check for implementations of the Stringer and error
// interfaces to be used for pretty printing ordinarily unaddressable and
// inaccessible values such as unexported struct fields.
func unsafeReflectValue(v reflect.Value) reflect.Value {
	if !v.IsValid() || (v.CanInterface() && v.CanAddr()) {
		return v
	}
	flagFieldPtr := flagField(&v)
	*flagFieldPtr &^= flagRO
	*flagFieldPtr |= flagAddr
	return v
}

// Sanity checks against future reflect package changes
// to the type or semantics of the Value.flag field.
func init() {
	field, ok := reflect.TypeOf(reflect.Value{}).FieldByName("flag")
	if !ok {
		panic("reflect.Value has no flag field")
	}
	if field.Type.Kind() != reflect.TypeOf(flag(0)).Kind() {
		panic("reflect.Value flag field has changed kind")
	}
	type t0 int
	var t struct {
		A t0
		// t0 will have flagEmbedRO set.
		t0
		// a will have flagStickyRO set
		a t0
	}
	vA := reflect.ValueOf(t).FieldByName("A")
	va := reflect.ValueOf(t).FieldByName("a")
	vt0 := reflect.ValueOf(t).FieldByName("t0")

	// Infer flagRO from the difference between the flags
	// for the (otherwise identical) fields in t.
	flagPublic := *flagField(&vA)
	flagWithRO := *flagField(&va) | *flagField(&vt0)
	flagRO = flagPublic ^ flagWithRO

	// Infer flagAddr from the difference between a value
	// taken from a pointer and not.
	vPtrA := reflect.ValueOf(&t).Elem().FieldByName("A")
	flagNoPtr := *flagField(&vA)
	flagPtr := *flagField(&vPtrA)
	flagAddr = flagNoPtr ^ flagPtr

	// Check that the inferred flags tally with one of the known versions.
	for _, f := range okFlags {
		if flagRO == f.ro && flagAddr == f.addr {
			return
		}
	}
	panic("reflect.Value read-only flag has changed semantics")
}
